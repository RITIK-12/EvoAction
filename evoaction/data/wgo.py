"""WGO-Bench ingest with strict gold separation.

The upstream parquet stores one episode per row group, so single episodes can be
range-read without downloading the full ~1.4 GB file.

On ingest every row is split into:
  * agent-visible files under `episodes/<id>/` (video.mp4, episode.json, telemetry.json)
  * evaluator-only reference labels under `protected/gold/<id>.json`
Only `evoaction.evaluation` reads the protected directory.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from evoaction.config import Settings, get_settings

REPO = "macrodata/WGO-Bench"
PARQUET = f"datasets/{REPO}/data/annotations.parquet"
LIGHT_COLUMNS = ["id", "instruction", "segments", "metadata"]


def _source(episode_id: str) -> str:
    if episode_id.startswith("robointer_droid"):
        return "droid"
    if episode_id.startswith("galaxea"):
        return "galaxea"
    return "homer"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_index(settings: Settings | None = None, refresh: bool = False) -> list[dict[str, Any]]:
    """Read light columns for all rows. Public index has no gold; gold summary goes to protected/."""
    s = settings or get_settings()
    public_path = s.cache_dir / "wgo_index.json"
    protected_path = s.protected_dir / "wgo_gold_index.json"
    if public_path.exists() and protected_path.exists() and not refresh:
        return json.loads(public_path.read_text())
    fs = HfFileSystem()
    with fs.open(PARQUET, "rb") as f:
        pf = pq.ParquetFile(f)
        md = pf.metadata
        rows = pf.read(columns=LIGHT_COLUMNS).to_pylist()
        sizes = [md.row_group(i).total_byte_size for i in range(md.num_row_groups)]
    public, protected = [], {}
    for i, r in enumerate(rows):
        meta = json.loads(r["metadata"]) if r["metadata"] else {}
        public.append({
            "id": r["id"], "row_group": i, "source": _source(r["id"]), "instruction": r["instruction"] or "",
            "bytes": sizes[i], "duration_sec": meta.get("duration_sec"), "fps": meta.get("fps"),
        })
        protected[r["id"]] = r["segments"] or []
    s.cache_dir.mkdir(parents=True, exist_ok=True)
    s.protected_dir.mkdir(parents=True, exist_ok=True)
    public_path.write_text(json.dumps(public, indent=1))
    protected_path.write_text(json.dumps(protected))
    return public


def ingest_episode(episode_id: str, settings: Settings | None = None) -> Path:
    """Range-read one episode row; write agent-visible media and protected gold."""
    s = settings or get_settings()
    index = {e["id"]: e for e in build_index(s)}
    entry = index[episode_id]
    ep_dir = s.episodes_dir / episode_id
    video_path = ep_dir / "video.mp4"
    gold_path = s.protected_dir / "gold" / f"{episode_id}.json"
    if video_path.exists() and (ep_dir / "episode.json").exists() and gold_path.exists():
        return ep_dir
    fs = HfFileSystem()
    with fs.open(PARQUET, "rb") as f:
        pf = pq.ParquetFile(f)
        row = pf.read_row_group(entry["row_group"]).to_pylist()[0]
    assert row["id"] == episode_id, (row["id"], episode_id)
    ep_dir.mkdir(parents=True, exist_ok=True)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(row["video"])
    probe = ffprobe(video_path)
    meta = json.loads(row["metadata"]) if row["metadata"] else {}
    episode = {
        "id": episode_id,
        "source": entry["source"],
        "instruction": row["instruction"] or meta.get("original_instruction", ""),
        "duration_sec": probe["duration_sec"],
        "video_fps": probe["fps"],
        "width": probe["width"],
        "height": probe["height"],
        "has_telemetry": row["perception_state"] is not None,
        "video_sha256": sha256_file(video_path),
        "source_dataset": meta.get("source_dataset"),
        "source_episode_id": meta.get("source_episode_id"),
    }
    (ep_dir / "episode.json").write_text(json.dumps(episode, indent=1))
    if row["perception_state"] is not None:
        (ep_dir / "telemetry.json").write_text(json.dumps(row["perception_state"]))
    gold_path.write_text(json.dumps({"episode_id": episode_id, "segments": row["segments"]}, indent=1))
    return ep_dir


def ffprobe(path: Path) -> dict[str, Any]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,nb_frames:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    d = json.loads(out.stdout)
    st = d["streams"][0]
    num, den = (st.get("avg_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    return {"duration_sec": float(d["format"]["duration"]), "fps": fps, "width": st["width"], "height": st["height"]}


def load_episode(episode_id: str, settings: Settings | None = None) -> dict[str, Any]:
    """Agent-visible episode record (no gold)."""
    s = settings or get_settings()
    ep_dir = s.episodes_dir / episode_id
    ep = json.loads((ep_dir / "episode.json").read_text())
    ep["video_path"] = str(ep_dir / "video.mp4")
    ep["telemetry_path"] = str(ep_dir / "telemetry.json") if (ep_dir / "telemetry.json").exists() else None
    return ep


# -- splits --------------------------------------------------------------

def _norm_instruction(text: str) -> str:
    t = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    t = re.sub(r"\b(on the green table|with the right gripper|with the left gripper)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def instruction_groups(entries: list[dict[str, Any]], threshold: float = 0.75) -> dict[str, str]:
    """Union near-duplicate instructions (same task family) into one leakage group."""
    ids = [e["id"] for e in entries]
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    norm = {e["id"]: (e["source"], _norm_instruction(e["instruction"])) for e in entries}
    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1 :]:
            (sa, ta), (sb, tb) = norm[a], norm[b]
            if sa == sb and difflib.SequenceMatcher(None, ta, tb).ratio() >= threshold:
                parent[find(a)] = find(b)
    roots = {}
    out = {}
    for i in ids:
        r = find(i)
        roots.setdefault(r, f"G{len(roots):02d}-{norm[r][0]}")
        out[i] = roots[r]
    return out


def make_splits(
    settings: Settings | None = None,
    seed: int = 17,
    sizes: dict[str, int] | None = None,
    pilot: dict[str, int] | None = None,
    max_duration: float = 60.0,
    max_bytes: int = 30_000_000,
    min_segments: int = 3,
    per_group: int = 2,
) -> dict[str, Any]:
    """Experimenter-side split design (uses gold segment counts; never agent-visible).

    Robot episodes only (telemetry available), duration/size capped for cost, at least
    `min_segments` reference subtasks. Whole instruction groups go to one split.
    """
    s = settings or get_settings()
    sizes = sizes or {"test": 6, "validation": 4, "dev": 5}
    pilot = pilot or {"dev": 2, "validation": 2, "test": 3}
    index = build_index(s)
    gold_counts = json.loads((s.protected_dir / "wgo_gold_index.json").read_text())
    pool = [
        e for e in index
        if e["source"] in ("droid", "galaxea")
        and (e["duration_sec"] or 1e9) <= max_duration
        and e["bytes"] <= max_bytes
        and len(gold_counts[e["id"]]) >= min_segments
    ]
    groups = instruction_groups(pool)
    by_group: dict[str, list[str]] = {}
    for e in pool:
        by_group.setdefault(groups[e["id"]], []).append(e["id"])
    rng = random.Random(seed)
    per_source: dict[str, list[str]] = {}
    for g in sorted(by_group):
        per_source.setdefault(g.split("-")[1], []).append(g)
    for src in per_source.values():
        rng.shuffle(src)
    # Interleave sources so each split mixes DROID and Galaxea.
    ordered: list[str] = []
    srcs = sorted(per_source)
    while any(per_source[x] for x in srcs):
        for x in srcs:
            if per_source[x]:
                ordered.append(per_source[x].pop(0))
    assignment: dict[str, list[str]] = {k: [] for k in sizes}
    group_of_split: dict[str, list[str]] = {k: [] for k in sizes}
    queue = list(ordered)
    # Fill splits round-robin by group so every split gets both sources early.
    while queue and any(len(assignment[k]) < sizes[k] for k in sizes):
        for k in sizes:
            need = sizes[k] - len(assignment[k])
            if need <= 0 or not queue:
                continue
            g = queue.pop(0)
            members = sorted(by_group[g])
            rng.shuffle(members)
            take = members[: min(per_group, need)]
            assignment[k].extend(take)
            group_of_split[k].append(g)
    splits: dict[str, Any] = {"seed": seed, "protocol": "wgo-temporal-f1@0.75-v1", "selection": {
        "sources": ["droid", "galaxea"], "max_duration_sec": max_duration, "max_bytes": max_bytes,
        "min_reference_segments": min_segments, "max_episodes_per_group": per_group}, "splits": {}}
    for k, ids in assignment.items():
        splits["splits"][k] = {
            "episode_ids": ids,
            "pilot_episode_ids": ids[: pilot.get(k, len(ids))],
            "groups": {i: groups[i] for i in ids},
        }
    return splits


def write_splits(splits: dict[str, Any], settings: Settings | None = None) -> Path:
    """Ingest every split episode, add hashes, and write immutable artifacts/splits.json."""
    s = settings or get_settings()
    path = s.artifacts_dir / "splits.json"
    if path.exists():
        raise FileExistsError(f"{path} already exists; splits are immutable once written")
    hashes = {}
    for k, sp in splits["splits"].items():
        for eid in sp["episode_ids"]:
            ingest_episode(eid, s)
            hashes[eid] = load_episode(eid, s)["video_sha256"]
    splits["episode_video_sha256"] = hashes
    body = json.dumps(splits, indent=1, sort_keys=True)
    splits_hash = hashlib.sha256(body.encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**splits, "splits_sha256": splits_hash}, indent=1, sort_keys=True))
    return path


def load_splits(settings: Settings | None = None) -> dict[str, Any]:
    s = settings or get_settings()
    return json.loads((s.artifacts_dir / "splits.json").read_text())


def split_ids(split: str, pilot: bool, settings: Settings | None = None) -> list[str]:
    sp = load_splits(settings)["splits"][split]
    return sp["pilot_episode_ids"] if pilot else sp["episode_ids"]
