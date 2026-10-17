"""Live demo: one episode's video, the human reference bar, and one bar per harness version as each finishes.

  python -m evoaction demo [--port 8765]

Each version (N0, then every recursion's candidate in a lineage) is executed through the real workflow engine on the
chosen episode, so a bar appears when that version's labeling is done (cached calls return instantly; uncached
ones call Nemotron live). Reference labels are read here only for display and scoring in this human-facing viewer;
no agent code path is involved.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import mimetypes
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from evoaction.config import ROOT, get_settings
from evoaction.evaluation import segments as M

PAGE = Path(__file__).with_name("demo_index.html")
LINEAGES = {"hitl": ("_hitl", "Human-in-the-loop (10% labeled sample)"), "label_free": ("", "Label-free (no labels)")}
_lock = threading.Lock()


def _settings():
    return get_settings()


def _set_of(episode_id: str) -> str:
    return "droid50" if episode_id.startswith("robointer_droid") else "rest50"


def _lineage(set_name: str, kind: str) -> dict[str, Any]:
    sfx = ("" if set_name == "droid50" else f"_{set_name}") + LINEAGES[kind][0]
    p = _settings().artifacts_dir / f"lineage{sfx}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def episodes() -> list[dict[str, Any]]:
    from evoaction.data.wgo import calibration_split, episode_set_ids, load_episode

    out = []
    for name in ("droid50", "rest50"):
        try:
            labeled = set(calibration_split(name)[0])
        except FileNotFoundError:
            labeled = set()
        for e in episode_set_ids(name):
            ep = load_episode(e)
            out.append({"id": e, "set": name, "source": ep["source"], "duration_sec": ep["duration_sec"],
                        "instruction": ep["instruction"], "labeled_sample": e in labeled})
    return out


def episode_detail(episode_id: str) -> dict[str, Any]:
    from evoaction.data.wgo import load_episode

    ep = load_episode(episode_id)
    gold = json.loads((_settings().protected_dir / "gold" / f"{episode_id}.json").read_text())["segments"]
    return {"id": episode_id, "instruction": ep["instruction"], "duration_sec": ep["duration_sec"], "source": ep["source"],
            "reference": [{"start_sec": g["start_sec"], "end_sec": g["end_sec"], "subtask": g.get("subtask", "")} for g in gold]}


def versions_for(set_name: str, kind: str) -> list[dict[str, Any]]:
    lin = _lineage(set_name, kind)
    out = [{"version_id": "N0", "label": "Recursion 0 · N0 (Macrodata protocol on Nemotron Omni)", "decision": "baseline"}]
    for k, e in enumerate(lin.get("experiments", []), start=1):
        if not e.get("version_dir"):
            continue
        out.append({"version_id": e["version_dir"], "label": f"Recursion {k} · {e['id']}", "decision": e.get("decision"),
                    "hypothesis": e.get("hypothesis")})
    return out


def score(episode_id: str, preds: list[dict[str, Any]], refs: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    norm, _ = M.normalize_predictions(preds, duration)
    snapped = M.outer_snap(norm, refs)
    matched = {i for i, _, _ in M.match(snapped, refs, M.PRIMARY_IOU)}
    s_snap = M.score_episode(episode_id, preds, refs, duration, snap=True)
    s_raw = M.score_episode(episode_id, preds, refs, duration)
    return {"segments": [{**p, "matched": i in matched} for i, p in enumerate(snapped)],
            "snapped_f1_075": s_snap.f1, "raw_f1_075": s_raw.f1, "n_pred": len(norm), "n_ref": len(refs)}


def run_version_on_episode(version_id: str, episode_id: str, client, store, run_id: str) -> dict[str, Any]:
    from evoaction.workflow.engine import VersionStore, run_episode

    s = _settings()
    v = VersionStore(s, store).load(version_id)
    demo_settings = dataclasses.replace(s, artifacts_dir=ROOT / "data" / "demo_runs")  # keep demo outputs out of artifacts/
    t0 = time.time()
    r = run_episode(v, episode_id, client, run_id, store, demo_settings)
    rows = store.query("SELECT COUNT(*) n, COALESCE(SUM(cache_hit),0) ch FROM model_calls WHERE run_id=? AND version_id=? AND episode_id=?",
                       (run_id, version_id, episode_id))
    return {"segments": r["segments"], "status": r["status"], "wall_s": round(time.time() - t0, 1),
            "calls": rows[0]["n"] if rows else 0, "cache_hits": rows[0]["ch"] if rows else 0, "nodes": [n.operator for n in v.workflow.nodes]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # quiet
        pass

    def _json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/api/episodes":
                self._json(episodes())
            elif u.path == "/api/episode":
                eid = q["id"]
                self._json({**episode_detail(eid), "versions": {k: versions_for(_set_of(eid), k) for k in LINEAGES},
                            "lineages": {k: v[1] for k, v in LINEAGES.items()}})
            elif u.path.startswith("/video/"):
                self._video(u.path.split("/video/", 1)[1])
            elif u.path == "/api/run":
                self._run(q["id"], q.get("lineage", "hitl"))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # surface errors to the page instead of hanging it
            try:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            except Exception:
                pass

    def _video(self, episode_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_]+", episode_id):
            self.send_error(HTTPStatus.BAD_REQUEST)
            return
        path = _settings().episodes_dir / episode_id / "video.mp4"
        size = path.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else 0
            end = min(int(b), size - 1) if b else size - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 16, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def _event(self, obj: dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps(obj, default=str)}\n\n".encode())
        self.wfile.flush()

    def _run(self, episode_id: str, kind: str) -> None:
        from evoaction.clients.crusoe import CrusoeClient
        from evoaction.persistence import get_store

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        detail = episode_detail(episode_id)
        store = get_store()
        client = CrusoeClient(_settings(), store)
        run_id = store.start_run("demo", {"episode": episode_id, "lineage": kind})
        client.run_id = run_id
        vs = versions_for(_set_of(episode_id), kind)
        self._event({"type": "plan", "versions": vs})
        for v in vs:
            self._event({"type": "start", "version_id": v["version_id"]})
            try:
                r = run_version_on_episode(v["version_id"], episode_id, client, store, run_id)
                sc = score(episode_id, r["segments"], detail["reference"], detail["duration_sec"])
                self._event({"type": "result", **v, **r, **sc})
            except Exception as e:
                self._event({"type": "error", "version_id": v["version_id"], "error": f"{type(e).__name__}: {str(e)[:300]}"})
        store.finish_run(run_id, "ok")
        self._event({"type": "done"})


def cmd_demo(args: argparse.Namespace) -> int:
    port = getattr(args, "port", None) or 8765
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"EvoAction demo on http://127.0.0.1:{port}")
    srv.serve_forever()
    return 0
