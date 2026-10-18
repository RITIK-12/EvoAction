"""Live demo: one episode's video, the human reference bar, and a live N0 -> N1 -> N2 self-improvement run.

  python -m evoaction demo [--port 8765]

Every click runs its own recursion, independent of the batch experiments: N0 (Macrodata protocol) labels the episode
and the batch's fixed 10% human-labeled sample; the main agent (Nemotron Super) reads the root-cause analysis of that
sample and patches the harness into N1; N1 runs the same way; the agent improves N1 into N2. All model calls are live
(cache reads off). The demo writes only to data/demo_runs/ (its own SQLite DB, events and outputs) and never to
artifacts/. Reference labels of the shown episode are used only to draw and score its bars.
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
_lock = threading.Lock()


def _settings():
    return get_settings()


def _set_of(episode_id: str) -> str:
    return "droid50" if episode_id.startswith("robointer_droid") else "rest50"


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


def score(episode_id: str, preds: list[dict[str, Any]], refs: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    norm, _ = M.normalize_predictions(preds, duration)
    snapped = M.outer_snap(norm, refs)
    matched = {i for i, _, _ in M.match(snapped, refs, M.PRIMARY_IOU)}
    s_snap = M.score_episode(episode_id, preds, refs, duration, snap=True)
    s_raw = M.score_episode(episode_id, preds, refs, duration)
    return {"segments": [{**p, "matched": i in matched} for i, p in enumerate(snapped)],
            "snapped_f1_075": s_snap.f1, "raw_f1_075": s_raw.f1, "n_pred": len(norm), "n_ref": len(refs)}


DEMO_DIR = ROOT / "data" / "demo_runs"
_demo_store = None


def demo_store():
    """Separate SQLite DB + event log so the demo never touches the experiment artifacts."""
    global _demo_store
    if _demo_store is None:
        from evoaction.persistence import Store

        _demo_store = Store(DEMO_DIR / "demo.db", DEMO_DIR / "events.jsonl")
    return _demo_store


def demo_settings():
    return dataclasses.replace(_settings(), artifacts_dir=DEMO_DIR)


def run_on(version, episode_ids: list[str], client, run_id: str) -> dict[str, Any]:
    from evoaction.workflow.engine import run_version

    return run_version(version, episode_ids, client, run_id, store=demo_store(), settings=demo_settings(), workers=6)


def calls_of(run_id: str, version_id: str, episode_id: str) -> int:
    rows = demo_store().query("SELECT COUNT(*) n FROM model_calls WHERE run_id=? AND version_id=? AND episode_id=?", (run_id, version_id, episode_id))
    return rows[0]["n"] if rows else 0


def patch_summary(patch: dict[str, Any] | None) -> str:
    patch = patch or {}
    parts = []
    for ins in patch.get("insert_nodes") or []:
        node = ins.get("node", ins) if isinstance(ins, dict) else {}
        parts.append(f"+{node.get('operator', '?')}")
    if patch.get("remove_nodes"):
        parts.append("-" + ",-".join(patch["remove_nodes"]))
    if patch.get("set_params"):
        parts.append("params: " + ", ".join(f"{n}.{k}" for n, kv in patch["set_params"].items() for k in (kv or {})))
    if (patch.get("prompt_updates") or {}).get("prompt.md"):
        parts.append("prompt rewritten")
    return "; ".join(parts) or "no change"


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
                self._json(episode_detail(eid))
            elif u.path.startswith("/video/"):
                self._video(u.path.split("/video/", 1)[1])
            elif u.path == "/api/run":
                self._run(q["id"])
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

    def _run(self, episode_id: str) -> None:
        from concurrent.futures import ThreadPoolExecutor

        from evoaction.agents.supervisor import engineer_patch
        from evoaction.clients.crusoe import CrusoeClient
        from evoaction.data.wgo import calibration_split
        from evoaction.experiments.evolution import root_cause_analysis
        from evoaction.schemas import WorkflowPatch
        from evoaction.workflow.engine import Version, VersionStore, apply_patch

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        detail = episode_detail(episode_id)
        store = demo_store()
        client = CrusoeClient(_settings(), store)
        client.read_cache = False  # live: every call goes to Crusoe
        run_id = store.start_run("demo", {"episode": episode_id, "live": True})
        client.run_id = run_id
        labeled = calibration_split(_set_of(episode_id))[0]
        version = VersionStore(_settings(), store).load("N0")
        history: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        pool = ThreadPoolExecutor(2)
        for k in range(3):
            name = f"N{k}"
            self._event({"type": "start", "label": name, "phase": f"{name} labeling the episode (live Crusoe calls)"})
            t0 = time.time()
            fut_ep = pool.submit(run_on, version, [episode_id], client, run_id)
            fut_lab = pool.submit(run_on, version, labeled, client, run_id) if k < 2 else None
            try:
                r = fut_ep.result()["episodes"][episode_id]
                sc = score(episode_id, r["segments"], detail["reference"], detail["duration_sec"])
                res = {"type": "result", "label": name, "status": r["status"], "wall_s": round(time.time() - t0, 1),
                       "calls": calls_of(run_id, version.id, episode_id), "nodes": [n.operator for n in version.workflow.nodes],
                       "change": results[-1]["next_change"] if results else "Macrodata protocol baseline", **sc}
                results.append({**res, "next_change": None})
                self._event(res)
            except Exception as e:
                self._event({"type": "error", "label": name, "error": f"{type(e).__name__}: {str(e)[:300]}"})
                break
            if fut_lab is None:
                break
            self._event({"type": "phase", "phase": f"Main agent reading {name}'s errors on the 10% labeled sample ({len(labeled)} episodes)"})
            try:
                analysis = root_cause_analysis({"run": fut_lab.result()}, labeled)
                data, wf, prompts, err = {}, None, None, None
                for attempt in range(2):
                    data = engineer_patch(client, version, analysis, history, {}, [{"error": err}] if err else [], f"demo_engineer_N{k + 1}")
                    try:
                        wf, prompts = apply_patch(version.workflow, version.prompts, WorkflowPatch(**(data.get("candidate_patch") or {})))
                        break
                    except Exception as e:
                        err = f"invalid patch: {str(e)[:500]}"
                change = patch_summary(data.get("candidate_patch")) if wf is not None else "agent patch invalid; unchanged"
                results[-1]["next_change"] = change
                history.append({"version": version.id, "labeled_scores": analysis["labeled_sample"]["pooled"], "hypothesis": data.get("hypothesis"),
                                "patch_summary": change})
                self._event({"type": "agent", "from": name, "to": f"N{k + 1}", "change": change, "hypothesis": data.get("hypothesis"),
                             "root_causes": data.get("root_causes"), "labeled_f1": analysis["labeled_sample"]["pooled"]["snapped_f1_075"]})
                if wf is not None:
                    version = Version(id=f"demo-{run_id}-N{k + 1}", path=version.path, workflow=wf, prompts=prompts, config=version.config,
                                      manifest={"workflow_sha256": "demo"})
            except Exception as e:
                self._event({"type": "phase", "phase": f"agent step failed ({type(e).__name__}); N{k + 1} keeps {name}'s harness"})
                results[-1]["next_change"] = "agent step failed; unchanged"
        pool.shutdown(wait=False)
        store.finish_run(run_id, "ok")
        top = max(results, key=lambda x: (x["snapped_f1_075"], x["raw_f1_075"])) if results else None
        best = top["label"] if top and top["snapped_f1_075"] > 0 else None  # no "best" tag when every version scores 0
        self._event({"type": "done", "best": best})


def cmd_demo(args: argparse.Namespace) -> int:
    port = getattr(args, "port", None) or 8765
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"EvoAction demo on http://127.0.0.1:{port}")
    srv.serve_forever()
    return 0
