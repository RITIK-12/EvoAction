"""Gate 1: live Crusoe preflight + real WGO-Bench episode access.

Writes a redacted artifacts/preflight.json. Never prints credentials.
Usage: uv run python scripts/preflight.py [--episode robointer_droid_000010] [--alt-vision MODEL ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evoaction.clients.crusoe import ROLE_NEMOTRON, ROLE_VISION, CrusoeClient, image_part  # noqa: E402
from evoaction.config import get_settings  # noqa: E402
from evoaction.data.wgo import build_index, ingest_episode, load_episode  # noqa: E402
from evoaction.media.contact_sheets import build_sheets, uniform_timestamps  # noqa: E402
from evoaction.persistence import get_store  # noqa: E402


def check(name: str, fn, report: dict) -> object:
    t0 = time.time()
    try:
        out = fn()
        report["checks"][name] = {"ok": True, "latency_s": round(time.time() - t0, 2), **(out if isinstance(out, dict) else {})}
        print(f"[ok]   {name}")
        return out
    except Exception as e:  # report and continue with independent checks
        report["checks"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:800], "latency_s": round(time.time() - t0, 2)}
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:300]}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="robointer_droid_000010")
    ap.add_argument("--alt-vision", nargs="*", default=["nvidia/NVIDIA-Nemotron-3.5-Super-VL"])
    args = ap.parse_args()

    s = get_settings()
    report: dict = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "api_base": s.api_base,
                    "api_key_present": bool(s.api_key), "configured": {"reasoning": s.reasoning_model, "vision": s.vision_model},
                    "checks": {}}
    store = get_store()
    client = CrusoeClient(s, store)
    run_id = store.start_run("preflight")
    client.run_id = run_id

    # 1. model discovery
    def discover():
        models = client.list_models()
        return {"models": [{"id": m["id"], "type": m.get("type"), "tags": m.get("tags"), "pricing_per_mtok": m.get("pricing"),
                            "context_length": m.get("context_length")} for m in models]}
    check("models_endpoint", discover, report)

    def resolve():
        r = client.nemotron_model
        v = client.vision_model
        return {"reasoning_model": r, "vision_model": v,
                "reasoning_tags": client.model_info(r).get("tags"), "vision_tags": client.model_info(v).get("tags")}
    resolved = check("resolve_models", resolve, report)
    if resolved:
        report["locked"] = {"CRUSOE_REASONING_MODEL": resolved["reasoning_model"], "CRUSOE_VISION_MODEL": resolved["vision_model"]}

    # 2. Nemotron structured JSON
    def nemotron_json():
        res = client.chat(ROLE_NEMOTRON, [
            {"role": "system", "content": "You are a precise planner. Reply with a single JSON object only."},
            {"role": "user", "content": "Return JSON with keys: ok (true), role (string 'supervisor'), "
                                        "next_steps (list of exactly 2 short strings for testing an annotation workflow)."},
        ], purpose="preflight_nemotron_json", json_mode=True, max_tokens=2048, use_cache=False)
        assert isinstance(res.data, dict) and res.data.get("ok") is True, res.text[:300]
        return {"model": res.model, "parsed": res.data, "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                "reasoning_tokens": res.reasoning_tokens, "cost_usd": round(res.cost_usd, 6), "call_latency_s": round(res.latency_s, 2)}
    check("nemotron_structured_json", nemotron_json, report)

    # 3. tools
    check("ffmpeg_ffprobe", lambda: {"ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe")}
          if shutil.which("ffmpeg") and shutil.which("ffprobe") else (_ for _ in ()).throw(RuntimeError("ffmpeg/ffprobe missing")), report)

    # 4. real WGO episode (agent-visible part only is reported)
    def episode():
        build_index(s)
        ingest_episode(args.episode, s)
        ep = load_episode(args.episode, s)
        gold_exists = (s.protected_dir / "gold" / f"{args.episode}.json").exists()
        return {"episode_id": ep["id"], "source": ep["source"], "duration_sec": ep["duration_sec"], "video_fps": ep["video_fps"],
                "resolution": [ep["width"], ep["height"]], "has_telemetry": ep["has_telemetry"],
                "reference_labels_present_in_protected_store": gold_exists, "video_sha256": ep["video_sha256"]}
    ep_info = check("wgo_episode", episode, report)

    # 5. multimodal image requests on a real contact sheet
    if ep_info:
        ep = load_episode(args.episode, s)
        sheets = build_sheets(ep["video_path"], uniform_timestamps(ep["duration_sec"], 1.0), tile_px=224, frames_per_sheet=20)
        out_dir = s.artifacts_dir / "preflight"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "contact_sheet_0.jpg").write_bytes(sheets[0].jpeg)
        half = sheets[0].timestamps
        two = build_sheets(ep["video_path"], uniform_timestamps(ep["duration_sec"], 2.0), tile_px=224, frames_per_sheet=10)

        def vision(model: str, sheet_list):
            prompt = ("These are contact sheets of a robot manipulation video. Each tile has its timestamp in seconds in yellow. "
                      f"Task instruction: {ep['instruction']!r}. Return a JSON object with keys: "
                      "timestamps_read (list of up to 5 timestamps you can read), "
                      "events (list of {start_sec, end_sec, subtask} for completed manipulation subtasks).")
            content = [{"type": "text", "text": prompt}] + [image_part(sh.jpeg) for sh in sheet_list]
            res = client.chat(ROLE_VISION, [{"role": "user", "content": content}], purpose="preflight_vision", model=model,
                              json_mode=True, max_tokens=4096, use_cache=False)
            assert isinstance(res.data, dict), res.text[:300]
            return {"model": res.model, "images": len(sheet_list), "parsed": res.data, "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens, "reasoning_tokens": res.reasoning_tokens,
                    "cost_usd": round(res.cost_usd, 6), "call_latency_s": round(res.latency_s, 2)}

        vm = resolved["vision_model"] if resolved else s.vision_model
        report["contact_sheet"] = {"timestamps": half, "sha256": sheets[0].sha256, "path": "artifacts/preflight/contact_sheet_0.jpg"}
        check(f"vision_single_image[{vm}]", lambda: vision(vm, sheets[:1]), report)
        check(f"vision_multi_image[{vm}]", lambda: vision(vm, two[:3]), report)
        for alt in args.alt_vision:
            try:
                alt_id = client.resolve_model(alt)
            except Exception as e:
                report["checks"][f"vision_single_image[{alt}]"] = {"ok": False, "error": str(e)[:300]}
                continue
            if alt_id != vm:
                check(f"vision_single_image[{alt_id}]", lambda a=alt_id: vision(a, sheets[:1]), report)

    report["exit_criterion_met"] = all(
        report["checks"].get(k, {}).get("ok") for k in ("nemotron_structured_json", "wgo_episode")
    ) and any(v.get("ok") for k, v in report["checks"].items() if k.startswith("vision_single_image"))
    usage = store.query("SELECT role, model, COUNT(*) n, SUM(prompt_tokens) pt, SUM(completion_tokens) ct, SUM(cost_usd) cost "
                        "FROM model_calls WHERE run_id = ? GROUP BY role, model", (run_id,))
    report["usage"] = usage
    s.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (s.artifacts_dir / "preflight.json").write_text(json.dumps(report, indent=1, default=str))
    store.finish_run(run_id, "ok" if report["exit_criterion_met"] else "failed")
    print(f"\nexit_criterion_met={report['exit_criterion_met']}  -> artifacts/preflight.json")
    return 0 if report["exit_criterion_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
