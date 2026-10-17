"""EvoAction CLI (protocol v3: all 50 WGO-Bench DROID episodes, Macrodata Refiner protocol replicated on Nemotron).

  python -m evoaction setup                   ingest all 100 episodes from data/wgo, lock batch sets droid50 and rest50
  python -m evoaction baseline [--set S]      run version N0 on a batch (label-free self-score + GT headline)
  python -m evoaction evolve [--set S]        N -> N+1 label-free recursive improvement; every version runs on the same batch
  (S = droid50, batch 1: all 50 DROID episodes [default]; rest50, batch 2: 25 Galaxea + 25 HomER)
  python -m evoaction report       build artifacts/report.html from persisted artifacts
  python -m evoaction run-demo     baseline -> evolve -> report
"""

from __future__ import annotations

import argparse
import json
import sys

from evoaction.config import get_settings


def cmd_setup(args: argparse.Namespace) -> int:
    from evoaction.data.wgo import EPISODE_SETS, SETS_DIR, ingest_all, write_episode_set

    s = get_settings()
    ids = ingest_all(s)
    print(f"ingested {len(ids)} episodes")
    for name in EPISODE_SETS:
        if not (s.artifacts_dir / SETS_DIR / f"{name}.json").exists():
            print(write_episode_set(name, s))
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    from evoaction.clients.crusoe import CrusoeClient
    from evoaction.data.wgo import episode_set_ids
    from evoaction.evaluation.protected import ProtectedEvaluator
    from evoaction.experiments.evolution import version_record
    from evoaction.persistence import get_store
    from evoaction.pipeline import ensure_n0, run_and_assess
    from evoaction.workflow.engine import VersionStore

    s = get_settings()
    store = get_store()
    client = CrusoeClient(s, store)
    set_name = args.set or "droid50"
    ids = episode_set_ids(set_name, s)
    run_id = store.start_run("baseline_v3", {"episode_set": set_name, "episodes": len(ids)})
    client.run_id = run_id
    n0 = ensure_n0(client, VersionStore(s, store))
    r = run_and_assess(n0, "eval", ids, client, run_id, evaluator=ProtectedEvaluator.for_episode_set(ids, s), store=store, settings=s)
    rec = version_record(r)
    suffix = "" if set_name == "droid50" else f"_{set_name}"
    (s.artifacts_dir / f"baseline_v3{suffix}.json").write_text(json.dumps({"run_id": run_id, "episode_set": set_name, **rec}, indent=1, default=str))
    store.finish_run(run_id, "ok")
    h = rec["headline_gt"]
    print(json.dumps({"version": "N0", "episode_set": set_name, "episodes": rec["episodes"], "snapped_f1@0.75": round(h["snapped_f1_075"], 4),
                      "raw_f1@0.75": round(h["raw_f1_075"], 4), "snapped_f1@0.5": round(h["snapped_f1_05"], 4),
                      "pred/ref": f"{h['n_pred']}/{h['n_ref']}", "label_free": rec["label_free_self_score"],
                      "annotation_cost_usd": rec["annotation_cost_usd_total"], "statuses_not_ok": {e: v for e, v in rec["statuses"].items() if v != "ok"}}))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evoaction")
    sub = ap.add_subparsers(dest="cmd", required=True)
    dp = sub.add_parser("demo", help="live UI: one episode, human reference bar, one bar per version as it finishes")
    dp.add_argument("--port", type=int, default=8765)
    for name in ("setup", "baseline", "evolve", "report", "run-demo"):
        sp = sub.add_parser(name)
        sp.add_argument("--set", default="droid50", choices=["droid50", "rest50"])
        sp.add_argument("--hitl", action="store_true", help="human-in-the-loop: agents and gate see a fixed ~10%% labeled sample")
    args = ap.parse_args(argv)
    if args.cmd == "demo":
        from evoaction.demo import cmd_demo

        return cmd_demo(args)
    if args.cmd == "setup":
        return cmd_setup(args)
    if args.cmd == "baseline":
        return cmd_baseline(args)
    if args.cmd == "evolve":
        from evoaction.experiments.evolution import cmd_evolve

        return cmd_evolve(args)
    if args.cmd == "report":
        from evoaction.report import cmd_report

        return cmd_report(args)
    if args.cmd == "run-demo":
        from evoaction.experiments.evolution import cmd_evolve
        from evoaction.report import cmd_report

        for fn in (cmd_baseline, cmd_evolve, cmd_report):
            rc = fn(args)
            if rc:
                return rc
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
