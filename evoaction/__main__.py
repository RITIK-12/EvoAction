"""EvoAction CLI.

  python -m evoaction baselines [--full]     run fixed A (N000) and B (strong_fixed) on dev + validation
  python -m evoaction evolve [--full]        recursive investigation -> experiments -> promotion/rejection
  python -m evoaction final [--full]         freeze A/B/C and run the sealed held-out evaluation once
  python -m evoaction report                 build artifacts/report.html from persisted artifacts
  python -m evoaction run-demo [--full]      all of the above in order
"""

from __future__ import annotations

import argparse
import json
import sys

from evoaction.config import get_settings


def cmd_baselines(args: argparse.Namespace) -> int:
    from evoaction.clients.crusoe import CrusoeClient
    from evoaction.data.wgo import split_ids
    from evoaction.evaluation.protected import ProtectedEvaluator
    from evoaction.persistence import get_store
    from evoaction.pipeline import ensure_baselines, run_and_score, summarize
    from evoaction.workflow.engine import VersionStore

    s = get_settings()
    store = get_store()
    client = CrusoeClient(s, store)
    run_id = store.start_run("baselines", {"pilot": not args.full})
    client.run_id = run_id
    versions = VersionStore(s, store)
    evaluator = ProtectedEvaluator(s)
    a, b = ensure_baselines(client, versions)
    out = {"run_id": run_id, "pilot": not args.full, "results": []}
    for split in ("dev", "validation"):
        ids = split_ids(split, pilot=not args.full)
        for v in (a, b):
            r = run_and_score(v, split, ids, client, evaluator, run_id, store, s)
            summ = summarize(r)
            out["results"].append(summ)
            print(json.dumps(summ, default=str))
    (s.artifacts_dir / "baselines.json").write_text(json.dumps(out, indent=1, default=str))
    store.finish_run(run_id, "ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evoaction")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("baselines", "evolve", "final", "run-demo"):
        p = sub.add_parser(name)
        p.add_argument("--full", action="store_true", help="use full splits instead of pilot subsets")
        if name == "final":
            p.add_argument("--dry-run", action="store_true", help="verify freeze inputs and budgets without inference or unsealing")
            p.add_argument("--test-set", choices=["fresh", "pilot", "full"], default="fresh",
                           help="fresh = never-evaluated full-split test episodes (default); pilot/full for comparison")
    sub.add_parser("report")
    args = ap.parse_args(argv)
    if args.cmd == "baselines":
        return cmd_baselines(args)
    if args.cmd == "evolve":
        from evoaction.experiments.evolution import cmd_evolve

        return cmd_evolve(args)
    if args.cmd == "final":
        from evoaction.experiments.final import cmd_final

        return cmd_final(args)
    if args.cmd == "report":
        from evoaction.report import cmd_report

        return cmd_report(args)
    if args.cmd == "run-demo":
        for fn in (cmd_baselines,):
            rc = fn(args)
            if rc:
                return rc
        from evoaction.experiments.evolution import cmd_evolve
        from evoaction.experiments.final import cmd_final
        from evoaction.report import cmd_report

        for fn in (cmd_evolve, cmd_final, cmd_report):
            rc = fn(args)
            if rc:
                return rc
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
