"""Complete the evaluation of a HITL candidate whose run was interrupted by a harness crash.

Recursion 2 (droid50, human-in-the-loop) attempt 1: the main agent selected its round-3 patch (labeled-sample snapped
F1 0.381 vs incumbent 0.276); the candidate's 50-episode run stopped after 41 episodes on a zero-frame-window crash
(fixed in inspect_transition_windows and label_free.visual_verification). This script rebuilds that exact candidate from
the patch recorded in events.jsonl, checks its workflow hash against the hash logged when it was first created, runs it
on all 50 episodes, applies the predeclared labeled-sample gate against the incumbent run the agent saw in that
attempt, and records the outcome in lineage_hitl.json as a separate, clearly labeled experiment.
"""

from __future__ import annotations

import json
import time

from evoaction.clients.crusoe import CrusoeClient
from evoaction.config import get_settings
from evoaction.data.wgo import calibration_split, episode_set_ids, load_episode
from evoaction.evaluation.protected import ProtectedEvaluator
from evoaction.experiments.evolution import (PROTOCOL, labeled_feedback, load_lineage, save_lineage, subset_gt, version_record,
                                             window_cost)
from evoaction.persistence import get_store
from evoaction.pipeline import run_and_assess
from evoaction.schemas import WorkflowPatch
from evoaction.workflow.engine import VersionStore, apply_patch

SRC_RUN, SRC_EXP, SRC_ROUND, ORIGINAL_SHA = "RUN-dd7d1852af", "hitl-EXP-0002", 3, "12c888f4b1acfb7401bd4fb28e324bcae3c5497a40dc08a3839c837e5763a2a7"
NEW_ID = "hitl-EXP-0002a"


def main() -> int:
    s, store = get_settings(), get_store()
    trial = next(d for d in map(json.loads, open(s.events_path)) if d.get("run_id") == SRC_RUN and d.get("kind") == "hitl_trial"
                 and d.get("exp_id") == SRC_EXP and d.get("round") == SRC_ROUND)
    versions = VersionStore(s, store)
    parent = versions.load("hitl-N001")
    wf, prompts = apply_patch(parent.workflow, parent.prompts, WorkflowPatch(**trial["patch"]))
    lin = load_lineage(s, "droid50", hitl=True)
    cand = versions.create(NEW_ID, wf, prompts, parent.config, parent_id=parent.id, status="candidate",
                           experiment={"hypothesis": trial["hypothesis"], "candidate_patch": trial["patch"], "protocol": PROTOCOL,
                                       "source": f"{SRC_RUN} {SRC_EXP} round {SRC_ROUND} (run interrupted after 41/50 episodes)"})
    print("rebuilt workflow sha", cand.workflow_sha256, "matches original:", cand.workflow_sha256 == ORIGINAL_SHA)
    if cand.workflow_sha256 != ORIGINAL_SHA:
        raise SystemExit("reconstruction does not match the original candidate; not evaluating")
    ids = episode_set_ids("droid50", s)
    calib, held = calibration_split("droid50", s)
    client = CrusoeClient(s, store)
    run_id = store.start_run("complete_interrupted_candidate", {"candidate": NEW_ID, "source_run": SRC_RUN})
    client.run_id = run_id
    t0 = time.time()
    cand_val = run_and_assess(cand, "eval", ids, client, run_id, evaluator=ProtectedEvaluator.for_episode_set(ids, s), store=store, settings=s)
    inc_eps = {e: {"segments": json.load(open(s.artifacts_dir / "runs" / SRC_RUN / "hitl-N001" / f"{e}.json"))["segments"],
                   "duration_sec": load_episode(e)["duration_sec"]} for e in ids}
    inc_fb, cand_fb = labeled_feedback({"run": {"episodes": inc_eps}}, calib, s), labeled_feedback(cand_val, calib, s)
    c, i = cand_fb["pooled"], inc_fb["pooled"]
    d75, draw = c["snapped_f1_075"] - i["snapped_f1_075"], c["raw_f1_075"] - i["raw_f1_075"]
    bad = {e: st for e, st in cand_val["statuses"].items() if st not in ("ok", "partial")}
    promote = (d75 > 1e-9 or (abs(d75) <= 1e-9 and draw > 1e-9)) and not bad
    decision = "PROMOTED" if promote else "REJECTED"
    final = f"{NEW_ID}_{decision.lower()}"
    versions.finalize(NEW_ID, decision, {"experiment_id": NEW_ID, "protocol": PROTOCOL, "labeled_sample": {"candidate": c, "incumbent": i}}, new_id_=final)
    exp = {"id": NEW_ID, "parent_version": "hitl-N001", "version_dir": final, "hypothesis": trial["hypothesis"],
           "patch_summary": {k: v for k, v in trial["patch"].items() if v}, "decision": decision, "selected_round": SRC_ROUND,
           "labeled_sample": {"candidate": c, "incumbent": i}, "topology_change": WorkflowPatch(**trial["patch"]).is_topology_change(),
           "label_free": {"candidate_self_score": cand_val["assessment"]["mean_self_score"]},
           "shadow_gt_heldout": {"candidate": subset_gt(cand_val, held, s)},
           "recursion": 2, "attempt": 1, "recursion_cost": window_cost(store, run_id, t0, time.time() + 1),
           "note": (f"Recursion 2, attempt 1. The agent selected round {SRC_ROUND} of {SRC_RUN}; that run stopped after 41/50 episodes on a "
                    "zero-frame-window crash (harness bug, fixed). Rebuilt from the recorded patch (workflow hash identical), re-run on all 50 "
                    "after the fix, predeclared labeled-sample gate applied against the incumbent run the agent saw in that attempt.")}
    lin["versions"][final] = version_record(cand_val, held, calib)
    lin["experiments"] = [x for x in lin["experiments"] if x.get("id") != NEW_ID] + [exp]
    if promote:
        lin["incumbent"] = versions.promote_copy(final, "hitl-N002").id
        exp["new_incumbent"] = "hitl-N002"
    save_lineage(s, lin, "droid50", hitl=True)
    store.finish_run(run_id, "ok")
    print(json.dumps({"decision": decision, "labeled": {"candidate": c, "incumbent": i},
                      "heldout": exp["shadow_gt_heldout"]["candidate"] and {k: exp["shadow_gt_heldout"]["candidate"][k] for k in
                                                                              ("snapped_f1_075", "raw_f1_075", "snapped_f1_05", "n_pred", "n_ref")},
                      "all50": {k: lin["versions"][final]["headline_gt"][k] for k in ("snapped_f1_075", "raw_f1_075", "n_pred", "n_ref")},
                      "not_ok": {e: v for e, v in cand_val["statuses"].items() if v != "ok"}}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
