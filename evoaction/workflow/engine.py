"""Workflow graph validation, patching, execution, and immutable version storage."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from evoaction.clients.crusoe import BudgetExceeded, CrusoeClient, ModelCallError
from evoaction.config import Settings, get_settings
from evoaction.data.wgo import load_episode
from evoaction.persistence import Store, get_store, new_id
from evoaction.schemas import PARAM_MODELS, HarnessWorkflow, WorkflowNode, WorkflowPatch
from evoaction.workflow.operators import (
    MODEL_OPERATORS, OPERATORS, PROVIDES, REQUIRES, EpisodeContext, OperatorBudgetError, extra_requires,
)

MAX_NODES = 8
MAX_MODEL_PASSES = 2
ALLOWED_PROMPTS = ("prompt.md", "refine.md")


class WorkflowValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# -- validation -------------------------------------------------------------

def topo_order(nodes: list[WorkflowNode]) -> list[WorkflowNode] | None:
    by_id = {n.id: n for n in nodes}
    indeg = {n.id: 0 for n in nodes}
    for n in nodes:
        for d in n.depends_on:
            if d in indeg:
                indeg[n.id] += 1
    ready = [n.id for n in nodes if indeg[n.id] == 0]
    out: list[WorkflowNode] = []
    children: dict[str, list[str]] = {n.id: [] for n in nodes}
    for n in nodes:
        for d in n.depends_on:
            if d in children:
                children[d].append(n.id)
    while ready:
        ready.sort(key=lambda i: [x.id for x in nodes].index(i))
        cur = ready.pop(0)
        out.append(by_id[cur])
        for c in children[cur]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    return out if len(out) == len(nodes) else None


def ancestors(nodes: list[WorkflowNode], node_id: str) -> set[str]:
    by_id = {n.id: n for n in nodes}
    seen: set[str] = set()
    stack = list(by_id[node_id].depends_on)
    while stack:
        cur = stack.pop()
        if cur in seen or cur not in by_id:
            continue
        seen.add(cur)
        stack.extend(by_id[cur].depends_on)
    return seen


def validate_workflow(wf: HarnessWorkflow | dict[str, Any], prompts: dict[str, str]) -> HarnessWorkflow:
    """Deterministic validation; raises WorkflowValidationError with actionable messages."""
    errors: list[str] = []
    if isinstance(wf, dict):
        try:
            wf = HarnessWorkflow(**wf)
        except ValidationError as e:
            raise WorkflowValidationError([f"schema: {x['loc']}: {x['msg']}" for x in e.errors()]) from None
    nodes = wf.nodes
    ids = [n.id for n in nodes]
    if len(nodes) > MAX_NODES:
        errors.append(f"too many nodes: {len(nodes)} > {MAX_NODES}")
    if len(set(ids)) != len(ids):
        errors.append("duplicate node ids")
    for n in nodes:
        try:
            PARAM_MODELS[n.operator](**n.params)
        except ValidationError as e:
            errors.append(f"node {n.id} ({n.operator}) params: " + "; ".join(f"{x['loc']}: {x['msg']}" for x in e.errors()))
        for d in n.depends_on:
            if d not in ids:
                errors.append(f"node {n.id} depends on unknown node {d!r}")
            if d == n.id:
                errors.append(f"node {n.id} depends on itself")
        for key in ("prompt",):
            if key in n.params and n.params[key] not in prompts:
                errors.append(f"node {n.id} references missing prompt file {n.params[key]!r}")
    if errors:
        raise WorkflowValidationError(errors)
    order = topo_order(nodes)
    if order is None:
        raise WorkflowValidationError(["workflow graph contains a cycle"])
    ops = [n.operator for n in nodes]
    if ops.count("initial_annotate") != 1:
        errors.append("exactly one initial_annotate node is required")
    model_nodes = [n for n in nodes if n.operator in MODEL_OPERATORS]
    if len(model_nodes) > MAX_MODEL_PASSES:
        errors.append(f"at most {MAX_MODEL_PASSES} multimodal passes per interval (initial_annotate + one refine_intervals)")
    if wf.max_calls < len(model_nodes):
        errors.append(f"max_calls={wf.max_calls} is below the number of model operators ({len(model_nodes)})")
    exports = [n for n in nodes if n.operator == "validate_export"]
    if len(exports) != 1:
        errors.append("exactly one validate_export node is required")
    else:
        exp = exports[0]
        if any(exp.id in n.depends_on for n in nodes):
            errors.append("validate_export must be terminal (no dependents)")
        if order[-1].id != exp.id:
            errors.append("validate_export must be last in topological order")
        dangling = set(ids) - {exp.id} - ancestors(nodes, exp.id)
        if dangling:
            errors.append(f"nodes not feeding validate_export: {sorted(dangling)}")
    for n in nodes:
        anc = ancestors(nodes, n.id)
        available = {"episode"} | set().union(*[PROVIDES[x.operator] for x in nodes if x.id in anc]) if anc else {"episode"}
        need = REQUIRES[n.operator] | extra_requires(n.operator, n.params)
        missing = need - available
        if missing:
            errors.append(f"node {n.id} ({n.operator}) requires {sorted(missing)} from an upstream dependency")
    if errors:
        raise WorkflowValidationError(errors)
    return wf


def _resolve_operator_refs(wf: HarnessWorkflow, patch: WorkflowPatch) -> WorkflowPatch:
    """Agents sometimes name a node by its operator (e.g. 'validate_export' for node 'export'). Map such a reference to the
    node id when exactly one existing or inserted node has that operator; anything else is left for validation to reject."""
    all_nodes = list(wf.nodes) + [ins.node for ins in patch.insert_nodes]
    ids = {n.id for n in all_nodes}
    by_op: dict[str, list[str]] = {}
    for n in all_nodes:
        by_op.setdefault(n.operator, []).append(n.id)
    fix = lambda r: r if r in ids or len(by_op.get(r, [])) != 1 else by_op[r][0]
    data = patch.model_dump()
    data["remove_nodes"] = [fix(r) for r in data["remove_nodes"]]
    for ins in data["insert_nodes"]:
        ins["node"]["depends_on"] = [fix(d) for d in ins["node"]["depends_on"]]
        if ins.get("rewire_dependents_of"):
            ins["rewire_dependents_of"] = fix(ins["rewire_dependents_of"])
    data["set_params"] = {fix(k): v for k, v in data["set_params"].items()}
    data["set_when"] = {fix(k): v for k, v in data["set_when"].items()}
    data["set_depends_on"] = {fix(k): [fix(d) for d in v] for k, v in data["set_depends_on"].items()}
    return WorkflowPatch(**data)


def apply_patch(wf: HarnessWorkflow, prompts: dict[str, str], patch: WorkflowPatch) -> tuple[HarnessWorkflow, dict[str, str]]:
    """Apply a typed patch to a copy; the result is re-validated by the caller."""
    patch = _resolve_operator_refs(wf, patch)
    nodes = [n.model_copy(deep=True) for n in wf.nodes]
    by_id = {n.id: n for n in nodes}
    errors: list[str] = []
    for rid in patch.remove_nodes:
        if rid not in by_id:
            errors.append(f"remove: unknown node {rid!r}")
            continue
        removed = by_id.pop(rid)
        nodes = [n for n in nodes if n.id != rid]
        for n in nodes:  # bridge dependents to the removed node's parents
            if rid in n.depends_on:
                n.depends_on = [d for d in n.depends_on if d != rid] + [d for d in removed.depends_on if d not in n.depends_on]
    for ins in patch.insert_nodes:
        node = ins.node.model_copy(deep=True)
        if node.id in by_id:
            errors.append(f"insert: node id {node.id!r} already exists")
            continue
        if ins.rewire_dependents_of:
            src = ins.rewire_dependents_of
            if src not in by_id:
                errors.append(f"insert: rewire source {src!r} unknown")
            for n in nodes:
                if src in n.depends_on and n.id != node.id:
                    n.depends_on = [node.id if d == src else d for d in n.depends_on]
        nodes.append(node)
        by_id[node.id] = node
    for nid, params in patch.set_params.items():
        if nid not in by_id:
            errors.append(f"set_params: unknown node {nid!r}")
            continue
        by_id[nid].params = {**by_id[nid].params, **params}
    for nid, cond in patch.set_when.items():
        if nid not in by_id:
            errors.append(f"set_when: unknown node {nid!r}")
            continue
        by_id[nid].when = cond
    for nid, deps in patch.set_depends_on.items():
        if nid not in by_id:
            errors.append(f"set_depends_on: unknown node {nid!r}")
            continue
        by_id[nid].depends_on = list(deps)
    if errors:
        raise WorkflowValidationError(errors)
    new_prompts = {**prompts, **patch.prompt_updates}
    if any(n.operator == "refine_intervals" for n in nodes) and "refine.md" not in new_prompts:
        from evoaction.workflow.baselines import DEFAULT_REFINE_PROMPT

        new_prompts["refine.md"] = DEFAULT_REFINE_PROMPT  # operator library default playbook
    # keep a stable, readable order: topological
    new_wf = HarnessWorkflow(nodes=nodes, max_calls=wf.max_calls, max_wall_seconds=wf.max_wall_seconds)
    order = topo_order(new_wf.nodes)
    if order is not None:
        new_wf = HarnessWorkflow(nodes=order, max_calls=max(wf.max_calls, sum(1 for n in order if n.operator in MODEL_OPERATORS)),
                                 max_wall_seconds=wf.max_wall_seconds)
    return validate_workflow(new_wf, new_prompts), new_prompts


# -- versions -----------------------------------------------------------------

@dataclass
class Version:
    id: str
    path: Path
    workflow: HarnessWorkflow
    prompts: dict[str, str]
    config: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def workflow_sha256(self) -> str:
        return self.manifest["workflow_sha256"]


def version_hash(wf: HarnessWorkflow, prompts: dict[str, str], config: dict[str, Any]) -> str:
    return sha256_text(canonical({"workflow": wf.model_dump(), "prompts": prompts, "models": config.get("models")}))


class VersionStore:
    def __init__(self, settings: Settings | None = None, store: Store | None = None):
        self.settings = settings or get_settings()
        self.root = self.settings.artifacts_dir / "versions"
        self.store = store or get_store()

    def create(self, version_id: str, wf: HarnessWorkflow, prompts: dict[str, str], config: dict[str, Any],
               parent_id: str | None, status: str, experiment: dict[str, Any] | None = None) -> Version:
        wf = validate_workflow(wf, prompts)
        path = self.root / version_id
        if path.exists():
            existing = self.load(version_id)
            if existing.manifest["workflow_sha256"] != version_hash(wf, prompts, config):
                raise FileExistsError(f"version {version_id} exists with different content")
            return existing
        path.mkdir(parents=True)
        files = {"workflow.json": json.dumps(wf.model_dump(), indent=1), "config.json": json.dumps(config, indent=1)}
        for name, text in prompts.items():
            if name not in ALLOWED_PROMPTS:
                raise ValueError(f"prompt file {name} not allowed")
            files[name] = text
        if experiment is not None:
            files["experiment.json"] = json.dumps(experiment, indent=1, default=str)
        for name, text in files.items():
            (path / name).write_text(text)
        manifest = {"id": version_id, "parent_id": parent_id, "status": status, "frozen": False, "created_at": time.time(),
                    "workflow_sha256": version_hash(wf, prompts, config),
                    "files": {name: sha256_text(text) for name, text in files.items()}}
        (path / "manifest.json").write_text(json.dumps(manifest, indent=1))
        self.store.insert("workflow_versions", {"id": version_id, "parent_id": parent_id, "status": status,
                                                "workflow_hash": manifest["workflow_sha256"],
                                                "prompt_hash": sha256_text(canonical(prompts)), "path": str(path),
                                                "manifest_json": manifest, "created_at": time.time()})
        self.store.event(None, "version_created", {"version_id": version_id, "parent_id": parent_id, "status": status,
                                                   "workflow_sha256": manifest["workflow_sha256"]})
        return Version(version_id, path, wf, prompts, config, manifest)

    def load(self, version_id: str) -> Version:
        path = self.root / version_id
        manifest = json.loads((path / "manifest.json").read_text())
        for name, h in manifest["files"].items():
            if sha256_text((path / name).read_text()) != h:
                raise ValueError(f"version {version_id}: {name} hash mismatch (immutable artifact modified)")
        wf = HarnessWorkflow(**json.loads((path / "workflow.json").read_text()))
        prompts = {p: (path / p).read_text() for p in ALLOWED_PROMPTS if (path / p).exists()}
        config = json.loads((path / "config.json").read_text())
        return Version(version_id, path, wf, prompts, config, manifest)

    def finalize(self, version_id: str, status: str, validation_report: dict[str, Any], new_id_: str | None = None) -> Version:
        """Write validation_report.json, set outcome status, freeze. Optionally rename to an outcome-derived id."""
        v = self.load(version_id)
        if v.manifest.get("frozen"):
            raise PermissionError(f"version {version_id} is frozen")
        text = json.dumps(validation_report, indent=1, default=str)
        (v.path / "validation_report.json").write_text(text)
        v.manifest["files"]["validation_report.json"] = sha256_text(text)
        v.manifest.update({"status": status, "frozen": True, "frozen_at": time.time()})
        target = v.path
        if new_id_ and new_id_ != version_id:
            target = self.root / new_id_
            shutil.move(str(v.path), str(target))
            v.manifest["id"] = new_id_
            v.manifest["original_id"] = version_id
        (target / "manifest.json").write_text(json.dumps(v.manifest, indent=1))
        final_id = new_id_ or version_id
        self.store.insert("workflow_versions", {"id": final_id, "parent_id": v.manifest.get("parent_id"), "status": status,
                                                "workflow_hash": v.manifest["workflow_sha256"],
                                                "prompt_hash": sha256_text(canonical(v.prompts)), "path": str(target),
                                                "manifest_json": v.manifest, "created_at": time.time()})
        if final_id != version_id:
            self.store.update("workflow_versions", version_id, {"status": f"renamed->{final_id}"})
        self.store.event(None, "version_finalized", {"version_id": final_id, "status": status})
        return self.load(final_id)

    def promote_copy(self, source_id: str, new_version_id: str) -> Version:
        src = self.load(source_id)
        target = self.root / new_version_id
        if target.exists():
            raise FileExistsError(new_version_id)
        shutil.copytree(src.path, target)
        m = dict(src.manifest, id=new_version_id, source_experiment=source_id, status="incumbent")
        (target / "manifest.json").write_text(json.dumps(m, indent=1))
        self.store.insert("workflow_versions", {"id": new_version_id, "parent_id": src.manifest.get("parent_id"), "status": "incumbent",
                                                "workflow_hash": m["workflow_sha256"], "prompt_hash": sha256_text(canonical(src.prompts)),
                                                "path": str(target), "manifest_json": m, "created_at": time.time()})
        self.store.event(None, "version_promoted", {"version_id": new_version_id, "source": source_id})
        return self.load(new_version_id)


# -- execution ----------------------------------------------------------------

def _condition(when: str, state: dict[str, Any]) -> bool:
    if when == "always":
        return True
    if when == "has_telemetry":
        return bool(state["episode"].get("telemetry_path"))
    if when == "has_boundaries":
        return len(state.get("segments") or []) >= 2
    if when == "has_short_segments":
        return any(s["end_sec"] - s["start_sec"] < 2.0 for s in state.get("segments") or [])
    if when == "has_hints":
        return bool(state.get("hints"))
    return False


def run_episode(version: Version, episode_id: str, client: CrusoeClient, run_id: str, store: Store,
                settings: Settings) -> dict[str, Any]:
    ep = load_episode(episode_id, settings)
    out_dir = settings.artifacts_dir / "runs" / run_id / version.id
    ctx = EpisodeContext(episode=ep, prompts=version.prompts, client=client, version_id=version.id, run_id=run_id,
                         out_dir=out_dir, max_calls=version.workflow.max_calls)
    state: dict[str, Any] = {"episode": ep}
    t_start = time.time()
    status = "ok"
    error = None
    trace = []
    inference_cost = 0.0
    tokens = {"prompt": 0, "completion": 0}
    for node in topo_order(version.workflow.nodes) or []:
        t0 = time.time()
        if time.time() - t_start > version.workflow.max_wall_seconds:
            status, error = "timeout", f"max_wall_seconds={version.workflow.max_wall_seconds} exceeded before {node.id}"
            break
        if not _condition(node.when, state):
            store.insert("operator_runs", {"id": new_id("OP"), "run_id": run_id, "version_id": version.id, "episode_id": episode_id,
                                           "node_id": node.id, "operator": node.operator, "status": "skipped",
                                           "input_hash": None, "output_hash": None, "duration_s": 0.0,
                                           "detail_json": {"when": node.when}, "created_at": time.time()})
            trace.append({"node": node.id, "operator": node.operator, "status": "skipped", "when": node.when})
            continue
        in_hash = sha256_text(canonical({k: v for k, v in state.items() if k in ("segments", "hints")}))
        try:
            updates, detail = OPERATORS[node.operator](ctx, state, node.params)
            op_status = "ok"
        except (OperatorBudgetError, BudgetExceeded) as e:
            updates, detail, op_status = {}, {"error": str(e)}, "budget_denied"
        except ModelCallError as e:
            updates, detail, op_status = {}, {"error": str(e)[:300]}, "model_error"
        except Exception as e:  # an operator bug degrades this episode (downstream continues) instead of aborting the run
            updates, detail, op_status = {}, {"error": f"{type(e).__name__}: {str(e)[:300]}"}, "operator_error"
        state.update(updates)
        inference_cost += float(detail.get("cost_usd") or 0.0)
        tokens["prompt"] += int(detail.get("prompt_tokens") or 0)
        tokens["completion"] += int(detail.get("completion_tokens") or 0)
        out_hash = sha256_text(canonical({k: v for k, v in state.items() if k in ("segments", "hints")}))
        store.insert("operator_runs", {"id": new_id("OP"), "run_id": run_id, "version_id": version.id, "episode_id": episode_id,
                                       "node_id": node.id, "operator": node.operator, "status": op_status, "input_hash": in_hash,
                                       "output_hash": out_hash, "duration_s": time.time() - t0, "detail_json": detail,
                                       "created_at": time.time()})
        trace.append({"node": node.id, "operator": node.operator, "status": op_status, **{k: v for k, v in detail.items() if k in (
            "n_segments", "n_sheets", "n_frames", "n_windows", "n_hints", "decisions", "skipped", "error", "n_in", "n_out", "cache_hit", "thinking_fallback")}})
        if op_status != "ok" and node.operator == "initial_annotate":
            status, error = op_status, detail.get("error")
            break
        if op_status != "ok":
            status = "partial"  # downstream operators continue from the last good state
    segments = state.get("export", {}).get("segments")
    if segments is None:
        segments = []
        if status == "ok":
            status = "no_export"
    return {"episode_id": episode_id, "segments": segments, "status": status, "error": error, "calls_used": ctx.calls_used,
            "wall_s": time.time() - t_start, "trace": trace, "duration_sec": ep["duration_sec"],
            "inference_cost_usd": inference_cost, "inference_tokens": tokens}


def run_version(version: Version, episode_ids: list[str], client: CrusoeClient, run_id: str, *, store: Store | None = None,
                settings: Settings | None = None, workers: int | None = None) -> dict[str, Any]:
    store = store or get_store()
    settings = settings or get_settings()
    store.event(run_id, "version_run_started", {"version_id": version.id, "episodes": episode_ids})
    workers = workers or (settings or get_settings()).workers
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(episode_ids)))) as ex:
        results = list(ex.map(lambda e: run_episode(version, e, client, run_id, store, settings), episode_ids))
    out = {"version_id": version.id, "workflow_sha256": version.workflow_sha256, "episodes": {r["episode_id"]: r for r in results}}
    store.event(run_id, "version_run_finished", {"version_id": version.id,
                                                 "statuses": {r["episode_id"]: r["status"] for r in results},
                                                 "calls": sum(r["calls_used"] for r in results)})
    return out


def usage_for(store: Store, run_id: str, version_id: str | None = None, role: str | None = None) -> dict[str, Any]:
    sql = ("SELECT COUNT(*) calls, SUM(cache_hit) cache_hits, COALESCE(SUM(prompt_tokens),0) pt, COALESCE(SUM(completion_tokens),0) ct, "
           "COALESCE(SUM(cost_usd),0) cost, COALESCE(SUM(latency_s),0) latency FROM model_calls WHERE run_id = ?")
    params: list[Any] = [run_id]
    if version_id:
        sql += " AND version_id = ?"
        params.append(version_id)
    if role:
        sql += " AND role = ?"
        params.append(role)
    r = store.query(sql, params)[0]
    return {"calls": r["calls"], "cache_hits": r["cache_hits"] or 0, "prompt_tokens": r["pt"], "completion_tokens": r["ct"],
            "cost_usd": round(r["cost"], 6), "model_latency_s": round(r["latency"], 2)}


def deepcopy_workflow(wf: HarnessWorkflow) -> HarnessWorkflow:
    return HarnessWorkflow(**copy.deepcopy(wf.model_dump()))
