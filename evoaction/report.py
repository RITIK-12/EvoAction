"""Static, self-contained HTML evidence report built only from persisted artifacts."""

from __future__ import annotations

import argparse
import difflib
import html
import json
from pathlib import Path
from typing import Any

from evoaction.config import get_settings
from evoaction.persistence import get_store


def _load(p: Path) -> Any:
    return json.loads(p.read_text()) if p.exists() else None


def esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return esc(x)


CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--card:#fff;--line:#e4e1d8;--accent:#3b6e5a;--warn:#a5542b;--A:#8a8a8a;--B:#4a6fa5;--C:#3b6e5a;--ref:#1d1d1b}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#151514;--fg:#ecebe6;--muted:#a19f97;--card:#1e1e1c;--line:#34332f;--accent:#7fb89d;--warn:#e0915f;--A:#9a9a9a;--B:#86a7d8;--C:#7fb89d;--ref:#ecebe6}}
:root[data-theme="dark"]{--bg:#151514;--fg:#ecebe6;--muted:#a19f97;--card:#1e1e1c;--line:#34332f;--accent:#7fb89d;--warn:#e0915f;--A:#9a9a9a;--B:#86a7d8;--C:#7fb89d;--ref:#ecebe6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:15px;margin:18px 0 6px}.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:10px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:600}
td.num{text-align:right;font-variant-numeric:tabular-nums}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;margin:0}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;border:1px solid var(--line)}.ok{color:var(--accent);border-color:var(--accent)}.bad{color:var(--warn);border-color:var(--warn)}
.tree{list-style:none;padding-left:18px;border-left:2px solid var(--line);margin:6px 0}.tree li{margin:8px 0}.q{font-weight:600}
.del{background:rgba(165,84,43,.15)}.add{background:rgba(59,110,90,.18)}code{font-size:12.5px}
.legend span{display:inline-block;margin-right:14px;font-size:12px}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
"""


def section_sponsor(store, pre) -> str:
    rows = store.query("SELECT COALESCE(r.kind || ' (' || r.status || ')', 'ad-hoc probe (no run id)') kind, m.role, m.model, COUNT(*) n, SUM(m.cache_hit) ch, SUM(CASE WHEN m.status!='ok' THEN 1 ELSE 0 END) failed, "
                       "COALESCE(SUM(m.prompt_tokens),0) pt, COALESCE(SUM(m.completion_tokens),0) ct, COALESCE(SUM(m.cost_usd),0) cost, "
                       "COALESCE(SUM(m.latency_s),0) lat FROM model_calls m LEFT JOIN runs r ON m.run_id = r.id "
                       "GROUP BY 1, m.role, m.model ORDER BY 1, m.role")
    t = "".join(f"<tr><td>{esc(r['kind'])}</td><td>{esc(r['role'])}</td><td><code>{esc(r['model'])}</code></td><td class=num>{r['n']}</td>"
                f"<td class=num>{r['ch'] or 0}</td><td class=num>{r['failed']}</td><td class=num>{r['pt']:,}</td><td class=num>{r['ct']:,}</td>"
                f"<td class=num>${r['cost']:.4f}</td><td class=num>{r['lat']:.0f}s</td></tr>" for r in rows)
    tot = store.query("SELECT COUNT(*) n, COALESCE(SUM(cost_usd),0) c FROM model_calls WHERE cache_hit=0")[0]
    locked = (pre or {}).get("locked", {})
    return f"""<h2>7. Sponsor-backed execution (Crusoe Managed Inference, NVIDIA Nemotron)</h2>
<div class=card><p>Reasoning / supervisor / investigators / strategy / critic: <code>{esc(locked.get('CRUSOE_REASONING_MODEL'))}</code><br>
Vision annotation and refinement: <code>{esc(locked.get('CRUSOE_VISION_MODEL'))}</code><br>
<span class=muted>Model IDs confirmed live via <code>GET /v1/models</code> in preflight. Every row below is a persisted real call (cache hits re-serve an earlier real response and cost $0).
Total non-cached calls: {tot['n']}, provider-token cost ${tot['c']:.4f} at listed per-token prices.</span></p>
<table><tr><th>run</th><th>role</th><th>model</th><th>calls</th><th>cache hits</th><th>failed attempts</th><th>prompt tok</th><th>completion tok</th><th>cost</th><th>model time</th></tr>{t}</table></div>"""


SETS = (("droid50", "Batch 1 · 50 DROID"), ("rest50", "Batch 2 · 25 Galaxea + 25 HomER"))
# Comparison uses only Macrodata's original model sweep (Segment F1 at tIoU >= 0.75, full 100-episode benchmark,
# contact-sheet setup, before their final prompt optimization). Predicted-segment counts are from the same post.
MACRODATA_REFERENCE: list[tuple[str, str, float]] = []
MACRODATA_SWEEP_100 = [("Gemini 3.5 Flash", 645, 0.290), ("Gemini Robotics ER 1.6", 556, 0.245), ("GPT-5.5 (low)", 602, 0.233),
                       ("GPT-5.4 (low)", 529, 0.233), ("Gemini 3.1 Pro Preview", 586, 0.232), ("Gemini 3 Flash Preview", 685, 0.210),
                       ("Claude Opus 4.8", 733, 0.168), ("Claude Sonnet 4.6", 698, 0.146), ("Gemini 2.5 Pro", 662, 0.145),
                       ("Qwen 3.5 Plus", 763, 0.141), ("Claude Haiku 4.5", 570, 0.128), ("Gemini 3.1 Flash Lite", 757, 0.119),
                       ("Qwen 3.6 Flash", 715, 0.110), ("GPT-5.4 mini", 317, 0.106), ("Gemini 2.5 Flash", 855, 0.059), ("Gemma 4 26B", 1080, 0.057)]


def _sfx(name: str) -> str:
    return "" if name == "droid50" else f"_{name}"


def recursion_levels(lin: dict[str, Any] | None, base: dict[str, Any] | None, set_name: str = "droid50", hitl: bool = False) -> list[dict[str, Any]]:
    """Level 0 = N0; level k = state after the k-th experiment (incumbent after the gate, plus the candidate tried)."""
    if not lin and not base:
        return []
    versions = (lin or {}).get("versions", {})
    n0 = versions.get("N0") or base
    levels = [{"level": 0, "incumbent_id": "N0", "incumbent": n0, "candidate_id": None, "candidate": None, "decision": "baseline",
               "recursion_cost": None}]
    inc_id, inc, promoted = "N0", n0, 0
    prefix = ("" if set_name == "droid50" else f"{set_name}-") + ("hitl-" if hitl else "")
    for e in (lin or {}).get("experiments", []):
        cand = versions.get(e.get("version_dir") or "")
        if e.get("decision") == "PROMOTED":
            promoted += 1
            inc_id, inc = f"{prefix}N{promoted:03d}", cand
        levels.append({"level": e.get("recursion") or len(levels), "attempt": e.get("attempt"), "incumbent_id": inc_id, "incumbent": inc, "candidate_id": e.get("version_dir") or e["id"],
                       "candidate": cand, "decision": e.get("decision"), "recursion_cost": e.get("recursion_cost"),
                       "hypothesis": e.get("hypothesis")})
    return levels


def _f1(tp: float, npred: float, nref: float) -> float | None:
    return 2 * tp / (npred + nref) if npred + nref else None


def _pooled(recs: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    hs = [(r or {}).get("headline_gt") or {} for r in recs]
    if not recs or any(not h for h in hs):
        return None
    sm = lambda k: sum(h.get(k) or 0 for h in hs)
    # The evaluator drops a segment that snapping inverts, so snapped scores use its snapped prediction count;
    # older records lack it, so derive it from the persisted snapped F1 and TP.
    nps = sum(h["n_pred_snapped"] if h.get("n_pred_snapped") is not None else
              (round(2 * h["tp_snapped"] / h["snapped_f1_075"] - h["n_ref"]) if h.get("snapped_f1_075") else h["n_pred"]) for h in hs)
    return {"snapped_f1_075": _f1(sm("tp_snapped"), nps, sm("n_ref")), "raw_f1_075": _f1(sm("tp_raw"), sm("n_pred"), sm("n_ref")),
            "snapped_f1_05": _f1(sm("tp05_snapped"), nps, sm("n_ref")), "n_pred": sm("n_pred"), "n_ref": sm("n_ref"),
            "episodes": sm("episodes"), "cost": sum((r or {}).get("annotation_cost_usd_total") or 0 for r in recs)}


def _row(label: str, rec: dict[str, Any] | None, extra: str = "", bold: bool = False) -> str:
    h = (rec or {}).get("headline_gt") or {}
    b = (lambda x: f"<b>{x}</b>") if bold else (lambda x: x)
    return (f"<tr><td>{label}</td><td class=num>{b(fmt(h.get('snapped_f1_075')))}</td><td class=num>{fmt(h.get('raw_f1_075'))}</td>"
            f"<td class=num>{fmt(h.get('snapped_f1_05'))}</td><td class=num>{fmt(h.get('n_pred'))}/{fmt(h.get('n_ref'))}</td>"
            f"<td class=num>{fmt((rec or {}).get('label_free_self_score'))}</td><td class=num>{('$' + format(rec['annotation_cost_usd_total'], '.4f')) if rec else '–'}</td>{extra}</tr>")


RESULT_HEAD = ("<tr><th>version</th><th>Segment F1@0.75 snapped (headline)</th><th>F1@0.75 raw</th><th>F1@0.5 snapped</th><th>pred/ref</th>"
               "<th>label-free self-score</th><th>annotation cost (all episodes)</th>")


def section_results(a: Path) -> str:
    out, combined = [], {}
    for name, title in SETS:
        lin, base = _load(a / f"lineage{_sfx(name)}.json"), _load(a / f"baseline_v3{_sfx(name)}.json")
        es = _load(a / "episode_sets" / f"{name}.json")
        levels = recursion_levels(lin, base, name)
        if not levels:
            continue
        refs = "".join(f"<tr class=muted><td>{esc(lbl)}</td><td class=num>{v:.3f}</td><td colspan=6>reference; not run here</td></tr>"
                       for lbl, s_, v in MACRODATA_REFERENCE if s_ == name)
        rows = ""
        for L in levels:
            if L["level"] == 0:
                rows += _row(f"recursion 0 · <code>N0</code> (Macrodata Refiner protocol on Nemotron Omni)", L["incumbent"],
                             "<td>baseline</td><td class=num>–</td>", bold=True)
                continue
            rc = (L.get("recursion_cost") or {}).get("total_cost_usd")
            dec = L["decision"]
            cls = "ok" if dec == "PROMOTED" else "bad"
            rows += _row(f"recursion {L['level']} · candidate <code>{esc(L['candidate_id'])}</code>", L["candidate"],
                         f"<td><span class='pill {cls}'>{esc(dec)}</span></td><td class=num>{('$' + format(rc, '.4f')) if rc is not None else '–'}</td>")
            rows += _row(f"recursion {L['level']} · incumbent <code>{esc(L['incumbent_id'])}</code>", L["incumbent"], "<td>incumbent</td><td></td>", bold=True)
            combined.setdefault(L["level"], {})[name] = (L["incumbent_id"], L["incumbent"])
        combined.setdefault(0, {})[name] = ("N0", levels[0]["incumbent"])
        n_ep = (es or {}).get("n_episodes")
        out.append(f"""<div class=card><h3>{esc(title)} ({esc(n_ep)} episodes; set sha256 <code>{esc((es or {}).get('set_sha256', '')[:16])}</code>)</h3>
<table>{RESULT_HEAD}<th>gate decision</th><th>recursion cost (agents + candidate run)</th></tr>{refs}{rows}</table></div>""")
    comb_rows = ""
    ranked: list[tuple[str, int | None, float, bool]] = [(m, n, f, False) for m, n, f in MACRODATA_SWEEP_100]
    for lvl in sorted(combined):
        got = combined[lvl]
        if len(got) == len(SETS):
            pc = _pooled([got[n][1] for n, _ in SETS])
            if pc:
                ids = " + ".join(f"<code>{esc(got[n][0])}</code>" for n, _ in SETS)
                comb_rows += (f"<tr><td>recursion {lvl} · {ids}</td><td class=num><b>{fmt(pc['snapped_f1_075'])}</b></td><td class=num>{fmt(pc['raw_f1_075'])}</td>"
                              f"<td class=num>{fmt(pc['snapped_f1_05'])}</td><td class=num>{pc['n_pred']}/{pc['n_ref']}</td><td class=num>{pc['episodes']}</td>"
                              f"<td class=num>${pc['cost']:.4f}</td></tr>")
                ranked.append((f"EvoAction recursion {lvl}: {' + '.join(got[n][0] for n, _ in SETS)} (Nemotron Omni 30B-A3B)", pc["n_pred"], pc["snapped_f1_075"] or 0.0, True))
    comb = (f"""<div class=card><h3>Full benchmark (pooled over both batches)</h3><table><tr><th>recursion level</th><th>Segment F1@0.75 snapped</th>
<th>F1@0.75 raw</th><th>F1@0.5 snapped</th><th>pred/ref</th><th>episodes</th><th>annotation cost</th></tr>{comb_rows}</table>
<p class=muted>Pooled micro F1 = 2·TP / (pred + ref) summed over both batches; each batch keeps its own lineage.</p></div>
<div class=card><h3>Against Macrodata's published 100-episode model sweep</h3><table><tr><th>#</th><th>system</th><th>predicted segments</th><th>Segment F1@0.75</th></tr>
{''.join(f"<tr><td>{i + 1}</td><td>{'<b>' + esc(m) + '</b>' if ours else esc(m)}</td><td class=num>{fmt(n)}</td><td class=num>{'<b>' + fmt(f) + '</b>' if ours else fmt(f)}</td></tr>" for i, (m, n, f, ours) in enumerate(sorted(ranked, key=lambda x: -x[2])))}</table>
<p class=muted>Published rows: Macrodata's original model sweep (Segment F1 at tIoU ≥ 0.75, full 100-episode benchmark, contact-sheet setup, before their final prompt optimization). Nemotron was not in their sweep.</p></div>""" if comb_rows else
            "<div class=card class=muted>Full-benchmark pooling appears once both batches have results.</div>")
    return f"""<h2>1. Results: Macrodata's metric on the same episodes for every version</h2>
<p class=muted>Headline metric follows Macrodata's WGO-Bench evaluation: temporal IoU ≥ 0.75, one-to-one maximum matching, pooled over episodes, with the first predicted
start and last predicted end snapped to the human episode boundaries (scoring-only); labels are ignored. Every version in a batch runs on exactly the same episodes.
Ground truth is measured by the protected evaluator for this table only; agents and the promotion gate never see it.</p>
{''.join(out) or '<div class=card>No v3 results yet.</div>'}{comb}"""


def _pooled_key(recs: list[dict[str, Any] | None], key: str) -> dict[str, Any] | None:
    return _pooled([{**(r or {}), "headline_gt": (r or {}).get(key)} for r in recs])


def section_hitl(a: Path) -> str:
    """Human-in-the-loop lineages: ~10% labeled sample shared with agents and gate; held-out 90% is the honest score."""
    rows_by_set, levels_by_set = {}, {}
    for name, title in SETS:
        lin = _load(a / f"lineage{_sfx(name)}_hitl.json")
        if not lin:
            continue
        levels = recursion_levels(lin, None, name, hitl=True)
        levels_by_set[name] = levels
        cal = _load(a / "episode_sets" / f"{name}_calibration.json") or {}
        rows = ""
        cands = [L for L in levels if L["level"] and L["candidate"]]
        best_vid = max(cands, key=lambda L: ((L["candidate"].get("headline_gt_heldout") or {}).get("snapped_f1_075") or 0))["candidate_id"] if cands else None
        for L in levels:
            for kind, rec, vid in ((("candidate", L["candidate"], L["candidate_id"]),) if L["level"] else ()) + (("incumbent", L["incumbent"], L["incumbent_id"]),):
                r = rec or {}
                lab, held, full = r.get("headline_gt_labeled") or {}, r.get("headline_gt_heldout") or {}, r.get("headline_gt") or {}
                dec = L["decision"] if kind == "candidate" else ("baseline" if L["level"] == 0 else "incumbent")
                cls = "ok" if dec == "PROMOTED" else "bad" if dec == "REJECTED" else ""
                att = f" (attempt {L['attempt']})" if L.get("attempt") and kind == "candidate" else ""
                flag = " <span class='pill ok'>best version found</span>" if kind == "candidate" and vid == best_vid else ""
                rows += (f"<tr><td>recursion {L['level']}{att} · {kind} <code>{esc(vid)}</code>{flag}</td><td class=num>{fmt(lab.get('snapped_f1_075'))}</td>"
                         f"<td class=num><b>{fmt(held.get('snapped_f1_075'))}</b></td><td class=num>{fmt(held.get('raw_f1_075'))}</td>"
                         f"<td class=num>{fmt(held.get('n_pred'))}/{fmt(held.get('n_ref'))}</td><td class=num>{fmt(full.get('snapped_f1_075'))}</td>"
                         f"<td class=num>{('$' + format(r['annotation_cost_usd_total'], '.4f')) if r else '–'}</td><td><span class='pill {cls}'>{esc(dec)}</span></td></tr>")
        rows_by_set[name] = (f"""<div class=card><h3>{esc(title)}: labeled sample {esc(', '.join(cal.get('calibration_ids', [])))}</h3>
<table><tr><th>version</th><th>labeled-sample snapped F1@0.75 (gate signal)</th><th>held-out snapped F1@0.75 (headline)</th><th>held-out raw F1@0.75</th>
<th>held-out pred/ref</th><th>all-episode snapped F1@0.75</th><th>annotation cost</th><th>decision</th></tr>{rows}</table></div>""")
    if not rows_by_set:
        return ""
    pooled = ""
    if len(levels_by_set) == len(SETS):
        n = max(len(v) for v in levels_by_set.values())
        for lvl in range(n):
            got = [levels_by_set[s_][min(lvl, len(levels_by_set[s_]) - 1)] for s_, _ in SETS]
            ph, pa = _pooled_key([g["incumbent"] for g in got], "headline_gt_heldout"), _pooled_key([g["incumbent"] for g in got], "headline_gt")
            ids = " + ".join(f"<code>{esc(g['incumbent_id'])}</code>" for g in got)
            if ph and pa:
                pooled += (f"<tr><td>recursion {lvl} · {ids}</td><td class=num><b>{fmt(ph['snapped_f1_075'])}</b></td><td class=num>{fmt(ph['raw_f1_075'])}</td>"
                           f"<td class=num>{ph['n_pred']}/{ph['n_ref']}</td><td class=num>{fmt(pa['snapped_f1_075'])}</td><td class=num>{fmt(pa['raw_f1_075'])}</td></tr>")
    return f"""<h2>2. Human-in-the-loop: 10% labeled sample shared with the agents</h2>
<p class=muted>As in production annotation, humans label a small sample after each pass. Here a fixed ~10% of each batch (whole instruction groups, chosen before the run)
is labeled: agents see those human segments next to the incumbent's predictions, and the gate promotes only if Segment F1@0.75 on that sample strictly improves.
The other 90% are never labeled for agents or the gate; their score is the honest measure of whether the fix generalized.</p>
{''.join(rows_by_set.values())}
<div class=card><h3>Pooled per recursion level</h3><table><tr><th>recursion level</th><th>held-out 90 snapped F1@0.75</th><th>held-out 90 raw F1@0.75</th>
<th>held-out pred/ref</th><th>all 100 snapped F1@0.75 (10 labeled)</th><th>all 100 raw F1@0.75</th></tr>{pooled}</table></div>"""


def _gate_rows(evo) -> str:
    rows = ""
    for it in (evo or {}).get("iterations", []):
        d = it.get("decision") or {}
        for name, c in (d.get("checks") or {}).items():
            rows += (f"<tr><td>{esc(it['iteration'])}</td><td>{esc(name)}</td><td>{esc(json.dumps(c.get('value'), default=str))[:300]}</td>"
                     f"<td>{esc(c.get('threshold'))}</td><td><span class='pill {'ok' if c.get('pass') else 'bad'}'>{'pass' if c.get('pass') else 'fail'}</span></td></tr>")
        if d.get("rollback"):
            rows += f"<tr><td>{esc(it['iteration'])}</td><td colspan=4 class=muted>rollback: {esc(d['rollback'])}</td></tr>"
        if it.get("stopped"):
            rows += f"<tr><td>{esc(it['iteration'])}</td><td colspan=4 class=muted>stopped: {esc(it['stopped'])}</td></tr>"
    return rows


def section_lineage(a: Path) -> str:
    blocks = []
    for name, title in SETS:
        lin, evo = _load(a / f"lineage{_sfx(name)}.json"), _load(a / f"evolution{_sfx(name)}.json")
        if not lin:
            continue
        items = []
        for e in lin.get("experiments", []):
            dec = e.get("decision")
            lf = e.get("label_free") or {}
            sg = e.get("shadow_gt") or {}
            cg, ig = sg.get("candidate") or {}, sg.get("incumbent") or {}
            rc = e.get("recursion_cost") or {}
            by = " · ".join(f"{k}: {v['calls']} calls (${v['cost_usd']:.4f})" for k, v in (rc.get("by_role") or {}).items())
            items.append(f"""<div class=card><h3>{esc(e['id'])} <span class="pill {'ok' if dec == 'PROMOTED' else 'bad'}">{esc(dec)}</span>
<span class=muted>parent {esc(e.get('parent_version'))}{' · topology change' if e.get('topology_change') else ''}</span></h3>
<p><b>Hypothesis.</b> {esc(e.get('hypothesis'))}</p>
<table><tr><th></th><th>label-free self-score (gate signal)</th><th>shadow GT snapped F1@0.75</th><th>shadow GT raw F1@0.75</th><th>shadow GT snapped F1@0.5</th></tr>
<tr><td>incumbent</td><td class=num><b>{fmt(lf.get('incumbent_self_score'))}</b></td><td class=num>{fmt(ig.get('snapped_f1_075'))}</td><td class=num>{fmt(ig.get('raw_f1_075'))}</td><td class=num>{fmt(ig.get('snapped_f1_05'))}</td></tr>
<tr><td>candidate</td><td class=num><b>{fmt(lf.get('candidate_self_score'))}</b></td><td class=num>{fmt(cg.get('snapped_f1_075'))}</td><td class=num>{fmt(cg.get('raw_f1_075'))}</td><td class=num>{fmt(cg.get('snapped_f1_05'))}</td></tr></table>
<p class=muted>Label-free components (candidate vs incumbent): {esc(json.dumps(lf.get('candidate_components')))} vs {esc(json.dumps(lf.get('incumbent_components')))}</p>
<p class=muted>Recursion cost: ${fmt(rc.get('total_cost_usd'), 4)} ({esc(by)})</p>
{('<p class=muted>Errors: ' + esc(e.get('errors')) + '</p>') if e.get('errors') else ''}</div>""")
        blocks.append(f"""<h3>{esc(title)}: incumbent now <code>{esc(lin.get('incumbent'))}</code></h3>{''.join(items) or '<div class=card>No experiments yet.</div>'}
<div class=card><h3>Promotion gate checks</h3><table><tr><th>iter</th><th>check</th><th>value</th><th>threshold</th><th>result</th></tr>{_gate_rows(evo)}</table></div>""")
    return ("""<h2>2. Version N → N+1 lineage (label-free gate)</h2>
<p class=muted>Agents and the promotion gate see <b>no ground truth</b>: only the system's own predictions, video frames, robot telemetry and a predeclared label-free
self-score. "Shadow GT" is measured by the protected evaluator after each decision and is never shown to agents or the gate. Rejected candidates are frozen, not deleted.</p>"""
            + ("".join(blocks) or "<div class=card>No evolution run yet.</div>"))


def section_per_episode(a: Path) -> str:
    blocks = []
    for name, title in SETS:
        lin = _load(a / f"lineage{_sfx(name)}.json")
        levels = recursion_levels(lin, _load(a / f"baseline_v3{_sfx(name)}.json"), name)
        if not levels:
            continue
        first, last = levels[0]["incumbent"], levels[-1]["incumbent"]
        cands = [(L["candidate_id"], L["candidate"]) for L in levels[1:] if L["candidate"]]
        cols = [("N0", first)] + cands + ([(f"final incumbent {levels[-1]['incumbent_id']}", last)] if len(levels) > 1 else [])
        per = {k: ((r or {}).get("headline_gt") or {}).get("per_episode_snapped_f1") or {} for k, r in cols}
        eps = sorted(per.get("N0", {}))
        head = "".join(f"<th>{esc(k)}</th>" for k, _ in cols)
        body = "".join(f"<tr><td><code>{esc(e)}</code></td>" + "".join(f"<td class=num>{fmt(per[k].get(e))}</td>" for k, _ in cols) + "</tr>" for e in eps)
        blocks.append(f"<div class=card><h3>{esc(title)}</h3><table><tr><th>episode</th>{head}</tr>{body}</table></div>")
    return "<h2>6. Per-episode snapped F1@0.75</h2>" + ("".join(blocks) or "<div class=card>No results yet.</div>")


def section_tree(evo) -> str:
    if not evo:
        return "<h2>4. Recursive task tree</h2><div class=card>No evolution run yet.</div>"
    tasks = evo.get("task_tree", [])
    by_parent: dict[Any, list] = {}
    for t in tasks:
        by_parent.setdefault(t.get("parent_task_id"), []).append(t)
    evidence = evo.get("evidence", {})

    def render(pid) -> str:
        out = ""
        for t in by_parent.get(pid, []):
            r = t.get("result") or {}
            st = t.get("status")
            cls = "ok" if st == "done" else "bad"
            ev = ", ".join(f"<code title='{esc(evidence.get(e, {}).get('tool'))}'>{esc(e)}</code>" for e in (r.get("evidence_ids") or [])[:6])
            tools = ", ".join(sorted({evidence[e]["tool"] for e in t.get("evidence_ids", []) if e in evidence}))
            body = (f"<div>{esc(r.get('finding'))}</div><div class=muted>implication: {esc(r.get('implication_for_workflow'))}</div>"
                    f"<div class=muted>tools run: {esc(tools) or '–'} · cites: {ev or '–'} · model: {esc(t.get('model_used'))}</div>") if st == "done" else \
                   f"<div class=muted>{esc(r.get('dispatch_decision') or r.get('error'))}</div>"
            out += f"<li><span class=q>[depth {t['depth']}] {esc(t['question'])}</span> <span class='pill {cls}'>{esc(st)}</span>{body}{render(t['id'])}</li>"
        return f"<ul class=tree>{out}</ul>" if out else ""

    plans = "".join(f"<p><b>Iteration {esc(it['iteration'])} supervisor:</b> {esc((it.get('plan') or {}).get('failure_summary'))}</p>"
                    for it in evo.get("iterations", []))
    return f"""<h2>4. Recursive task tree (real dispatch)</h2>
<p class=muted>Max depth actually reached: <b>{esc(evo.get('max_depth_reached'))}</b> (limit 3). Children exist only where an investigator raised an evidence-cited new question
that passed the dispatcher's novelty, cycle, depth, fan-out and budget gates. Rejected/deduplicated requests are shown too.</p>
<div class=card>{plans}{render(None)}</div>"""


def _diff(a: str, b: str) -> str:
    out = []
    for line in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=2):
        if line.startswith(("---", "+++")):
            continue
        cls = "add" if line.startswith("+") else "del" if line.startswith("-") else ""
        out.append(f"<div class='{cls}'>{esc(line)}</div>")
    return "<pre>" + "".join(out) + "</pre>" if out else "<p class=muted>no change</p>"


def section_diff(s) -> str:
    root = s.artifacts_dir / "versions"
    blocks = []
    for name, _ in SETS:
        for e in (_load(s.artifacts_dir / f"lineage{_sfx(name)}.json") or {}).get("experiments", []):
            vd = e.get("version_dir")
            if not vd or not (root / vd).exists() or not (root / e["parent_version"]).exists():
                continue
            p, c = root / e["parent_version"], root / vd
            wf_a = json.dumps(json.loads((p / "workflow.json").read_text()), indent=1)
            wf_b = json.dumps(json.loads((c / "workflow.json").read_text()), indent=1)
            pr = ""
            for fn in sorted({f.name for f in p.glob("*.md")} | {f.name for f in c.glob("*.md")}):
                x = (p / fn).read_text() if (p / fn).exists() else ""
                y = (c / fn).read_text() if (c / fn).exists() else ""
                if x != y:
                    pr += f"<h3>{fn}</h3>{_diff(x, y)}"
            nodes_a = " → ".join(n["operator"] for n in json.loads(wf_a)["nodes"])
            nodes_b = " → ".join(n["operator"] for n in json.loads(wf_b)["nodes"])
            blocks.append(f"""<div class=card><h3>{esc(e['parent_version'])} → {esc(vd)}</h3>
<p class=muted>before: {esc(nodes_a)}<br>after: {esc(nodes_b)}</p><h3>workflow.json</h3>{_diff(wf_a, wf_b)}{pr}</div>""")
    return "<h2>5. Executable harness diff</h2>" + ("".join(blocks) or "<div class=card class=muted>No candidate versions.</div>")


def build_report() -> Path:
    s = get_settings()
    store = get_store()
    a = s.artifacts_dir
    pre = _load(a / "preflight.json")
    trees = ""
    for name, title in SETS:
        evo = _load(a / f"evolution{_sfx(name)}.json")
        if evo:
            trees += f"<h3>{esc(title)}</h3>" + section_tree(evo).split("</h2>", 1)[1]
    if trees:
        trees = section_tree({"task_tree": [], "iterations": []}).split("<div class=card>")[0] + trees
    body = f"""<main><h1>EvoAction evidence report</h1>
<p class=muted>A self-evolving, recursively delegated annotation harness for robot manipulation video, evaluated on WGO-Bench temporal subtask segmentation
with Macrodata's protocol. Every number and agent message on this page comes from persisted real runs in <code>artifacts/</code>.</p>
{section_results(a)}{section_hitl(a)}{section_lineage(a)}{trees or section_tree(None)}{section_diff(s)}{section_per_episode(a)}{section_sponsor(store, pre)}</main>"""
    doc = f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>EvoAction Evidence Report</title><style>{CSS}</style></head><body>{body}</body></html>"
    out = a / "report.html"
    out.write_text(doc)
    return out


def cmd_report(args: argparse.Namespace) -> int:
    print(build_report())
    return 0
