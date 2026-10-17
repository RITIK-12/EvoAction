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
    rows = store.query("SELECT r.kind, m.role, m.model, COUNT(*) n, SUM(m.cache_hit) ch, SUM(CASE WHEN m.status!='ok' THEN 1 ELSE 0 END) failed, "
                       "COALESCE(SUM(m.prompt_tokens),0) pt, COALESCE(SUM(m.completion_tokens),0) ct, COALESCE(SUM(m.cost_usd),0) cost, "
                       "COALESCE(SUM(m.latency_s),0) lat FROM model_calls m LEFT JOIN runs r ON m.run_id = r.id "
                       "GROUP BY r.kind, m.role, m.model ORDER BY r.kind, m.role")
    t = "".join(f"<tr><td>{esc(r['kind'])}</td><td>{esc(r['role'])}</td><td><code>{esc(r['model'])}</code></td><td class=num>{r['n']}</td>"
                f"<td class=num>{r['ch'] or 0}</td><td class=num>{r['failed']}</td><td class=num>{r['pt']:,}</td><td class=num>{r['ct']:,}</td>"
                f"<td class=num>${r['cost']:.4f}</td><td class=num>{r['lat']:.0f}s</td></tr>" for r in rows)
    tot = store.query("SELECT COUNT(*) n, COALESCE(SUM(cost_usd),0) c FROM model_calls WHERE cache_hit=0")[0]
    locked = (pre or {}).get("locked", {})
    return f"""<h2>1. Sponsor-backed execution (Crusoe Managed Inference, NVIDIA Nemotron)</h2>
<div class=card><p>Reasoning / supervisor / investigators / strategy / critic: <code>{esc(locked.get('CRUSOE_REASONING_MODEL'))}</code><br>
Vision annotation and refinement: <code>{esc(locked.get('CRUSOE_VISION_MODEL'))}</code><br>
<span class=muted>Model IDs confirmed live via <code>GET /v1/models</code> in preflight. Every row below is a persisted real call (cache hits re-serve an earlier real response and cost $0).
Total non-cached calls: {tot['n']}, provider-token cost ${tot['c']:.4f} at listed per-token prices.</span></p>
<table><tr><th>run</th><th>role</th><th>model</th><th>calls</th><th>cache hits</th><th>failed attempts</th><th>prompt tok</th><th>completion tok</th><th>cost</th><th>model time</th></tr>{t}</table></div>"""


def section_lineage(lin, evo) -> str:
    if not lin:
        return "<h2>2. Version lineage</h2><div class=card class=muted>No evolution run yet.</div>"
    items = []
    for e in lin.get("experiments", []):
        dec = e.get("decision")
        cls = "ok" if dec == "PROMOTED" else "bad"
        cv, iv = e.get("candidate_val") or {}, e.get("incumbent_val") or {}
        items.append(f"""<div class=card><h3>{esc(e['id'])} <span class="pill {cls}">{esc(dec)}</span> <span class=muted>parent {esc(e.get('parent_version'))}
{' · topology change' if e.get('topology_change') else ''}</span></h3>
<p><b>Hypothesis.</b> {esc(e.get('hypothesis'))}</p>
<table><tr><th></th><th>val F1@0.75</th><th>val F1@0.5</th><th>short recall</th><th>boundary MAE s</th></tr>
<tr><td>incumbent</td><td class=num>{fmt(iv.get('micro_f1'))}</td><td class=num>{fmt(iv.get('micro_f1_iou05'))}</td><td class=num>{fmt(iv.get('short_event_recall'))}</td><td class=num>{fmt(iv.get('mean_boundary_mae_sec'),2)}</td></tr>
<tr><td>candidate</td><td class=num>{fmt(cv.get('micro_f1'))}</td><td class=num>{fmt(cv.get('micro_f1_iou05'))}</td><td class=num>{fmt(cv.get('short_event_recall'))}</td><td class=num>{fmt(cv.get('mean_boundary_mae_sec'),2)}</td></tr></table>
{('<p class=muted>Errors: ' + esc(e.get('errors')) + '</p>') if e.get('errors') else ''}</div>""")
    gate_rows = ""
    for it in (evo or {}).get("iterations", []):
        d = it.get("decision") or {}
        for name, c in (d.get("checks") or {}).items():
            gate_rows += (f"<tr><td>{esc(it['iteration'])}</td><td>{esc(name)}</td><td>{esc(json.dumps(c.get('value'), default=str))}</td>"
                          f"<td>{esc(c.get('threshold'))}</td><td><span class='pill {'ok' if c.get('pass') else 'bad'}'>{'pass' if c.get('pass') else 'fail'}</span></td></tr>")
        if d.get("rollback"):
            gate_rows += f"<tr><td>{esc(it['iteration'])}</td><td colspan=4 class=muted>rollback: {esc(d['rollback'])}</td></tr>"
        if it.get("stopped"):
            gate_rows += f"<tr><td>{esc(it['iteration'])}</td><td colspan=4 class=muted>stopped: {esc(it['stopped'])}</td></tr>"
    return f"""<h2>2. Version N → N+1 lineage</h2>
<p class=muted>Incumbent now: <b>{esc(lin.get('incumbent'))}</b>. The controller applies a predeclared gate; agents never decide promotion. Rejected candidates are frozen, not deleted.</p>
{''.join(items) or '<div class=card>No experiments recorded.</div>'}
<div class=card><h3>Promotion gate checks</h3><table><tr><th>iter</th><th>check</th><th>value</th><th>threshold</th><th>result</th></tr>{gate_rows}</table></div>"""


def section_tree(evo) -> str:
    if not evo:
        return "<h2>3. Recursive task tree</h2><div class=card>No evolution run yet.</div>"
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
    return f"""<h2>3. Recursive task tree (real dispatch)</h2>
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


def section_diff(s, lin) -> str:
    root = s.artifacts_dir / "versions"
    blocks = []
    for e in (lin or {}).get("experiments", []):
        vd = e.get("version_dir")
        if not vd or not (root / vd).exists() or not (root / e["parent_version"]).exists():
            continue
        p, c = root / e["parent_version"], root / vd
        wf_a = json.dumps(json.loads((p / "workflow.json").read_text()), indent=1)
        wf_b = json.dumps(json.loads((c / "workflow.json").read_text()), indent=1)
        pr = ""
        for name in ("prompt.md", "refine.md"):
            a = (p / name).read_text() if (p / name).exists() else ""
            b = (c / name).read_text() if (c / name).exists() else ""
            if a != b:
                pr += f"<h3>{name}</h3>{_diff(a, b)}"
        nodes_a = " → ".join(n["operator"] for n in json.loads(wf_a)["nodes"])
        nodes_b = " → ".join(n["operator"] for n in json.loads(wf_b)["nodes"])
        blocks.append(f"""<div class=card><h3>{esc(e['parent_version'])} → {esc(vd)}</h3>
<p class=muted>before: {esc(nodes_a)}<br>after: {esc(nodes_b)}</p><h3>workflow.json</h3>{_diff(wf_a, wf_b)}{pr}</div>""")
    return "<h2>4. Executable harness diff</h2>" + ("".join(blocks) or "<div class=card class=muted>No candidate versions.</div>")


def _timeline_svg(tl: dict[str, Any]) -> str:
    dur = float(tl.get("duration_sec") or max((s["end_sec"] for s in tl["reference"]), default=1.0))
    W, H, pad, row = 860, 0, 70, 22
    rows = [("reference", "var(--ref)")] + [(k, f"var(--{k})") for k in ("A", "B", "C") if k in tl]
    H = row * len(rows) + 24
    svg = [f"<svg viewBox='0 0 {W} {H}' width='100%' role='img' aria-label='timeline'>"]
    for i, (name, color) in enumerate(rows):
        y = 4 + i * row
        svg.append(f"<text x='0' y='{y + 13}' font-size='11' fill='var(--muted)'>{esc(name)}</text>")
        for seg in tl[name]:
            x0 = pad + (W - pad) * seg["start_sec"] / dur
            x1 = pad + (W - pad) * seg["end_sec"] / dur
            svg.append(f"<rect x='{x0:.1f}' y='{y}' width='{max(1.0, x1 - x0 - 1.5):.1f}' height='16' rx='2' fill='{color}' opacity='0.85'>"
                       f"<title>{esc(seg.get('subtask'))} [{seg['start_sec']:.2f}-{seg['end_sec']:.2f}s]</title></rect>")
    for t in range(0, int(dur) + 1, max(1, int(dur // 10) or 1)):
        x = pad + (W - pad) * t / dur
        svg.append(f"<text x='{x:.1f}' y='{H - 4}' font-size='10' fill='var(--muted)' text-anchor='middle'>{t}s</text>")
    svg.append("</svg>")
    return "".join(svg)


def section_final(fin, base) -> str:
    if not fin:
        btab = ""
        if base:
            btab = "".join(f"<tr><td>{esc(r['version'])}</td><td>{esc(r['split'])}</td><td class=num>{esc(r['micro_f1@0.75'])}</td><td class=num>{esc(r['micro_f1@0.5'])}</td>"
                           f"<td class=num>{esc(r['n_pred/n_ref'])}</td><td>{esc(r['statuses'])}</td></tr>" for r in base.get("results", []))
        return f"""<h2>5. Sealed held-out scoreboard</h2><div class=card>The sealed test has not been run yet.</div>
<div class=card><h3>Fixed baselines on dev/validation (pre-optimization)</h3><table><tr><th>version</th><th>split</th><th>F1@0.75</th><th>F1@0.5</th><th>pred/ref</th><th>status</th></tr>{btab}</table></div>"""
    sm = fin["summary"]
    rows = ""
    names = {"A": "A · N000 (1 Hz, simple prompt)", "B": "B · strong fixed (2 Hz, world-state prompt, no agents)", "C": f"C · {fin.get('C_label')}"}
    for k in "ABC":
        x = sm[k]
        rows += (f"<tr><td><span class=sw style='background:var(--{k})'></span>{esc(names[k])}<br><code>{esc(x['version_id'])}</code></td>"
                 f"<td class=num><b>{fmt(x['micro_f1'])}</b></td><td class=num>{fmt(x['micro_f1_iou05'])}</td><td class=num>{fmt(x['outer_snapped_micro_f1'])}</td>"
                 f"<td class=num>{fmt(x['short_event_recall'])}</td><td class=num>{fmt(x['mean_boundary_mae_sec'], 2)}</td>"
                 f"<td class=num>{x['invalid_items']}/{x['empty_output_episodes']}</td><td class=num>{x['vision_calls']}</td>"
                 f"<td class=num>{x['annotation_tokens']['completion']:,}</td><td class=num>${x['annotation_cost_usd']:.4f}</td><td class=num>{x['wall_s']:.0f}s</td></tr>")
    eps = fin["test_episode_ids"]
    per = "".join(f"<tr><td><code>{esc(e)}</code></td>" + "".join(f"<td class=num>{fmt(sm[k]['per_episode_f1'].get(e))}</td>" for k in "ABC")
                  + f"<td class=num>{fmt(sm['C']['per_episode_f1'].get(e, 0) - sm['B']['per_episode_f1'].get(e, 0))}</td></tr>" for e in eps)
    tls = "".join(f"<h3><code>{esc(e)}</code></h3>{_timeline_svg(tl)}" for e, tl in fin["timelines"].items())
    ovh = fin.get("optimization_overhead", {})
    ovh_rows = "".join(f"<tr><td>{esc(kind)}</td><td><code>{esc(m)}</code></td><td class=num>{u['calls']}</td><td class=num>{u['completion_tokens']:,}</td><td class=num>${u['cost_usd']:.4f}</td></tr>"
                       for kind in ("evolve_runs", "baseline_runs") for m, u in (ovh.get(kind) or {}).items())
    p = fin["paired"]
    lim = "".join(f"<li>{esc(x)}</li>" for x in fin.get("limitations", []))
    return f"""<h2>5. Sealed held-out scoreboard (run once after freezing)</h2>
<div class=card><p class=muted>{len(eps)} test episodes never seen by any agent tool. Systems frozen with checksums before the test evaluator was unsealed. Timelines below are produced only by the final evaluator.</p>
<table><tr><th>system</th><th>F1@0.75</th><th>F1@0.5</th><th>F1@0.75 outer-snapped</th><th>short recall</th><th>boundary MAE s</th><th>invalid/empty</th><th>vision calls</th><th>completion tok</th><th>annotation cost</th><th>wall</th></tr>{rows}</table></div>
<div class=card><h3>Paired per-episode F1@0.75</h3><table><tr><th>episode</th><th>A</th><th>B</th><th>C</th><th>C−B</th></tr>{per}</table>
<p class=muted>C vs A: {p['C_vs_A']['improved']} improved / {p['C_vs_A']['regressed']} regressed · C vs B: {p['C_vs_B']['improved']} improved / {p['C_vs_B']['regressed']} regressed · B vs A: {p['B_vs_A']['improved']} improved / {p['B_vs_A']['regressed']} regressed</p></div>
<div class=card><h3>Optimization overhead (all supervisor, investigation, experiment and baseline calls)</h3><table><tr><th>phase</th><th>model</th><th>calls</th><th>completion tok</th><th>cost</th></tr>{ovh_rows}</table></div>
<div class=card><div class=legend><span><i class=sw style='background:var(--ref)'></i>reference</span><span><i class=sw style='background:var(--A)'></i>A</span><span><i class=sw style='background:var(--B)'></i>B</span><span><i class=sw style='background:var(--C)'></i>C</span></div>{tls}</div>
<div class=card><h3>Limitations</h3><ul>{lim}</ul></div>"""


def build_report() -> Path:
    s = get_settings()
    store = get_store()
    a = s.artifacts_dir
    pre, splits, base = _load(a / "preflight.json"), _load(a / "splits.json"), _load(a / "baselines.json")
    evo, lin, fin = _load(a / "evolution.json"), _load(a / "lineage.json"), _load(a / "final_test.json")
    split_line = ""
    if splits:
        split_line = " · ".join(f"{k}: {len(v['episode_ids'])} (pilot {len(v['pilot_episode_ids'])})" for k, v in splits["splits"].items())
    body = f"""<main><h1>EvoAction evidence report</h1>
<p class=muted>A self-evolving, recursively delegated annotation harness for robot manipulation video (WGO-Bench temporal subtask segmentation).
Every number and agent message on this page comes from persisted real runs in <code>artifacts/</code>. Splits: {esc(split_line)}; splits sha256 <code>{esc((splits or {}).get('splits_sha256', '')[:16])}</code>.</p>
{section_sponsor(store, pre)}{section_lineage(lin, evo)}{section_tree(evo)}{section_diff(s, lin)}{section_final(fin, base)}</main>"""
    doc = f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>EvoAction Evidence Report</title><style>{CSS}</style></head><body>{body}</body></html>"
    out = a / "report.html"
    out.write_text(doc)
    return out


def cmd_report(args: argparse.Namespace) -> int:
    print(build_report())
    return 0
