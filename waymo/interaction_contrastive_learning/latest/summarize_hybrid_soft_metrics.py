"""Create a self-contained quantitative report for hybrid-soft-v2 training."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


REPORT_METRICS = (
    "loss_contrastive",
    "loss_separation",
    "loss_rank_kl",
    "separation_accuracy",
    "positive_rank_accuracy",
    "cosine_margin",
    "positive_cosine_drop",
    "loss_reconstruction",
)


def metric_cell(record: dict, key: str) -> str:
    value = record.get(key)
    return "—" if value is None else f"{float(value):.5f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_html", type=Path, required=True)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.metrics.read_text().splitlines()
        if line.strip()
    ]
    evaluations = [record for record in records if record.get("kind") == "eval"]
    if not evaluations:
        raise RuntimeError(f"No evaluation records in {args.metrics}")
    stage_b = [record for record in evaluations if record.get("stage") == "B"]
    initial = evaluations[0]
    best = min(stage_b or evaluations, key=lambda record: record["loss_contrastive"])
    final = evaluations[-1]
    summary = {
        "initial": initial,
        "best_stage_b": best,
        "final": final,
        "evaluation_count": len(evaluations),
    }
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    header = "".join(f"<th>{html.escape(key)}</th>" for key in REPORT_METRICS)
    rows = []
    for label, record in (("initial", initial), ("best Stage B", best), ("final", final)):
        cells = "".join(f"<td>{metric_cell(record, key)}</td>" for key in REPORT_METRICS)
        rows.append(
            f"<tr><th>{html.escape(label)} (step {int(record['step'])})</th>{cells}</tr>"
        )
    embedded = json.dumps(evaluations).replace("</", "<\\/")
    report = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Hybrid Soft v2 Report</title>
<style>
body {{ font: 14px system-ui; margin: 24px; color: #18212b; }}
table {{ border-collapse: collapse; font-size: 12px; }}
th, td {{ border: 1px solid #cad2dc; padding: 7px; text-align: right; }}
th:first-child {{ text-align: left; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(420px,1fr)); gap: 16px; margin-top: 22px; }}
.card {{ border: 1px solid #d9e0e8; border-radius: 8px; padding: 12px; }}
canvas {{ width: 100%; height: 230px; }}
</style></head><body>
<h1>Hybrid Soft v2 Validation</h1>
<p>Lower losses are better; higher accuracies/margins are better. All curves use the fixed stratified validation manifest.</p>
<table><thead><tr><th>checkpoint</th>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class="grid" id="grid"></div>
<script>
const records={embedded};
const metrics={json.dumps(REPORT_METRICS)};
const grid=document.getElementById('grid');
function chart(metric) {{
  const points=records.filter(r => Number.isFinite(r[metric])).map(r => [r.step,r[metric]]);
  if (!points.length) return;
  const card=document.createElement('div'); card.className='card';
  const title=document.createElement('h3'); title.textContent=metric; card.appendChild(title);
  const canvas=document.createElement('canvas'); canvas.width=720; canvas.height=230; card.appendChild(canvas); grid.appendChild(card);
  const c=canvas.getContext('2d'), pad=34;
  const xs=points.map(p=>p[0]), ys=points.map(p=>p[1]);
  const xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=Math.min(...ys), ymax=Math.max(...ys), dy=Math.max(ymax-ymin,1e-8);
  c.strokeStyle='#d8dee8'; c.strokeRect(pad,8,canvas.width-pad-8,canvas.height-pad);
  c.strokeStyle='#2563eb'; c.lineWidth=2; c.beginPath();
  points.forEach((p,i)=>{{ const x=pad+(p[0]-xmin)/Math.max(xmax-xmin,1)*(canvas.width-pad-8); const y=8+(ymax-p[1])/dy*(canvas.height-pad-8); i?c.lineTo(x,y):c.moveTo(x,y); }}); c.stroke();
  c.fillStyle='#475569'; c.font='12px system-ui'; c.fillText(String(xmin),pad,canvas.height-7); c.fillText(String(xmax),canvas.width-45,canvas.height-7); c.fillText(ymax.toFixed(4),2,18); c.fillText(ymin.toFixed(4),2,canvas.height-pad);
}}
metrics.forEach(chart);
</script></body></html>"""
    args.output_html.write_text(report)
    print(f"saved {args.output_json}")
    print(f"saved {args.output_html}")


if __name__ == "__main__":
    main()
