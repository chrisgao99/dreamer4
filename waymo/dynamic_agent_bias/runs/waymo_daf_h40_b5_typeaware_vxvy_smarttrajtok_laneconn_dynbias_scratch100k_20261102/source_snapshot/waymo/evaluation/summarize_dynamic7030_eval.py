"""Summarize completed dynamic/static 70/30 experiments; launcher holds a file lock."""
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for horizon, step, commitment in ((40, 95000, 40), (40, 95000, 15), (40, 95000, 5), (15, 90000, 15), (15, 90000, 5)):
    path = root / f"h{horizon}_b{commitment}" / "result.json"
    if not path.exists() or not (path.parent / "ade_by_gt_motion.json").exists():
        continue
    result = json.loads(path.read_text())
    assert result["checkpoint_step"] == step
    assert result["scene_count"] == 512
    assert result["model_plan_horizon"] == horizon
    assert result["rollout_commitment_steps"] == commitment
    motion = json.loads((path.parent / "ade_by_gt_motion.json").read_text())
    groups = motion["groups"]["full80/net_displacement"]
    rows.append({
        "plan_steps": horizon,
        "checkpoint_step": step,
        "commitment_steps": commitment,
        "scenes": result["scene_count"],
        "mean_ade_m": result["mean_ade_m"],
        "minade_at_32_m": result["minade_m"],
        "cpd": result["flow_erd_cpd"],
        **{f"motion_{group['group']}_mean_ade_m": group["mean_ade_m"] for group in groups},
        **result["physical_metrics"],
        **{f"gt_{name}": value for name, value in result["ground_truth_physical_metrics"].items()},
    })
if rows:
    temporary = root / "summary.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "summary.csv")
print(f"Completed {len(rows)}/5; summary: {root / 'summary.csv'}")
