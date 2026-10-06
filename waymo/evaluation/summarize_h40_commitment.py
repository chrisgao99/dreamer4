"""Summarize completed H40 commitment experiments; launcher holds a file lock."""
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for commitment in (40, 15, 5):
    path = root / f"b{commitment}" / "result.json"
    if not path.exists():
        continue
    result = json.loads(path.read_text())
    assert result["checkpoint_step"] == 95000
    assert result["scene_count"] == 512
    assert result["model_plan_horizon"] == 40
    assert result["rollout_commitment_steps"] == commitment
    rows.append({
        "commitment_steps": commitment,
        "scenes": result["scene_count"],
        "mean_ade_m": result["mean_ade_m"],
        "minade_at_32_m": result["minade_m"],
        "cpd": result["flow_erd_cpd"],
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
print(f"Completed {len(rows)}/3; summary: {root / 'summary.csv'}")
