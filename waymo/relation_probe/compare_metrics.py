"""Compare raw_only/raw_z/raw_shuffled_z relation probe metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_metrics(run_dir: str, metrics_name: str) -> Dict[str, Any]:
    path = Path(run_dir) / metrics_name
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        if value != value:
            return "nan"
        return f"{float(value):.6f}"
    return str(value)


def compare(args: argparse.Namespace) -> None:
    names = args.name or [Path(p).name for p in args.run_dir]
    metrics = [load_metrics(p, args.metrics_name) for p in args.run_dir]
    rows: List[Dict[str, Any]] = []

    summary_keys = sorted(set().union(*(m.get("summary", {}).keys() for m in metrics)))
    for key in summary_keys:
        row = {"group": "summary", "label": key, "metric": key}
        for name, m in zip(names, metrics):
            row[name] = m.get("summary", {}).get(key, float("nan"))
        rows.append(row)

    reg_names = sorted(set().union(*(m.get("regression", {}).keys() for m in metrics)))
    for label in reg_names:
        row = {"group": "regression", "label": label, "metric": "mae"}
        for name, m in zip(names, metrics):
            row[name] = m.get("regression", {}).get(label, {}).get("mae", float("nan"))
        rows.append(row)

    bin_names = sorted(set().union(*(m.get("binary", {}).keys() for m in metrics)))
    for label in bin_names:
        for metric_name in ("ap", "auroc", "pos_rate"):
            row = {"group": "binary", "label": label, "metric": metric_name}
            for name, m in zip(names, metrics):
                row[name] = m.get("binary", {}).get(label, {}).get(metric_name, float("nan"))
            rows.append(row)

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["group", "label", "metric"] + names)
            writer.writeheader()
            writer.writerows(rows)

    header = ["group", "label", "metric"] + names
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        print("| " + " | ".join(_fmt(row.get(col, "")) for col in header) + " |")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare relation probe metrics.")
    p.add_argument("--run_dir", type=str, nargs="+", required=True)
    p.add_argument("--name", type=str, nargs="*", default=None)
    p.add_argument("--metrics_name", type=str, default="best_metrics.json")
    p.add_argument("--output_csv", type=str, default="")
    return p


if __name__ == "__main__":
    compare(build_argparser().parse_args())

