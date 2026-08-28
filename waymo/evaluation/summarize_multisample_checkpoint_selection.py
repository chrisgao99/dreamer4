"""Rank fixed-protocol multi-rollout validation JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from waymo.training.world_model.multisample_validation import multisample_selection_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--reference_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--diversity_floor_ratio", type=float, default=0.5)
    parser.add_argument("--glob_pattern", type=str, default="*_h90_n8_val32.json")
    args = parser.parse_args()

    reference_payload = json.loads(args.reference_json.read_text())
    reference = reference_payload["metrics"]["h90"]
    rows = []
    for path in sorted(args.input_dir.glob(args.glob_pattern)):
        payload = json.loads(path.read_text())
        metrics = payload["metrics"]["h90"]
        selection = multisample_selection_score(
            metrics,
            reference,
            diversity_floor_ratio=args.diversity_floor_ratio,
        )
        rows.append(
            {
                "result_json": str(path.resolve()),
                "checkpoint": payload["eval_ckpt"],
                "step": int(payload.get("ckpt_step", -1)),
                "score": selection["checkpoint_selection_score"],
                "eligible": bool(selection["checkpoint_selection_eligible"] > 0.5),
                "diversity_ratio": selection["checkpoint_diversity_ratio_to_reference"],
                "nonfocus_minade_m": metrics["multisample_nonfocus_minade_m"],
                "nonfocus_ade_winner_fde_m": metrics[
                    "multisample_nonfocus_ade_winner_fde_m"
                ],
                "nonfocus_mean_ade_m": metrics["multisample_nonfocus_mean_ade_m"],
                "nonfocus_mean_fde_m": metrics["multisample_nonfocus_mean_fde_m"],
                "nonfocus_worst_ade_m": metrics["multisample_nonfocus_worst_ade_m"],
                "pairwise_trajectory_distance_m": metrics[
                    "multisample_nonfocus_pairwise_trajectory_distance_m"
                ],
                "pairwise_endpoint_distance_m": metrics[
                    "multisample_nonfocus_pairwise_endpoint_distance_m"
                ],
                "collision_overlap_rate_proxy": metrics["collision_overlap_rate_proxy"],
                "offroad_rate_proxy": metrics["offroad_rate_proxy"],
                "kinematic_xy_mae_m": metrics["kinematic_xy_mae_m"],
            }
        )

    rows.sort(key=lambda row: (not row["eligible"], row["score"]))
    recommended = next((row for row in rows if row["eligible"]), None)
    output = {
        "protocol": args.glob_pattern,
        "reference_json": str(args.reference_json.resolve()),
        "diversity_floor_ratio": float(args.diversity_floor_ratio),
        "recommended": recommended,
        "ranking": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    print(
        "eligible score div_ratio minADE meanADE worstADE collision offroad checkpoint",
        flush=True,
    )
    for row in rows:
        print(
            f"{int(row['eligible'])} {row['score']:.4f} {row['diversity_ratio']:.3f} "
            f"{row['nonfocus_minade_m']:.3f} {row['nonfocus_mean_ade_m']:.3f} "
            f"{row['nonfocus_worst_ade_m']:.3f} {row['collision_overlap_rate_proxy']:.4f} "
            f"{row['offroad_rate_proxy']:.4f} {row['checkpoint']}",
            flush=True,
        )
    print(f"wrote summary: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
