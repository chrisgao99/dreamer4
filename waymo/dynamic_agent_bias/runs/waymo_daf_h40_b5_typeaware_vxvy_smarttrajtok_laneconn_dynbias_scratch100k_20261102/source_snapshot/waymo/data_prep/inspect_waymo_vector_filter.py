"""Inspect the output shapes of the Waymo vector filter."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

WAYMO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = WAYMO_ROOT / "core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from waymo_vector_filter import WaymoVectorConfig, iter_filtered_scenarios, summarize_filtered


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tfrecord", type=str)
    p.add_argument("--max_records", type=int, default=1)
    p.add_argument("--num_agents", type=int, default=32)
    p.add_argument("--agent_distance_threshold", type=float, default=80.0)
    p.add_argument("--map_distance_threshold", type=float, default=100.0)
    p.add_argument("--max_map_polylines", type=int, default=256)
    p.add_argument("--max_points_per_polyline", type=int, default=20)
    p.add_argument("--history_only_selection", action="store_true")
    args = p.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    cfg = WaymoVectorConfig(
        num_agents=args.num_agents,
        agent_distance_threshold=args.agent_distance_threshold,
        map_distance_threshold=args.map_distance_threshold,
        max_map_polylines=args.max_map_polylines,
        max_points_per_polyline=args.max_points_per_polyline,
        use_all_timesteps_for_selection=not args.history_only_selection,
    )

    for idx, item in enumerate(iter_filtered_scenarios(args.tfrecord, cfg=cfg, max_records=args.max_records)):
        summary = summarize_filtered(item)
        summary["record_index"] = idx
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
