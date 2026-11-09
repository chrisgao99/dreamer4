#!/usr/bin/env python3
"""Estimate explicit per-type Flow-ERD CPD scales from Waymo training NPZs.

The Flow-ERD paper leaves ``sigma_c`` underspecified.  This utility implements
the documented convention used by this repository:

    sigma_c = sqrt(E_train[||p_t^i - p_0^i||_2^2])

where p_0 is the final context pose and the expectation covers valid logged
future agent-time entries of type c.  The output records all sufficient sums so
the convention and result remain auditable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


TYPE_IDS = (1, 2, 3)
TYPE_NAMES = ("vehicle", "pedestrian", "cyclist")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--context_index", type=int, default=10)
    parser.add_argument("--future_steps", type=int, default=80)
    return parser


def main(args: argparse.Namespace) -> None:
    paths = sorted(Path(args.data_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No NPZ files found in {args.data_dir}")

    squared_displacement_sum = {type_id: 0.0 for type_id in TYPE_IDS}
    valid_agent_time_count = {type_id: 0 for type_id in TYPE_IDS}
    valid_agent_count = {type_id: 0 for type_id in TYPE_IDS}
    context_index = int(args.context_index)
    future_end = context_index + 1 + int(args.future_steps)

    for file_index, path in enumerate(paths, start=1):
        with np.load(path, allow_pickle=False) as data:
            agents = np.asarray(data["agents"], dtype=np.float64)
            agent_mask = np.asarray(data["agent_mask"], dtype=bool)
        if agents.ndim != 3 or agents.shape[-1] < 8:
            raise ValueError(f"Unexpected agents shape in {path}: {agents.shape}")
        if context_index < 0 or future_end > agents.shape[1]:
            raise ValueError(
                f"Requested context/future [{context_index}, {future_end}) outside {path} shape {agents.shape}"
            )

        context = agents[:, context_index]
        future = agents[:, context_index + 1 : future_end]
        roster_valid = agent_mask & (context[:, 5] > 0.5)
        agent_type = np.rint(context[:, 7]).astype(np.int64)
        future_valid = future[..., 5] > 0.5
        displacement_squared = np.square(future[..., 0:2] - context[:, None, 0:2]).sum(axis=-1)

        for type_id in TYPE_IDS:
            selected_agents = roster_valid & (agent_type == type_id)
            selected = selected_agents[:, None] & future_valid
            squared_displacement_sum[type_id] += float(displacement_squared[selected].sum())
            valid_agent_time_count[type_id] += int(selected.sum())
            valid_agent_count[type_id] += int(selected_agents.sum())

        if file_index % 5000 == 0 or file_index == len(paths):
            print(f"processed {file_index}/{len(paths)}", flush=True)

    scales = {}
    for type_id, type_name in zip(TYPE_IDS, TYPE_NAMES):
        count = valid_agent_time_count[type_id]
        if count <= 0:
            raise RuntimeError(f"No valid training entries found for {type_name}")
        scales[type_name] = math.sqrt(squared_displacement_sum[type_id] / count)

    payload = {
        "definition": "sigma_c = sqrt(E_train[||p_t - p_0||_2^2])",
        "data_dir": str(Path(args.data_dir).resolve()),
        "num_files": len(paths),
        "context_index": context_index,
        "future_steps": int(args.future_steps),
        "type_order": list(TYPE_NAMES),
        "type_ids": list(TYPE_IDS),
        "type_scales_m": scales,
        "squared_displacement_sum_m2": {
            name: squared_displacement_sum[type_id]
            for type_id, name in zip(TYPE_IDS, TYPE_NAMES)
        },
        "valid_agent_time_count": {
            name: valid_agent_time_count[type_id]
            for type_id, name in zip(TYPE_IDS, TYPE_NAMES)
        },
        "valid_agent_count": {
            name: valid_agent_count[type_id]
            for type_id, name in zip(TYPE_IDS, TYPE_NAMES)
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(build_argparser().parse_args())
