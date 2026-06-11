"""PyTorch dataset wrapper for filtered Waymo vector-tokenizer NPZ files."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class WaymoVectorDataset(Dataset):
    """Loads NPZ files produced by `waymo_vector_filter.py`.

    Returned tensors use the tokenizer-friendly layout:

    - agents: (T, K, F_agent)
    - agent_mask: (K,)
    - map_polylines: (M, P, F_map)
    - map_mask: (M, P)
    - lights: (T, L, F_light)
    - light_mask: (T, L)
    """

    def __init__(self, roots: Union[str, Sequence[str]]):
        if isinstance(roots, (str, Path)):
            roots = [str(roots)]

        paths = []
        for root in roots:
            root = str(root)
            if root.endswith(".npz"):
                paths.append(root)
            else:
                paths.extend(glob.glob(str(Path(root) / "*.npz")))

        self.paths = sorted(set(paths))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No .npz files found in {roots}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx]
        with np.load(path, allow_pickle=False) as data:
            item = {
                "agents": torch.from_numpy(data["agents"]).float(),
                "agent_mask": torch.from_numpy(data["agent_mask"]).bool(),
                "agent_ids": torch.from_numpy(data["agent_ids"]).long(),
                "map_polylines": torch.from_numpy(data["map_polylines"]).float(),
                "map_mask": torch.from_numpy(data["map_mask"]).bool(),
                "map_ids": torch.from_numpy(data["map_ids"]).long(),
                "lights": torch.from_numpy(data["lights"]).float(),
                "light_mask": torch.from_numpy(data["light_mask"]).bool(),
                "light_ids": torch.from_numpy(data["light_ids"]).long(),
                "ego_origin_xy": torch.from_numpy(data["ego_origin_xy"]).float(),
                "ego_heading": torch.as_tensor(float(data["ego_heading"]), dtype=torch.float32),
            }
            if "scenario_id" in data:
                item["scenario_id"] = str(data["scenario_id"])

        item["path"] = path
        return item
