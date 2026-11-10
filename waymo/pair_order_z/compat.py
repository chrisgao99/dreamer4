"""Cross-server compatibility: paths, sys.path, map-cache identity check.

Caches and checkpoints were written on the server where the tree lives at
``/p/yufeng/tri30``; here it is ``/scratch/baz7dy/tri30``. Every location is
resolved relative to ``TRI30`` (this file's ``tri30/`` ancestor).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

TRI30 = Path(__file__).resolve().parents[3]
DREAMER = TRI30 / "dreamer4"
for _p in (DREAMER, DREAMER / "waymo" / "core"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Old absolute prefix -> this server's prefix (override with ZPAIR_PATH_MAP="old=new").
_OLD, _NEW = os.environ.get("ZPAIR_PATH_MAP", "/p/yufeng/=/scratch/baz7dy/").split("=", 1)

DATA = DREAMER / "data"
TRAIN_DIR = DATA / "waymo_vector_dataset_ooi_centered_50k_with_lengths" / "train"
VAL_DIR = DATA / "waymo_vector_dataset_ooi_centered_50k_with_lengths" / "val"
MAP_CACHE = DATA / "waymo_map_smarttrajtok_5m_v2"
LANE_TOPOLOGY = DATA / "waymo_lane_connectivity_5m_v1"
STAGE1_CKPT = (DREAMER / "waymo" / "checkpoints"
               / "waymo_daf_h40_b5_typeaware_vxvy_smarttrajtok_laneconn_scratch100k_20261007" / "best.pt")
CPD_SCALES = (25.423524547767016, 4.925798640017075, 18.77286026475977)


def map_path(path: str) -> str:
    return path.replace(_OLD, _NEW, 1) if path.startswith(_OLD) else path


_patched = False


def patch_map_cache_identity() -> None:
    """Map-cache sidecars store the source NPZ's absolute path on the old server.

    Keep the strict identity check, but compare after mapping the old prefix.
    Everything else in ``MapAdaptationDataset.__getitem__`` is unchanged.
    """
    global _patched
    if _patched:
        return
    from waymo.adaptations import map_dataset as md
    from waymo.core.waymo_vector_dataset import WaymoVectorDataset

    def __getitem__(self, index):
        item = WaymoVectorDataset.__getitem__(self, index)
        source = Path(self.paths[index])
        sidecar = self.cache_root / source.parent.name / source.name
        with np.load(sidecar, allow_pickle=False) as d:
            stored = Path(map_path(str(d["source_npz"])))
            if str(d["scenario_id"]) != item["scenario_id"] or stored.resolve() != source.resolve():
                raise ValueError(f"Map cache source identity mismatch: {sidecar}")
            if int(d["cache_version"]) != 2:
                raise ValueError("Unsupported map cache version")
            for key in md.FIELDS:
                item[key] = torch.from_numpy(d[key].copy())
        item["map_adaptation"] = torch.tensor(True)
        return item

    md.MapAdaptationDataset.__getitem__ = __getitem__
    _patched = True
