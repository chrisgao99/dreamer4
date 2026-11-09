"""Compatibility wrapper for the world-model trainer.

The implementation lives in ``waymo/training/world_model`` so tokenizer and
world-model training stay separated.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "world_model" / "train_waymo_world_model.py"
    runpy.run_path(str(target), run_name="__main__")
