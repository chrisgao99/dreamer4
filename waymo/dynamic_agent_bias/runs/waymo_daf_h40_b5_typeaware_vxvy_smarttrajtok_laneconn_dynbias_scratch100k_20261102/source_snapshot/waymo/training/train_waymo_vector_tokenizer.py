"""Compatibility wrapper for the tokenizer trainer.

The implementation lives in ``waymo/training/tokenizer`` so tokenizer and
world-model training stay separated.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "tokenizer" / "train_waymo_vector_tokenizer.py"
    runpy.run_path(str(target), run_name="__main__")
