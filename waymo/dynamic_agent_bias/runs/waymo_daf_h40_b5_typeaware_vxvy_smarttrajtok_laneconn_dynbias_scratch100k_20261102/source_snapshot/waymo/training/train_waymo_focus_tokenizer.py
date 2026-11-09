"""Compatibility wrapper for the focus-only tokenizer trainer."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "tokenizer" / "train_waymo_focus_tokenizer.py"
    runpy.run_path(str(target), run_name="__main__")
