#!/usr/bin/env python3
"""Stage 1: train the calibrated similarity metric M^A (low-rank bilinear)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

GRIDREL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GRIDREL))
sys.path.insert(0, str(GRIDREL / "alignment"))

# Reuse alignment/train_bilinear_swap.py as the Stage-1 entry.
# Default config: configs/train_ma.yaml (override with --config).
if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", str(GRIDREL / "configs" / "train_ma.yaml")])
    runpy.run_path(str(GRIDREL / "alignment" / "train_bilinear_swap.py"), run_name="__main__")
