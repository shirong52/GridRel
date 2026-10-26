from __future__ import annotations

import json
import random
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)

def git_hash(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

def resized_hw(width: int, height: int, shortest: int = 224) -> tuple[int, int]:
    """Match transformers get_resize_output_image_size, shortest-edge mode.

    Returns (new_height, new_width).
    """
    if width <= height:
        new_long = int(shortest * height / width)
        return new_long, shortest
    new_long = int(shortest * width / height)
    return shortest, new_long

def canvas_box(
    xywh: list[float],
    width: int,
    height: int,
    canvas: int = 224,
    shortest: int = 224,
) -> tuple[float, float, float, float] | None:
    """Map a COCO xywh box into the CLIP 224 canvas. None if the crop misses it."""
    x, y, w, h = (float(v) for v in xywh)
    new_h, new_w = resized_hw(width, height, shortest)
    sx = new_w / float(width)
    sy = new_h / float(height)
    left = (new_w - canvas) // 2
    top = (new_h - canvas) // 2
    x0 = x * sx - left
    y0 = y * sy - top
    x1 = (x + w) * sx - left
    y1 = (y + h) * sy - top
    x0 = max(0.0, x0)
    y0 = max(0.0, y0)
    x1 = min(float(canvas), x1)
    y1 = min(float(canvas), y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1

def patch_inside_mask(
    box: tuple[float, float, float, float],
    grid: int = 14,
    patch: int = 16,
) -> np.ndarray:
    """True where the patch center lies inside the canvas box. Row-major 14x14."""
    x0, y0, x1, y1 = box
    mask = np.zeros(grid * grid, dtype=bool)
    for i in range(grid):
        cy = patch * (i + 0.5)
        if not (y0 <= cy < y1):
            continue
        for j in range(grid):
            cx = patch * (j + 0.5)
            if x0 <= cx < x1:
                mask[i * grid + j] = True
    return mask

def find_phrase(caption: str, name: str) -> tuple[int, int, str] | None:
    """Character span of `name` in `caption`. Exact word, else a single trailing s."""
    exact = list(re.finditer(rf"\b{re.escape(name)}\b", caption, flags=re.IGNORECASE))
    if len(exact) == 1:
        return exact[0].start(), exact[0].end(), "exact"
    if len(exact) > 1:
        return None
    if not name.endswith("s"):
        plural = list(
            re.finditer(rf"\b{re.escape(name)}s\b", caption, flags=re.IGNORECASE)
        )
        if len(plural) == 1:
            return plural[0].start(), plural[0].end(), "plural_s"
    return None

def token_indices_for_span(
    offsets: list[tuple[int, int]],
    special_mask: list[bool],
    start: int,
    end: int,
) -> list[int]:
    hits = []
    for i, ((a, b), special) in enumerate(zip(offsets, special_mask)):
        if special or b <= a:
            continue
        if a < end and b > start:
            hits.append(i)
    return hits