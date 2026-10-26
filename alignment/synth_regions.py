"""Approximate two-object regions on synthetic images.

Regions come from a blur residual, not from CLIP. Images where the two
blobs overlap or one blob is missing are rejected.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage


def subject_component_index(relation: str, centroids: list[tuple[float, float]]) -> int:
    """Index of the subject blob. Centroids are (row, col)."""
    if relation == "leftof":
        return int(np.argmin([c[1] for c in centroids]))
    if relation in ("above", "on", "behind"):
        return int(np.argmin([c[0] for c in centroids]))
    raise ValueError(f"unsupported relation {relation!r}")


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def blobs_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    """True when either centroid sits inside the other blob's box."""
    boxes = [_bbox(mask_a), _bbox(mask_b)]
    masks = [mask_a, mask_b]
    for i in (0, 1):
        ys, xs = np.nonzero(masks[i])
        cy, cx = float(ys.mean()), float(xs.mean())
        y0, x0, y1, x1 = boxes[1 - i]
        if y0 <= cy < y1 and x0 <= cx < x1:
            return True
    return False


def two_object_masks(
    image: np.ndarray,
    *,
    sigma: float = 25.0,
    threshold: float = 18.0,
    min_area_frac: float = 0.004,
    min_separation_frac: float = 0.12,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return two foreground masks, or None when the split is not usable."""
    blur = ndimage.gaussian_filter(image.astype(np.float32), sigma=(sigma, sigma, 0))
    residual = np.abs(image.astype(np.float32) - blur).mean(axis=2)
    foreground = residual > threshold
    foreground = ndimage.binary_opening(foreground, iterations=1)
    foreground = ndimage.binary_closing(foreground, iterations=3)
    labels, n_labels = ndimage.label(foreground)
    if n_labels < 2:
        return None
    sizes = np.array(ndimage.sum(foreground, labels, index=range(1, n_labels + 1)))
    order = np.argsort(sizes)[::-1]
    min_area = min_area_frac * foreground.size
    chosen = [int(i) for i in order if sizes[i] >= min_area][:2]
    if len(chosen) < 2:
        return None
    masks = [(labels == (index + 1)) for index in chosen]
    height, width = foreground.shape
    centroids = []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        centroids.append((float(ys.mean()), float(xs.mean())))
    distance = float(np.hypot(centroids[0][0] - centroids[1][0], centroids[0][1] - centroids[1][1]))
    if distance < min_separation_frac * min(height, width):
        return None
    if blobs_overlap(masks[0], masks[1]):
        return None
    return masks[0], masks[1]


def resize_center_crop(mask: np.ndarray, canvas: int = 224) -> np.ndarray:
    """Match CLIP shortest-edge resize and center crop. mask is HxW bool."""
    height, width = mask.shape
    if width <= height:
        new_w, new_h = canvas, int(canvas * height / width)
    else:
        new_h, new_w = canvas, int(canvas * width / height)
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize((new_w, new_h), Image.NEAREST)
    left = max((new_w - canvas) // 2, 0)
    top = max((new_h - canvas) // 2, 0)
    cropped = resized.crop((left, top, left + canvas, top + canvas))
    arr = np.asarray(cropped)
    if arr.shape[0] != canvas or arr.shape[1] != canvas:
        canvas_arr = np.zeros((canvas, canvas), dtype=np.uint8)
        canvas_arr[: arr.shape[0], : arr.shape[1]] = arr
        arr = canvas_arr
    return arr > 127


def patches_from_canvas(mask: np.ndarray, grid: int = 14, patch: int = 16, min_frac: float = 0.3) -> np.ndarray:
    """True when at least min_frac of a ViT patch lies inside the mask."""
    out = np.zeros(grid * grid, dtype=bool)
    for i in range(grid):
        for j in range(grid):
            block = mask[i * patch : (i + 1) * patch, j * patch : (j + 1) * patch]
            out[i * grid + j] = float(block.mean()) >= min_frac
    return out
