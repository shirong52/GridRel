"""DCSM layout: patch reshape + CLS branch fused at train/eval. See plan/00_simplified_m_only_ablation.md."""

from __future__ import annotations

import torch

PATCH_GRID = 14
NUM_PATCHES = PATCH_GRID * PATCH_GRID  # 196
IMG_SEQ = NUM_PATCHES + 1              # 197 = CLS + patches
TEXT_SEQ = 30


def set_patch_grid(grid: int) -> int:
    """Point reshape/assert helpers at a CLIP grid. Default stays 14 (ViT-B/16)."""
    global PATCH_GRID, NUM_PATCHES, IMG_SEQ
    grid = int(grid)
    if grid < 1:
        raise ValueError(f"patch grid must be positive, got {grid}")
    PATCH_GRID = grid
    NUM_PATCHES = grid * grid
    IMG_SEQ = NUM_PATCHES + 1
    return grid


def split_cls_and_patches(m_hw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., 197, 30) -> m_cls (...,1,30), m_patch (...,196,30)."""
    assert m_hw.shape[-2] == IMG_SEQ
    assert m_hw.shape[-1] == TEXT_SEQ
    return m_hw[..., 0:1, :], m_hw[..., 1:, :]


def patches_to_token_spatial(m_patch: torch.Tensor) -> torch.Tensor:
    """(..., 196, 30) -> (..., 30, 14, 14). Training input (no CLS row)."""
    m_th = m_patch.transpose(-2, -1)  # (..., 30, 196)
    return m_th.reshape(*m_th.shape[:-1], PATCH_GRID, PATCH_GRID)


def dcsm_to_token_spatial(m_hw: torch.Tensor) -> torch.Tensor:
    """(..., 197, 30) -> (..., 30, 14, 14). Drops CLS row."""
    m_th = m_hw.transpose(-2, -1)[..., 1:]  # (..., 30, 196)
    return m_th.reshape(*m_th.shape[:-1], PATCH_GRID, PATCH_GRID)


def dcsm_to_train_spatial(m_hw: torch.Tensor) -> torch.Tensor:
    """
    Training: drop CLS row, reshape patches only.

    (..., 197, 30) -> (..., 30, 14, 14)
    EOS/PAD stay as token slices; no special handling at train time.
    """
    return dcsm_to_token_spatial(m_hw)


def dcsm_to_train_3d(m_hw: torch.Tensor) -> torch.Tensor:
    """Training 3D: (..., 1, 30, 14, 14) — single channel, patch-only."""
    spatial = dcsm_to_train_spatial(m_hw)
    return spatial.unsqueeze(-4)


def dcsm_stacked_to_train_3d_stacked(stacked: torch.Tensor) -> torch.Tensor:
    """
    Scheme B: M/G patch volumes + R planes broadcast over H/W.

    Layout from stack_dcsm_channels: m=1, mg=3, mr=9, mgr=11.
    """
    c = stacked.shape[-3]
    m_vol = dcsm_to_train_3d(stacked[..., 0, :, :])
    if c <= 1:
        return m_vol

    if c == 3:
        num_g, num_r = 2, 0
    elif c == 8:
        # R only (no M): broadcast R planes over the patch grid.
        r_token = stacked[..., 0, :]
        return r_token.unsqueeze(-1).unsqueeze(-1).expand(
            *r_token.shape, PATCH_GRID, PATCH_GRID
        )
    elif c == 9:
        num_g, num_r = 0, 8
    elif c == 10:
        # G(2) + R(8), no similarity channel (shortcut G_R_only).
        vols = [
            dcsm_to_train_3d(stacked[..., 0, :, :]),
            dcsm_to_train_3d(stacked[..., 1, :, :]),
        ]
        r_maps = stacked[..., 2:, :, :]
        r_token = r_maps[..., 0, :]
        r_vol = r_token.unsqueeze(-1).unsqueeze(-1).expand(
            *r_token.shape, PATCH_GRID, PATCH_GRID
        )
        vols.append(r_vol)
        return torch.cat(vols, dim=-4)
    elif c == 11:
        num_g, num_r = 2, 8
    elif c == 12:
        # M, M^A, Gx2, Rx8. M^A uses the same patch reshape as M.
        num_g, num_r = 2, 8
        vols = [m_vol, dcsm_to_train_3d(stacked[..., 1, :, :])]
        idx = 2
        for _ in range(num_g):
            vols.append(dcsm_to_train_3d(stacked[..., idx, :, :]))
            idx += 1
        r_maps = stacked[..., idx:, :, :]
        r_token = r_maps[..., 0, :]
        r_vol = r_token.unsqueeze(-1).unsqueeze(-1).expand(
            *r_token.shape, PATCH_GRID, PATCH_GRID
        )
        vols.append(r_vol)
        return torch.cat(vols, dim=-4)
    else:
        num_g, num_r = 0, c - 1

    vols = [m_vol]
    idx = 1
    for _ in range(num_g):
        vols.append(dcsm_to_train_3d(stacked[..., idx, :, :]))
        idx += 1
    if num_r > 0:
        r_maps = stacked[..., idx:, :, :]
        r_token = r_maps[..., 0, :]
        r_vol = r_token.unsqueeze(-1).unsqueeze(-1).expand(
            *r_token.shape, PATCH_GRID, PATCH_GRID
        )
        vols.append(r_vol)
    return torch.cat(vols, dim=-4)


# backward-compatible alias
dcsm_stacked_to_train_3d_mr = dcsm_stacked_to_train_3d_stacked


def score_cls_eos(m_hw: torch.Tensor, *, token_mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Eval-only S1 from full DCSM (before reshape).

    Args:
        m_hw: (..., 197, 30)
        token_mask: (..., 30) bool, True = valid token (excl. PAD). If None, use all 30.
    Returns:
        (...,) scalar S1 per pair
    """
    m_cls = m_hw[..., 0, :]  # CLS row
    if token_mask is not None:
        mask = token_mask.to(dtype=m_cls.dtype)
        denom = mask.sum(dim=-1).clamp(min=1.0)
        s1 = (m_cls * mask).sum(dim=-1) / denom
    else:
        s1 = m_cls.mean(dim=-1)
    return s1


def fuse_scores(
    s_cnn: torch.Tensor,
    s_cls_eos: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Eval: S = S1(cls,eos) + alpha * S2(cnn). Default alpha=1."""
    return s_cls_eos + alpha * s_cnn
