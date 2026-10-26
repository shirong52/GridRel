"""Build multi-channel DCSM input X = Concat(M, G, R) per PDF §2.2."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

from src.util.relation_labels import L_REL_CLASSES, NUM_L_REL_CLASSES
from src.util.role_masks import role_masks_for_texts

NUM_G_CHANNELS = 2
NUM_G_REL_CHANNELS = 6
NUM_R_CHANNELS = NUM_L_REL_CLASSES
NUM_INPUT_CHANNELS = 1 + NUM_G_CHANNELS + NUM_R_CHANNELS  # M + G + R = 11
NUM_MRR_CHANNELS = 1 + NUM_G_REL_CHANNELS + NUM_R_CHANNELS  # M + G_rel + R = 15

VALID_CHANNEL_MODES = (
    "m",
    "mg",
    "mr",
    "mgr",
    "mmgr",
    "mgr_a",
    "gridrel",  # paper alias for mgr_a: Concat(M, M^A, G, R)
    "mgr_a_perm",  # same as mgr_a, then fixed permute of all patch slots (G/R move with patches)
    "mgr_ashuf",
    "ma_gr",
    "mrr",
    "ma",  # M + M^A, no G, no R
    "mr_a",  # M + M^A + R, no G
    "mg_a",  # M + M^A + G, no R
    "r",  # R only (shortcut; no similarity channels; CLS off at scorer)
    "gr",  # G+R only (shortcut; no similarity channels; CLS off at scorer)
)
# Modes whose spatial volume contains the absolute row/col maps and relation planes.
_WITH_G = ("mg", "mgr", "mmgr", "mgr_a", "gridrel", "mgr_a_perm", "mgr_ashuf", "ma_gr", "mg_a", "gr")
_WITH_R = ("mr", "mgr", "mmgr", "mgr_a", "gridrel", "mgr_a_perm", "mgr_ashuf", "ma_gr", "mrr", "mr_a", "r", "gr")
# Similarity-free shortcut modes: no M / M^A in the volume.
_SHORTCUT_NO_SIM = frozenset({"r", "gr"})
MA_SHUFFLE_SEED = 15
# Fixed seed for grid_perm_fixed (notebook §3). Same for train and eval.
GRID_PERM_SEED = 2026


def num_input_channels(
    channel_mode: str = "mgr",
    *,
    pad_to_mgr: bool = False,
    pad_to_mrr: bool = False,
) -> int:
    """Return scorer input channel count for ablation / negative-control mode."""
    counts = {
        "m": 1,
        "mg": 1 + NUM_G_CHANNELS,
        "mr": 1 + NUM_R_CHANNELS,
        "mgr": NUM_INPUT_CHANNELS,
        "mmgr": NUM_INPUT_CHANNELS + 1,  # M + M + G + R
        "mgr_a": NUM_INPUT_CHANNELS + 1,  # M + M^A + G + R
        "gridrel": NUM_INPUT_CHANNELS + 1,  # alias of mgr_a
        "mgr_a_perm": NUM_INPUT_CHANNELS + 1,  # same layout; patch slots permuted
        "mgr_ashuf": NUM_INPUT_CHANNELS + 1,  # M + shuffled M^A + G + R
        "ma_gr": NUM_INPUT_CHANNELS,  # M^A + G + R; CLS still reads raw M
        "mrr": NUM_MRR_CHANNELS,
        "ma": 2,  # M + M^A, no G, no R
        "mr_a": 2 + NUM_R_CHANNELS,  # M + M^A + R
        "mg_a": 2 + NUM_G_CHANNELS,  # M + M^A + G
        "r": NUM_R_CHANNELS,  # R only
        "gr": NUM_G_CHANNELS + NUM_R_CHANNELS,  # G + R
    }
    if channel_mode not in counts:
        raise ValueError(
            f"channel_mode must be one of {VALID_CHANNEL_MODES}, got {channel_mode!r}"
        )
    n = counts[channel_mode]
    if pad_to_mrr:
        return max(n, NUM_MRR_CHANNELS)
    if pad_to_mgr:
        return max(n, NUM_INPUT_CHANNELS)
    return n


# ViT-B/16 default. Training and eval set this from the loaded CLIP image grid.
PATCH_GRID = 14


def _patch_row_col_norm(img_seq: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalized row/col coordinate per patch index (CLS at 0.5)."""
    rows = torch.full((img_seq,), 0.5, device=device)
    cols = torch.full((img_seq,), 0.5, device=device)
    grid = PATCH_GRID
    for i in range(1, min(img_seq, grid * grid + 1)):
        p = i - 1
        r, c = p // grid, p % grid
        rows[i] = r / max(grid - 1, 1)
        cols[i] = c / max(grid - 1, 1)
    return rows, cols


def _soft_anchor_from_m(
    m_hw: torch.Tensor,
    col_mask: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Softmax-weighted patch centroid for subject/object token columns."""
    if col_mask.sum() == 0:
        z = torch.tensor(0.5, device=m_hw.device, dtype=m_hw.dtype)
        return z, z
    patch_scores = m_hw[1:, col_mask].mean(dim=-1)
    w = torch.softmax(patch_scores, dim=0)
    r_a = (w * rows[1:]).sum()
    c_a = (w * cols[1:]).sum()
    return r_a, c_a


def _relational_geometric_map_single(
    m_hw: torch.Tensor,
    subj_mask: torch.Tensor,
    obj_mask: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
) -> torch.Tensor:
    """Return G_rel stack (6, H, W) for one image-text pair."""
    h, w = m_hw.shape
    r_s, c_s = _soft_anchor_from_m(m_hw, subj_mask, rows, cols)
    r_o, c_o = _soft_anchor_from_m(m_hw, obj_mask, rows, cols)

    row_grid = rows.unsqueeze(1).expand(h, w)
    col_grid = cols.unsqueeze(1).expand(h, w)
    ch0 = row_grid - r_s
    ch1 = col_grid - c_s
    ch2 = row_grid - r_o
    ch3 = col_grid - c_o
    ch4 = torch.full((h, w), float(r_o - r_s), device=m_hw.device, dtype=m_hw.dtype)
    ch5 = torch.full((h, w), float(c_o - c_s), device=m_hw.device, dtype=m_hw.dtype)
    return torch.stack([ch0, ch1, ch2, ch3, ch4, ch5], dim=0)


def _relational_geometric_maps_batch(
    cossim_mat: torch.Tensor,
    texts: Sequence[str],
) -> torch.Tensor:
    """Build G_rel for all (B1, B2) pairs; returns (B1, B2, 6, H, W)."""
    b1, b2, h, w = cossim_mat.shape
    device = cossim_mat.device
    dtype = cossim_mat.dtype
    rows, cols = _patch_row_col_norm(h, device)
    subj_masks, obj_masks = role_masks_for_texts(texts, text_seq=w, device=device)

    m_patch = cossim_mat[:, :, 1:, :]  # (B1, B2, 196, W)
    sm = subj_masks.unsqueeze(0).unsqueeze(2).to(dtype=dtype)
    om = obj_masks.unsqueeze(0).unsqueeze(2).to(dtype=dtype)
    subj_den = sm.sum(dim=-1).clamp(min=1.0)
    obj_den = om.sum(dim=-1).clamp(min=1.0)
    patch_subj = (m_patch * sm).sum(dim=-1) / subj_den
    patch_obj = (m_patch * om).sum(dim=-1) / obj_den

    w_s = torch.softmax(patch_subj, dim=-1)
    w_o = torch.softmax(patch_obj, dim=-1)
    row_p = rows[1:].view(1, 1, -1)
    col_p = cols[1:].view(1, 1, -1)
    r_s = (w_s * row_p).sum(dim=-1)
    c_s = (w_s * col_p).sum(dim=-1)
    r_o = (w_o * row_p).sum(dim=-1)
    c_o = (w_o * col_p).sum(dim=-1)

    row_grid = rows.view(1, 1, h, 1).expand(b1, b2, h, w)
    col_grid = cols.view(1, 1, h, 1).expand(b1, b2, h, w)
    rs = r_s.view(b1, b2, 1, 1)
    cs = c_s.view(b1, b2, 1, 1)
    ro = r_o.view(b1, b2, 1, 1)
    co = c_o.view(b1, b2, 1, 1)

    ch0 = row_grid - rs
    ch1 = col_grid - cs
    ch2 = row_grid - ro
    ch3 = col_grid - co
    ch4 = (ro - rs).expand(b1, b2, h, w)
    ch5 = (co - cs).expand(b1, b2, h, w)
    return torch.stack([ch0, ch1, ch2, ch3, ch4, ch5], dim=2)


def _geometric_maps(
    img_seq: int,
    text_seq: int,
    device: torch.device,
    *,
    shuffle_g: bool = False,
    shuffle_g_seed: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Patch row/col coordinates broadcast to (H, W)."""
    gh = torch.full((img_seq, text_seq), 0.5, device=device)
    gw = torch.full((img_seq, text_seq), 0.5, device=device)
    grid = PATCH_GRID
    for i in range(1, min(img_seq, grid * grid + 1)):
        p = i - 1
        r, c = p // grid, p % grid
        gh[i, :] = r / max(grid - 1, 1)
        gw[i, :] = c / max(grid - 1, 1)

    if shuffle_g:
        n_patch = max(img_seq - 1, 0)
        if n_patch > 0:
            gen = torch.Generator()
            gen.manual_seed(int(shuffle_g_seed))
            perm = torch.randperm(n_patch, generator=gen)
            gh[1:, :] = gh[1:, :][perm.to(device)]
            gw[1:, :] = gw[1:, :][perm.to(device)]
    return gh, gw


def _relation_map_for_text(
    text: str,
    img_seq: int,
    text_seq: int,
    device: torch.device,
    *,
    random_r: bool = False,
    random_r_seed: int = 15,
) -> torch.Tensor:
    """One-hot over 8 verbal directions, broadcast to (8, H, W)."""
    if random_r:
        hsh = abs(hash(text)) % (2**31)
        gen = torch.Generator()
        gen.manual_seed(int(random_r_seed) + hsh)
        vec = torch.rand(NUM_R_CHANNELS, generator=gen)
        return torch.stack(
            [torch.full((img_seq, text_seq), float(v), device=device) for v in vec],
            dim=0,
        )

    from src.util.relation_labels import _detect_verbal_key

    try:
        key = _detect_verbal_key(text)
    except ValueError:
        return torch.zeros((NUM_R_CHANNELS, img_seq, text_seq), device=device)
    maps = []
    for rel in L_REL_CLASSES:
        val = 1.0 if key == rel else 0.0
        maps.append(torch.full((img_seq, text_seq), val, device=device))
    return torch.stack(maps, dim=0)


def shuffle_patch_rows(cossim: torch.Tensor, seed: int = MA_SHUFFLE_SEED) -> torch.Tensor:
    """Permute the 196 patch rows with one fixed seed. The CLS row stays."""
    n_patch = int(cossim.shape[-2]) - 1
    if n_patch <= 0:
        return cossim
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    perm = torch.randperm(n_patch, generator=gen).to(cossim.device)
    out = cossim.clone()
    out[..., 1:, :] = cossim[..., 1:, :][..., perm, :]
    return out


def _require_aligned(channel_mode: str, aligned: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    if aligned is None:
        raise ValueError(f"channel_mode={channel_mode!r} requires aligned_cossim")
    if aligned.shape != like.shape:
        raise ValueError(
            f"aligned_cossim shape {tuple(aligned.shape)} != M shape {tuple(like.shape)}"
        )
    return aligned


def stack_dcsm_channels(
    cossim_mat: torch.Tensor,
    texts: Sequence[str],
    *,
    channel_mode: str = "mgr",
    shuffle_g: bool = False,
    shuffle_g_seed: int = 15,
    random_r: bool = False,
    random_r_seed: int = 15,
    pad_to_mgr: bool = False,
    pad_to_mrr: bool = False,
    aligned_cossim: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Stack M, G, R into (B1, B2, C, H, W).

    channel_mode: m | mg | mr | mgr | mrr
    cossim_mat: (B1, B2, H, W) standardized cosine map M
    texts: length B2 captions (one per text index)

    Negative-control flags (Stage 6 / Stage 9):
      shuffle_g: permute G_abs patch coordinates (fixed seed)
      random_r: replace R one-hot with deterministic random vectors
      pad_to_mgr: zero-pad channel axis to 11 (matched capacity vs MGR)
      pad_to_mrr: zero-pad channel axis to 15 (matched capacity vs MRR)
    """
    if channel_mode not in VALID_CHANNEL_MODES:
        raise ValueError(
            f"channel_mode must be one of {VALID_CHANNEL_MODES}, got {channel_mode!r}"
        )

    b1, b2, h, w = cossim_mat.shape
    device = cossim_mat.device

    m = cossim_mat.unsqueeze(2)
    if channel_mode in _SHORTCUT_NO_SIM:
        # R_only / G_R_only: no image–text similarity channels (and CLS is off).
        parts = []
    elif channel_mode == "ma_gr":
        aligned = _require_aligned(channel_mode, aligned_cossim, cossim_mat)
        parts = [aligned.unsqueeze(2)]
    else:
        parts = [m]

    if channel_mode == "mmgr":
        parts.append(m)
    elif channel_mode in ("mgr_a", "gridrel", "mgr_a_perm", "mgr_ashuf", "ma", "mr_a", "mg_a"):
        aligned = _require_aligned(channel_mode, aligned_cossim, cossim_mat)
        if channel_mode == "mgr_ashuf":
            aligned = shuffle_patch_rows(aligned, seed=MA_SHUFFLE_SEED)
        parts.append(aligned.unsqueeze(2))

    if channel_mode in _WITH_G:
        gh, gw = _geometric_maps(
            h, w, device, shuffle_g=shuffle_g, shuffle_g_seed=shuffle_g_seed
        )
        g = torch.stack([gh, gw], dim=0).unsqueeze(0).unsqueeze(0)
        g = g.expand(b1, b2, -1, -1, -1)
        parts.append(g)

    if channel_mode == "mrr":
        g_rel = _relational_geometric_maps_batch(cossim_mat, texts)
        parts.append(g_rel)

    if channel_mode in _WITH_R:
        r_list = []
        for j in range(b2):
            txt = texts[j] if j < len(texts) else texts[-1]
            r_list.append(
                _relation_map_for_text(
                    txt,
                    h,
                    w,
                    device,
                    random_r=random_r,
                    random_r_seed=random_r_seed,
                )
            )
        r = torch.stack(r_list, dim=0).unsqueeze(0).expand(b1, -1, -1, -1, -1)
        parts.append(r)

    out = torch.cat(parts, dim=2)

    target = None
    if pad_to_mrr:
        target = NUM_MRR_CHANNELS
    elif pad_to_mgr:
        target = NUM_INPUT_CHANNELS
    if target is not None and out.size(2) < target:
        pad_n = target - out.size(2)
        zeros = torch.zeros(b1, b2, pad_n, h, w, device=device, dtype=out.dtype)
        out = torch.cat([out, zeros], dim=2)

    # grid_perm_fixed: one fixed permutation of the 196 patch slots applied to
    # every channel (M, M^A, G, R). CLS row stays put; neighbourhoods break.
    if channel_mode == "mgr_a_perm":
        out = shuffle_patch_rows(out, seed=GRID_PERM_SEED)
    return out
