"""Topology scorers: CLS branch + spatial reshape branch, S = S1 + alpha * S2."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from lib import topology
from lib.topology import dcsm_stacked_to_train_3d_stacked, dcsm_to_train_3d, dcsm_to_train_spatial
from src.util.dcsm_channels import num_input_channels, stack_dcsm_channels

TEXT_SEQ = 30

TOPOLOGY_SCORER_TYPES = frozenset(
    {
        "reshape2d",
        "reshape2d_m",
        "conv3d",
        "conv3d_m",
        "reshape3d",
        "reshape3d_m",
        # Flatten→MLP spatial branch (CNN/Conv3d trunk kept; this is the ablation path)
        "reshape3d_mlp",
        "conv3d_mlp",
        "reshape3d_mlp_m",
        "conv3d_mlp_m",
        # Param-matched flatten MLP (≈ Conv3d / M3D trunk size; P0 fair control)
        "reshape3d_mlp_matched",
        "conv3d_mlp_matched",
        # Point-4 staged: T-axis kernel/stride = 1 (Stage-1)
        "conv3d_kt1",
        "reshape3d_kt1",
        # Point-4 Stage-2: per-token 2D encode + joint FC (no T-local conv)
        "spatial_factor_joint",
        "spatial_factor_joint_m",
    }
)

# Spatial branch uses Conv3d trunk (default kT=3)
_CONV3D_TYPES = frozenset({"conv3d", "conv3d_m", "reshape3d", "reshape3d_m"})
# Same trunk but token-axis kernel/stride forced to 1
_CONV3D_KT1_TYPES = frozenset({"conv3d_kt1", "reshape3d_kt1"})
# Spatial branch skips Conv3d: flatten volume → MLP
_MLP3D_TYPES = frozenset(
    {"reshape3d_mlp", "conv3d_mlp", "reshape3d_mlp_m", "conv3d_mlp_m"}
)
# Same input volume, 2-layer MLP width chosen to ≈ match M3DSpatialScorer params
_MLP3D_MATCHED_TYPES = frozenset({"reshape3d_mlp_matched", "conv3d_mlp_matched"})
# Per-token 2D then joint score head (no T-axis convolution)
_SPATIAL_SET_TYPES = frozenset({"spatial_factor_joint", "spatial_factor_joint_m"})
# 2D token-spatial reshape (Conv2d)
_RESHAPE2D_TYPES = frozenset({"reshape2d", "reshape2d_m"})


class CLSBranch(nn.Module):
    """Map CLS row of DCSM (..., 30) to scalar S1."""

    def __init__(
        self,
        text_seq: int = TEXT_SEQ,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        mid = max(hidden_dim, 16)
        self.net = nn.Sequential(
            nn.Linear(text_seq, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 1),
        )

    def forward(self, cossim_hw: torch.Tensor) -> torch.Tensor:
        m_cls = cossim_hw[..., 0, :]
        return self.net(m_cls)


class Reshape2DSpatialScorer(nn.Module):
    """
    Pure 2D spatial branch: (B1, B2, 30, 14, 14) -> (B1, B2, 1).

    Token slices are input channels; Conv2d kernels slide on the 14x14 patch grid.
    """

    def __init__(
        self,
        text_seq: int = TEXT_SEQ,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_seq = text_seq
        c_mid = hidden_dim // 32
        self.conv1 = nn.Conv2d(text_seq, c_mid, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(dropout)
        self.dropout2 = nn.Dropout2d(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        grid = topology.PATCH_GRID
        flat_dim = c_mid * grid * grid
        self.fc1 = nn.Linear(flat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 6:
            x = x[:, :, 0]
        b1, b2 = x.shape[:2]
        flat = x.reshape(b1 * b2, *x.shape[2:])
        flat = self.dropout1(F.relu(self.conv1(flat)))
        flat = self.dropout2(F.relu(self.conv2(flat)))
        flat = flat.reshape(b1 * b2, -1)
        flat = self.dropout3(F.relu(self.fc1(flat)))
        flat = self.dropout4(self.fc2(flat))
        return flat.view(b1, b2, 1)


class FlattenMLPSpatialScorer(nn.Module):
    """
    Ablation spatial branch: same 3D volume layout as M3DSpatialScorer, but no Conv3d.

    (B1, B2, C, T, H, W) -> flatten -> MLP -> (B1, B2, 1).
    Keeps CLS+alpha fusion and channel stacking; only removes convolutional inductive bias.
    """

    def __init__(
        self,
        text_seq: int = TEXT_SEQ,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.text_seq = text_seq
        self.in_channels = in_channels
        grid = topology.PATCH_GRID
        flat_dim = in_channels * text_seq * grid * grid
        mid = max(hidden_dim, 64)
        self.net = nn.Sequential(
            nn.Linear(flat_dim, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, mid // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 6:
            b1, b2 = x.shape[:2]
            vol = x.reshape(b1 * b2, -1)
        elif x.dim() == 5:
            b1, b2 = 1, 1
            vol = x.reshape(1, -1)
        else:
            raise ValueError(
                f"FlattenMLPSpatialScorer expects 5D or 6D input, got {x.dim()}D"
            )
        return self.net(vol).view(b1, b2, 1)


def matched_mlp_width(flat_dim: int, target_spatial_params: int) -> int:
    """Pick 2-layer MLP width minimizing |params − target| (Linear flat→w→1)."""
    if flat_dim < 1 or target_spatial_params < 1:
        raise ValueError("flat_dim and target_spatial_params must be positive")
    best_w, best_err = 1, None
    w_max = max(1, min(target_spatial_params // max(flat_dim, 1) + 8, 2048))
    for w in range(1, w_max + 1):
        n = w * (flat_dim + 2) + 1
        err = abs(n - target_spatial_params)
        if best_err is None or err < best_err:
            best_w, best_err = w, err
    return int(best_w)


class FlattenMLPParamMatchedSpatialScorer(nn.Module):
    """
    Fair flat control vs M3DSpatialScorer: same multi-channel volume, no Conv3d.

    Keep the full token axis T, adaptive-avg-pool the 14×14 patch grid to 2×2
    (destroys fine neighbourhood / local conv structure), flatten (C,T,2,2),
    then a 2-layer MLP with width matched so spatial-branch params ≈ Conv3d
    trunk (gap <15%; for mgr_a ≈ width 173).

    Failed alternatives (do not revive without a new design note):
    - Full flatten 70560→width4: param-matched but untrainable bottleneck.
    - Mean-pool T first, then MLP on (C,H,W): stalls ~0.45 train acc.
    - Per-token MLP then mean-T: still weak (no cross-token mixing in the MLP).
    """

    # Coarse spatial size after pooling (H=W). Kept as class attr for tests.
    POOL_HW = 2

    def __init__(
        self,
        text_seq: int = TEXT_SEQ,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.text_seq = text_seq
        self.in_channels = in_channels
        s = self.POOL_HW
        flat_dim = in_channels * text_seq * s * s
        ref = M3DSpatialScorer(
            text_seq=text_seq,
            hidden_dim=hidden_dim,
            dropout=dropout,
            in_channels=in_channels,
            token_kernel_t=3,
        )
        target = int(sum(p.numel() for p in ref.parameters()))
        del ref
        width = matched_mlp_width(flat_dim, target)
        self.mlp_width = width
        self.flat_dim = flat_dim
        self.target_spatial_params = target
        self.net = nn.Sequential(
            nn.Linear(flat_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B1,B2,C,T,H,W) or (C,T,H,W)
        if x.dim() == 6:
            b1, b2 = x.shape[:2]
            vol = x.reshape(b1 * b2, *x.shape[2:])
        elif x.dim() == 5:
            b1, b2 = 1, 1
            vol = x.unsqueeze(0)
        else:
            raise ValueError(
                f"FlattenMLPParamMatchedSpatialScorer expects 5D or 6D, got {x.dim()}D"
            )
        n, c, t, h, w = vol.shape
        # Pool only H,W — keep all tokens so the MLP can mix them jointly.
        planes = vol.reshape(n, c * t, h, w)
        pooled = F.adaptive_avg_pool2d(planes, (self.POOL_HW, self.POOL_HW))
        flat = pooled.reshape(n, -1)
        return self.net(flat).view(b1, b2, 1)


class M3DSpatialScorer(nn.Module):
    """
    Pure 3D spatial branch: (B1, B2, C, T, H, W) -> (B1, B2, 1).

    Conv3d trunk with input residual, stride downsampling on T/H/W, then flatten+FC.
    No global mean pool — analogous to the 2D flatten head.

    token_kernel_t=1 (Stage-1): kernels/strides on T are 1, so tokens are not
    locally mixed by convolution; spatial (H,W) behavior stays the same.
    """

    def __init__(
        self,
        text_seq: int = TEXT_SEQ,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        in_channels: int = 1,
        token_kernel_t: int = 3,
    ) -> None:
        super().__init__()
        if token_kernel_t not in (1, 3):
            raise ValueError(f"token_kernel_t must be 1 or 3, got {token_kernel_t}")
        self.text_seq = text_seq
        self.in_channels = in_channels
        self.token_kernel_t = token_kernel_t
        c1 = max(hidden_dim // 4, 24)
        c2 = max(hidden_dim // 2, 32)
        c_mid = max(hidden_dim // 32, 4)

        kt = token_kernel_t
        pad_t = kt // 2  # 1 for kt=3, 0 for kt=1
        k_full = (kt, 3, 3)
        pad_full = (pad_t, 1, 1)
        # Downsample T only when kt=3 (matches legacy abl3); kt=1 keeps T intact.
        stride_down = (2, 2, 2) if kt == 3 else (1, 2, 2)

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, c1, kernel_size=k_full, padding=pad_full),
            nn.GELU(),
            nn.Dropout3d(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(c1, c2, kernel_size=k_full, padding=pad_full),
            nn.GELU(),
            nn.Dropout3d(dropout),
        )
        self.res_proj = nn.Conv3d(in_channels, c2, kernel_size=1)
        self.conv3 = nn.Sequential(
            nn.Conv3d(c2, c2, kernel_size=k_full, stride=stride_down, padding=pad_full),
            nn.GELU(),
            nn.Dropout3d(dropout),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(c2, c_mid, kernel_size=k_full, stride=stride_down, padding=pad_full),
            nn.GELU(),
        )
        with torch.no_grad():
            grid = topology.PATCH_GRID
            dummy = torch.zeros(1, in_channels, text_seq, grid, grid)
            flat_dim = self._encode_volume(dummy).shape[1]
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(flat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def _encode_volume(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.conv2(h)
        h = h + self.res_proj(x)
        h = self.conv3(h)
        h = self.conv4(h)
        return h.flatten(1)

    def _encode_volume_fit(self, vol: torch.Tensor) -> torch.Tensor:
        """Same conv as one full pass. Large 24x24 batches are split so the
        64-channel map (about 4GB at 1024 pairs) is not allocated at once.
        Checkpoint keeps the chunked backward identical, including dropout.
        """
        n = vol.shape[0]
        wide = vol.shape[-1] > 14 or vol.shape[-2] > 14
        chunk = 128
        if n <= chunk or not wide:
            return self._encode_volume(vol)
        parts = []
        for start in range(0, n, chunk):
            piece = vol[start : start + chunk]
            if self.training:
                parts.append(
                    checkpoint(
                        self._encode_volume,
                        piece,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                )
            else:
                parts.append(self._encode_volume(piece))
        return torch.cat(parts, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 6:
            b1, b2 = x.shape[:2]
            vol = x.reshape(b1 * b2, *x.shape[2:])
        elif x.dim() == 5:
            b1, b2 = 1, 1
            vol = x
        else:
            raise ValueError(f"M3DSpatialScorer expects 5D or 6D input, got {x.dim()}D")
        feat = self._encode_volume_fit(vol)
        feat = self.dropout(F.gelu(self.fc1(feat)))
        feat = self.fc2(feat)
        return feat.view(b1, b2, 1)


class SpatialFactorJointScorer(nn.Module):
    """
    Stage-2 spatial branch (trainable): per-token shared 2D encoder + joint FC.

    Failed set-only aggregators (single-query / mean / LSE) score or pool tokens
    independently and cannot model cross-token interactions. Abl3 / Stage-1 train
    well because the final Linear mixes all tokens' spatial features jointly.

    This module keeps that joint head, but replaces T-axis Conv3d with an explicit
    per-token Conv2d tower (weights shared across T). No local convolution along T.

    Tradeoff: NOT permutation-invariant (same as kt1 flatten). Stage-2 claim becomes
    "factorized spatial encode, then joint score" — not "pure set pool".

    Input: (B1, B2, C, T, 14, 14) → (B1, B2, 1)
    """

    def __init__(
        self,
        text_seq: int = TEXT_SEQ,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.text_seq = text_seq
        self.in_channels = in_channels
        # End width matches abl3 c_mid so flat_dim ≈ kt1 (T * c_mid * 4 * 4).
        c1, c2, c_mid = 32, 64, 4

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),  # 14 → 7
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(c2, c_mid, kernel_size=3, stride=2, padding=1),  # 7 → 4
            nn.GELU(),
        )
        self.res_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.res_proj = nn.Conv2d(in_channels, c_mid, kernel_size=1)
        flat_dim = text_seq * c_mid * 4 * 4
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(flat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def encode_volume(self, vol: torch.Tensor) -> torch.Tensor:
        """vol: (B, C, T, H, W) → flat: (B, T*c_mid*4*4)."""
        b, c, t, h, w = vol.shape
        x = vol.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        h_enc = self.enc(x) + self.res_proj(self.res_pool(x))
        # (B, T, c_mid, 4, 4) then flatten token+space together for joint FC
        h_enc = h_enc.view(b, t, *h_enc.shape[1:])
        return h_enc.reshape(b, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 6:
            b1, b2 = x.shape[:2]
            vol = x.reshape(b1 * b2, *x.shape[2:])
        elif x.dim() == 5:
            b1, b2 = 1, 1
            vol = x
        else:
            raise ValueError(
                f"SpatialFactorJointScorer expects 5D or 6D input, got {x.dim()}D"
            )
        feat = self.encode_volume(vol)
        feat = self.dropout(F.gelu(self.fc1(feat)))
        return self.fc2(feat).view(b1, b2, 1)


# Shortcut modes: no similarity channels and no CLS branch (spatial score only).
_CLS_OFF_MODES = frozenset({"r", "gr"})


class TopologyScorer(nn.Module):
    """
    Full topology scorer: cossim (B1, B2, 197, 30) -> S = S1(CLS) + alpha * S2(spatial).

    For channel_mode in {r, gr}: similarity channels are absent and CLS is off;
    the score is the spatial branch alone.
    """

    def __init__(
        self,
        scorer_type: str,
        *,
        text_seq: int = TEXT_SEQ,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        learnable_alpha: bool = True,
        channel_mode: str = "m",
    ) -> None:
        super().__init__()
        self.scorer_type = scorer_type
        self.channel_mode = channel_mode
        self.cls_off = channel_mode in _CLS_OFF_MODES
        if self.cls_off:
            self.cls_branch = None
            self.register_buffer("alpha", torch.tensor(0.0), persistent=False)
        else:
            self.cls_branch = CLSBranch(
                text_seq=text_seq,
                hidden_dim=max(hidden_dim // 4, 32),
                dropout=dropout,
            )
        if scorer_type in _RESHAPE2D_TYPES:
            self.spatial_branch = Reshape2DSpatialScorer(
                text_seq=text_seq, hidden_dim=hidden_dim, dropout=dropout
            )
        elif scorer_type in _CONV3D_TYPES:
            in_ch = num_input_channels(channel_mode)
            self.spatial_branch = M3DSpatialScorer(
                text_seq=text_seq,
                hidden_dim=hidden_dim,
                dropout=dropout,
                in_channels=in_ch,
                token_kernel_t=3,
            )
        elif scorer_type in _CONV3D_KT1_TYPES:
            in_ch = num_input_channels(channel_mode)
            self.spatial_branch = M3DSpatialScorer(
                text_seq=text_seq,
                hidden_dim=hidden_dim,
                dropout=dropout,
                in_channels=in_ch,
                token_kernel_t=1,
            )
        elif scorer_type in _SPATIAL_SET_TYPES:
            in_ch = num_input_channels(channel_mode)
            self.spatial_branch = SpatialFactorJointScorer(
                text_seq=text_seq,
                hidden_dim=hidden_dim,
                dropout=dropout,
                in_channels=in_ch,
            )
        elif scorer_type in _MLP3D_TYPES:
            in_ch = num_input_channels(channel_mode)
            self.spatial_branch = FlattenMLPSpatialScorer(
                text_seq=text_seq,
                hidden_dim=hidden_dim,
                dropout=dropout,
                in_channels=in_ch,
            )
        elif scorer_type in _MLP3D_MATCHED_TYPES:
            in_ch = num_input_channels(channel_mode)
            self.spatial_branch = FlattenMLPParamMatchedSpatialScorer(
                text_seq=text_seq,
                hidden_dim=hidden_dim,
                dropout=dropout,
                in_channels=in_ch,
            )
        else:
            raise ValueError(f"unknown topology scorer_type: {scorer_type}")

        if not self.cls_off:
            if learnable_alpha:
                self.alpha = nn.Parameter(torch.tensor(1.0))
            else:
                self.register_buffer("alpha", torch.tensor(1.0), persistent=True)

    def forward(
        self,
        cossim_hw: torch.Tensor,
        texts: Sequence[str] | None = None,
        aligned_cossim: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m_hw = cossim_hw
        if self.channel_mode != "m":
            if texts is None:
                raise ValueError(f"texts required for channel_mode={self.channel_mode!r}")
            stacked = stack_dcsm_channels(
                cossim_hw,
                texts,
                channel_mode=self.channel_mode,
                aligned_cossim=aligned_cossim,
            )
            # ma_gr puts M^A in the spatial volume only. CLS still reads raw M.
            if self.channel_mode not in ("ma_gr", "r", "gr"):
                m_hw = stacked[..., 0, :, :]
            if self.scorer_type in _RESHAPE2D_TYPES:
                raise ValueError("multi-channel reshape2d not supported")
            spatial = dcsm_stacked_to_train_3d_stacked(stacked)
        elif self.scorer_type in _RESHAPE2D_TYPES:
            spatial = dcsm_to_train_spatial(cossim_hw)
        else:
            # conv3d / reshape3d / *mlp: same 3D volume layout
            spatial = dcsm_to_train_3d(cossim_hw)
        s2 = self.spatial_branch(spatial)
        if self.cls_off:
            return s2
        s1 = self.cls_branch(m_hw)
        return s1 + self.alpha * s2


def is_topology_scorer_type(scorer_type: str) -> bool:
    return scorer_type in TOPOLOGY_SCORER_TYPES


def build_topology_scorer(
    scorer_type: str,
    *,
    text_seq: int = TEXT_SEQ,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    learnable_alpha: bool = True,
    channel_mode: str = "m",
) -> nn.Module:
    if not is_topology_scorer_type(scorer_type):
        raise ValueError(f"unknown topology scorer_type: {scorer_type}")
    return TopologyScorer(
        scorer_type,
        text_seq=text_seq,
        hidden_dim=hidden_dim,
        dropout=dropout,
        learnable_alpha=learnable_alpha,
        channel_mode=channel_mode,
    )


def load_topology_scorer_state(task_net: nn.Module, state: dict) -> None:
    task_net.load_state_dict(state, strict=True)
