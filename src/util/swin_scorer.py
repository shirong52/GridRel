"""Swin-style lightweight Transformer scorer for DCSM dense relation maps."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_to_window(x: torch.Tensor, window_size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad (B, C, H, W) so H,W are multiples of window_size."""
    _, _, h, w = x.shape
    wh, ww = window_size
    pad_h = (wh - h % wh) % wh
    pad_w = (ww - w % ww) % ww
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, (pad_h, pad_w)


def _window_partition(x: torch.Tensor, window_size: Tuple[int, int]) -> torch.Tensor:
    """(B, C, H, W) -> (B*nW, ws_h*ws_w, C)."""
    b, c, h, w = x.shape
    wh, ww = window_size
    x = x.view(b, c, h // wh, wh, w // ww, ww)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(-1, wh * ww, c)


def _window_reverse(
    windows: torch.Tensor,
    window_size: Tuple[int, int],
    h: int,
    w: int,
    batch_size: int,
) -> torch.Tensor:
    """(B*nW, ws_h*ws_w, C) -> (B, C, H, W)."""
    wh, ww = window_size
    nh, nw = h // wh, w // ww
    x = windows.view(batch_size, nh, nw, wh, ww, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(batch_size, -1, h, w)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        ws = window_size[0] * window_size[1]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = torch.flatten(coords, 1)
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size[0] - 1
        relative[:, :, 1] += window_size[1] - 1
        relative[:, :, 0] *= 2 * window_size[1] - 1
        rel_idx = relative.sum(-1)
        self.register_buffer("relative_position_index", rel_idx, persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(self.window_size[0] * self.window_size[1], -1, self.num_heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj(out)


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int],
        shift_size: Tuple[int, int] = (0, 0),
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        shortcut = x.permute(0, 2, 3, 1).contiguous()

        if any(s > 0 for s in self.shift_size):
            x_map = torch.roll(
                shortcut,
                shifts=(-self.shift_size[0], -self.shift_size[1]),
                dims=(1, 2),
            )
        else:
            x_map = shortcut

        x_bchw = x_map.permute(0, 3, 1, 2).contiguous()
        attn_windows = _window_partition(x_bchw, self.window_size)
        attn_windows = self.attn(self.norm1(attn_windows))
        attn_out = _window_reverse(attn_windows, self.window_size, h, w, b)

        if any(s > 0 for s in self.shift_size):
            attn_out = torch.roll(
                attn_out.permute(0, 2, 3, 1),
                shifts=(self.shift_size[0], self.shift_size[1]),
                dims=(1, 2),
            )
        else:
            attn_out = attn_out.permute(0, 2, 3, 1)

        x_map = shortcut + attn_out
        x_map = x_map + self.mlp(self.norm2(x_map))
        return x_map.permute(0, 3, 1, 2).contiguous()


class CrossModalSwinNetwork(nn.Module):
    """Swin scorer on Concat(M,G,R): flatten + MLP head (~3M params)."""

    def __init__(
        self,
        img_seq: int,
        text_seq: int,
        hidden_dim: int = 128,
        dropout_prob: float = 0.1,
        in_channels: int = 11,
        embed_dim: int = 4,
        num_heads: int = 2,
        depth: int = 3,
        window_size: Tuple[int, int] = (7, 5),
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.img_seq = img_seq
        self.text_seq = text_seq
        self.window_size = window_size
        shift = (window_size[0] // 2, window_size[1] // 2)

        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        blocks = []
        for i in range(depth):
            blocks.append(
                SwinTransformerBlock(
                    embed_dim,
                    num_heads,
                    window_size,
                    shift_size=shift if i % 2 == 1 else (0, 0),
                    mlp_ratio=mlp_ratio,
                    dropout=dropout_prob,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        flat_dim = embed_dim * img_seq * text_seq
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(flat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        b1, b2, _, h, w = x.shape
        flat = x.reshape(b1 * b2, x.size(2), h, w)
        flat, _ = _pad_to_window(flat, self.window_size)
        flat = self.input_proj(flat)
        for blk in self.blocks:
            flat = blk(flat)
        feat = flat[:, :, :h, :w]
        return feat.reshape(b1 * b2, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1, b2 = x.size(0), x.size(1)
        feat_flat = self._encode(x)
        out = self.dropout2(self.fc2(self.dropout1(F.relu(self.fc1(feat_flat)))))
        return out.view(b1, b2, 1)


def normalize_scorer_type(scorer_type: str) -> str:
    """Map legacy checkpoint tags to current scorer names."""
    if scorer_type in ("swin", "swin_v2", "swin_v1"):
        return "swin"
    return scorer_type


def load_scorer_state_dict(task_net: nn.Module, state: dict) -> None:
    """Load weights; ignore removed L_rel / legacy modules."""
    filtered = {
        k: v for k, v in state.items()
        if not k.startswith("rel_")
    }
    task_net.load_state_dict(filtered, strict=False)


def build_scorer(
    scorer_type: str,
    img_seq: int,
    text_seq: int,
    hidden_dim: int = 128,
    dropout_prob: float = 0.1,
    *,
    channel_mode: str = "mgr",
    in_channels: int | None = None,
) -> nn.Module:
    from src.util.models_and_dataloaders import CrossModalConvNetwork

    scorer_type = normalize_scorer_type(scorer_type)

    from src.util.dcsm_channels import num_input_channels

    if scorer_type == "cnn":
        if in_channels is not None:
            ic = in_channels
        elif channel_mode in ("mg", "mr", "mgr", "mrr"):
            ic = num_input_channels(channel_mode)
        else:
            ic = 1
        return CrossModalConvNetwork(
            img_seq=img_seq, text_seq=text_seq,
            hidden_dim=hidden_dim, dropout_prob=dropout_prob,
            in_channels=ic,
        )
    if scorer_type in ("mlp", "cnn_mlp"):
        # Flatten (img_seq, text_seq) → MLP; CNN trunk not used (kept in CrossModalConvNetwork).
        from src.util.models_and_dataloaders import CrossModalMLPNetwork

        return CrossModalMLPNetwork(
            img_seq=img_seq,
            text_seq=text_seq,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob,
        )
    if scorer_type in ("mlp_matched", "cnn_mlp_matched"):
        # Capacity-matched Flatten-2D MLP (no Conv); default hidden_dim=512 ≈ CNN params.
        from src.util.models_and_dataloaders import CrossModalMLPParamMatchedNetwork

        return CrossModalMLPParamMatchedNetwork(
            img_seq=img_seq,
            text_seq=text_seq,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob,
        )
    if scorer_type == "swin":
        ic = in_channels if in_channels is not None else num_input_channels(channel_mode)
        return CrossModalSwinNetwork(
            img_seq=img_seq, text_seq=text_seq,
            hidden_dim=hidden_dim, dropout_prob=dropout_prob,
            in_channels=ic,
        )
    if scorer_type in ("s3prime", "rgrc_s3prime"):
        from src.util.rgrc_scorer import RGRCScorerS3Prime

        return RGRCScorerS3Prime(
            mode="tied",
            use_geometry=True,
            latent_dim=hidden_dim,
            dropout=dropout_prob,
        )
    raise ValueError(f"Unknown scorer_type: {scorer_type}")
