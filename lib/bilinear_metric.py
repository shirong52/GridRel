"""Low-rank correction of the CLIP dot product.

    s(v, t) = v·t + alpha * (A^T v)·(B^T t)

alpha starts at 0, so the first forward pass is the raw dot product.
A and B are small random factors. Setting both to exact zeros would
leave the product with no gradient. The matrix is shared across patches
and tokens. It cannot copy a token into a patch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LowRankBilinear(nn.Module):
    def __init__(self, dim: int = 512, rank: int = 16) -> None:
        super().__init__()
        self.dim = dim
        self.rank = rank
        scale = dim ** -0.5
        self.A = nn.Parameter(torch.randn(dim, rank) * scale)
        self.B = nn.Parameter(torch.randn(dim, rank) * scale)
        self.alpha = nn.Parameter(torch.zeros(()))

    def scores(self, patches: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """patches [N, D], tokens [T, D] -> [T, N]."""
        if patches.dim() != 2 or tokens.dim() != 2:
            raise ValueError(
                f"expected patches [N,D] and tokens [T,D], got {tuple(patches.shape)} {tuple(tokens.shape)}"
            )
        base = tokens @ patches.T
        extra = (tokens @ self.B) @ (patches @ self.A).T
        return base + self.alpha * extra

    def pairwise_map(self, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        """images [B1, 197, D], texts [B2, T, D] -> [B1, B2, 197, T].

        Patch rows use the learned metric. The CLS row stays the raw dot
        product; the spatial scorer drops that row.
        """
        if images.dim() != 3 or texts.dim() != 3:
            raise ValueError(
                f"expected images [B1,S,D] and texts [B2,T,D], got {tuple(images.shape)} {tuple(texts.shape)}"
            )
        patches = images[:, 1:, :]
        cls = torch.einsum("id,jtd->ijt", images[:, 0], texts)
        base = torch.einsum("ind,jtd->ijnt", patches, texts)
        extra = torch.einsum("jtr,inr->ijnt", texts @ self.B, patches @ self.A)
        body = base + self.alpha * extra
        return torch.cat([cls.unsqueeze(2), body], dim=2)

    def factor_norm(self) -> tuple[float, float, float]:
        return (
            float(self.A.detach().norm()),
            float(self.B.detach().norm()),
            float(self.alpha.detach()),
        )


def region_mean(cell: torch.Tensor, token_index: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
    """Mean score of the selected tokens on the selected patches.

    cell: [T, N], token_index: [K], patch_mask: [N] bool.
    """
    chosen = cell.index_select(0, token_index)
    values = chosen[:, patch_mask.bool()]
    if values.numel() == 0:
        raise ValueError("region_mean received an empty token or patch set")
    return values.mean()


def swap_loss(
    cell: torch.Tensor,
    index_a: torch.Tensor,
    index_b: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Hinge: token A should prefer region A, token B should prefer region B."""
    q_aa = region_mean(cell, index_a, mask_a)
    q_ab = region_mean(cell, index_a, mask_b)
    q_bb = region_mean(cell, index_b, mask_b)
    q_ba = region_mean(cell, index_b, mask_a)
    return F.relu(margin - q_aa + q_ab) + F.relu(margin - q_bb + q_ba)
