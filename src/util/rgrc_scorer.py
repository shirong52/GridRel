"""S3′: role-aware geometry–relation composition with a spatial role encoder.

Appended as a standalone scorer. Forward contract differs from CNN-M:

    CNN-M:      f(M)                         -> [B1, B2, 1]
    S3′ / RGRC: f(M, subj_mask, obj_mask, r) -> [B1, B2]

CLIP stays frozen; only this module is trained.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

DEFAULT_CLIP_PATH = "/root/autodl-tmp/model/clip_vit_b"
IMG_SEQ = 14 * 14 + 1
TEXT_SEQ = 30
RGRC_PATCH_GRID = 14
RGRC_LATENT = 128
RGRC_HEAD = 64

SEEN_RELATIONS = ("above", "behind", "leftof", "on")
HELD_OUT_INVERSES = ("below", "in front of", "rightof", "under")

INVERSE_OF = {
    "above": "below",
    "below": "above",
    "leftof": "rightof",
    "rightof": "leftof",
    "on": "under",
    "under": "on",
    "behind": "in front of",
    "in front of": "behind",
}

RELATION_PHRASE = {
    "above": "above",
    "below": "below",
    "leftof": "to the left of",
    "rightof": "to the right of",
    "on": "on",
    "under": "under",
    "behind": "behind",
    "in front of": "in front of",
}

RGRC_RELATIONS = tuple(INVERSE_OF.keys())
RGRC_REL2ID = {r: i for i, r in enumerate(RGRC_RELATIONS)}
RGRC_CANON = (
    ("above", "below"),
    ("leftof", "rightof"),
    ("on", "under"),
    ("behind", "in front of"),
)
RGRC_CANON_ID = {r: i for i, pair in enumerate(RGRC_CANON) for r in pair}
RGRC_CANON_SIGN = {r: (1 if r == pair[0] else -1) for pair in RGRC_CANON for r in pair}

_TOKENIZER = None


def render_caption(subject: str, relation: str, obj: str) -> str:
    phrase = RELATION_PHRASE[relation.lower().strip()]
    return f"A picture of a {subject} {phrase} a {obj}"


def inverse_triplet(subject: str, relation: str, obj: str) -> tuple[str, str, str]:
    relation = relation.lower().strip()
    return obj, INVERSE_OF[relation], subject


def role_char_spans(subject: str, relation: str, obj: str):
    phrase = RELATION_PHRASE[relation.lower().strip()]
    prefix = "A picture of a "
    subj_start = len(prefix)
    subj_end = subj_start + len(subject)
    rel_start = subj_end + 1
    rel_end = rel_start + len(phrase)
    mid = " a "
    obj_start = rel_end + len(mid)
    obj_end = obj_start + len(obj)
    return (subj_start, subj_end), (rel_start, rel_end), (obj_start, obj_end)


def get_role_tokenizer(clip_path: str = DEFAULT_CLIP_PATH):
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import CLIPTokenizerFast

        _TOKENIZER = CLIPTokenizerFast.from_pretrained(clip_path)
    return _TOKENIZER


def build_role_masks(
    subjects: Sequence[str],
    relations: Sequence[str],
    objects: Sequence[str],
    text_seq: int = TEXT_SEQ,
    tokenizer=None,
) -> tuple[np.ndarray, np.ndarray]:
    subjects = list(map(str, subjects))
    relations = list(map(str, relations))
    objects = list(map(str, objects))
    captions = [render_caption(s, r, o) for s, r, o in zip(subjects, relations, objects)]
    tok = tokenizer if tokenizer is not None else get_role_tokenizer()
    enc = tok(
        captions,
        add_special_tokens=True,
        padding="max_length",
        max_length=text_seq,
        truncation=True,
        return_offsets_mapping=True,
    )
    n = len(captions)
    subj_mask = np.zeros((n, text_seq), dtype=bool)
    obj_mask = np.zeros((n, text_seq), dtype=bool)
    for i, (s, r, o) in enumerate(zip(subjects, relations, objects)):
        (ss, se), _, (os_, oe) = role_char_spans(s, r, o)
        offsets = enc["offset_mapping"][i]
        for j, (a, b) in enumerate(offsets):
            if b <= a:
                continue
            if a < se and b > ss:
                subj_mask[i, j] = True
            if a < oe and b > os_:
                obj_mask[i, j] = True
    return subj_mask, obj_mask


def build_patch_coords(
    img_seq: int = IMG_SEQ, grid: int = RGRC_PATCH_GRID
) -> torch.Tensor:
    coords = torch.zeros(img_seq, 2)
    coords[0] = torch.tensor([0.5, 0.5])
    for k in range(1, img_seq):
        idx = k - 1
        row, col = divmod(idx, grid)
        coords[k, 0] = (col + 0.5) / grid
        coords[k, 1] = (row + 0.5) / grid
    return coords


class SpatialRoleEncoder(nn.Module):
    """Encode a role evidence profile x ∈ R^{197} = [CLS | 14x14 patches] with 2D locality.

    Controlled change vs S3 Linear(197→128): same I/O shape, patches as a spatial map.
    """

    def __init__(self, latent_dim: int = RGRC_LATENT, grid: int = RGRC_PATCH_GRID):
        super().__init__()
        self.grid = grid
        self.cls_proj = nn.Linear(1, 32)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.out = nn.Sequential(nn.Linear(32 + 64, latent_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        cls = flat[:, :1]
        patches = flat[:, 1:].reshape(-1, 1, self.grid, self.grid)
        h = torch.cat([self.cls_proj(cls), self.conv(patches)], dim=-1)
        return self.out(h).reshape(*lead, -1)


class RGRCScorer(nn.Module):
    """Role-aware Geometry-Relation Composition (S1/S2/S3 linear role encoder)."""

    def __init__(
        self,
        mode: str = "tied",
        use_geometry: bool = True,
        img_seq: int = IMG_SEQ,
        latent_dim: int = RGRC_LATENT,
        head_dim: int = RGRC_HEAD,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert latent_dim % 2 == 0
        assert mode in {"independent", "tied"}
        self.mode = mode
        self.use_geometry = use_geometry
        self.latent_dim = latent_dim
        self.role_encoder = nn.Sequential(nn.Linear(img_seq, latent_dim), nn.GELU())
        compose_in = latent_dim * 2 + (2 if use_geometry else 0)
        self.composer = nn.Sequential(nn.Linear(compose_in, latent_dim), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(latent_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )
        n_theta = 8 if mode == "independent" else 4
        self.theta = nn.Parameter(torch.zeros(n_theta, latent_dim // 2))
        self.register_buffer("patch_coords", build_patch_coords(img_seq), persistent=False)

    @staticmethod
    def _pool_columns(cossim: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(cossim.dtype)
        denom = m.sum(dim=-1).clamp_min(1.0)
        pooled = torch.einsum("bphw,pw->bph", cossim, m)
        return pooled / denom.view(1, -1, 1)

    def compose(
        self, cossim: torch.Tensor, subj_mask: torch.Tensor, obj_mask: torch.Tensor
    ) -> torch.Tensor:
        zs_raw = self._pool_columns(cossim, subj_mask)
        zo_raw = self._pool_columns(cossim, obj_mask)
        zs = self.role_encoder(zs_raw)
        zo = self.role_encoder(zo_raw)
        feats = [zs, zo]
        if self.use_geometry:
            attn_s = torch.softmax(zs_raw, dim=-1)
            attn_o = torch.softmax(zo_raw, dim=-1)
            p_s = torch.einsum("bph,hc->bpc", attn_s, self.patch_coords)
            p_o = torch.einsum("bph,hc->bpc", attn_o, self.patch_coords)
            feats.append(p_o - p_s)
        return self.composer(torch.cat(feats, dim=-1))

    def relation_angles(self, relation_ids: torch.Tensor) -> torch.Tensor:
        relation_ids = relation_ids.long()
        if self.mode == "independent":
            return self.theta[relation_ids]
        names = RGRC_RELATIONS
        pair_ids = torch.as_tensor(
            [RGRC_CANON_ID[r] for r in names],
            device=relation_ids.device,
            dtype=torch.long,
        )[relation_ids]
        signs = torch.as_tensor(
            [RGRC_CANON_SIGN[r] for r in names],
            device=relation_ids.device,
            dtype=self.theta.dtype,
        )[relation_ids]
        return self.theta[pair_ids] * signs.unsqueeze(1)

    def transform(self, u0: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        b1, b2, d = u0.shape
        ang = self.relation_angles(relation_ids).unsqueeze(0).expand(b1, -1, -1)
        zz = u0.reshape(b1, b2, d // 2, 2)
        c, s = ang.cos(), ang.sin()
        x, y = zz[..., 0], zz[..., 1]
        return torch.stack((c * x - s * y, s * x + c * y), dim=-1).reshape(b1, b2, d)

    def relation_latent(self, cossim, subj_mask, obj_mask, relation_ids):
        u0 = self.compose(cossim, subj_mask, obj_mask)
        return self.transform(u0, relation_ids), u0

    def forward(self, cossim, subj_mask, obj_mask, relation_ids):
        u, _ = self.relation_latent(cossim, subj_mask, obj_mask, relation_ids)
        return self.head(u.reshape(-1, self.latent_dim)).reshape(u.shape[0], u.shape[1])


class RGRCScorerS3Prime(RGRCScorer):
    """S3′: same as tied+geometry RGRC, with SpatialRoleEncoder instead of Linear."""

    def __init__(
        self,
        mode: str = "tied",
        use_geometry: bool = True,
        img_seq: int = IMG_SEQ,
        latent_dim: int = RGRC_LATENT,
        head_dim: int = RGRC_HEAD,
        dropout: float = 0.1,
    ):
        super().__init__(
            mode=mode,
            use_geometry=use_geometry,
            img_seq=img_seq,
            latent_dim=latent_dim,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.role_encoder = SpatialRoleEncoder(latent_dim=latent_dim)


def relation_ids_from_names(names: Sequence[str], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        [RGRC_REL2ID[str(n)] for n in names], device=device, dtype=torch.long
    )


def role_masks_to_torch(
    subjects: Sequence[str],
    relations: Sequence[str],
    objects: Sequence[str],
    device: torch.device,
    text_seq: int = TEXT_SEQ,
    tokenizer=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    subj, obj = build_role_masks(
        subjects, relations, objects, text_seq=text_seq, tokenizer=tokenizer
    )
    return (
        torch.from_numpy(subj).to(device=device),
        torch.from_numpy(obj).to(device=device),
    )
