"""Subject/object token column masks for G_rel anchor pooling."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

from src.util.rgrc_scorer import TEXT_SEQ, build_role_masks
from src.util.spatial_caption_parse import parse_spatial_triplet


def triplet_from_caption(caption: str) -> Tuple[str, str, str] | None:
    """Parse (subject, relation_key, object) from a spatial caption."""
    return parse_spatial_triplet(caption)


def role_masks_for_texts(
    texts: Sequence[str],
    *,
    text_seq: int = TEXT_SEQ,
    device: torch.device | str | None = None,
    tokenizer=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (subj_mask, obj_mask) each (len(texts), text_seq) bool on ``device``."""
    subjects: list[str] = []
    relations: list[str] = []
    objects: list[str] = []
    for text in texts:
        trip = triplet_from_caption(text)
        if trip is None:
            subjects.append("")
            relations.append("on")
            objects.append("")
        else:
            subjects.append(trip[0])
            relations.append(trip[1])
            objects.append(trip[2])

    subj_np, obj_np = build_role_masks(
        subjects,
        relations,
        objects,
        text_seq=text_seq,
        tokenizer=tokenizer,
    )
    subj = torch.from_numpy(subj_np)
    obj = torch.from_numpy(obj_np)
    if device is not None:
        subj = subj.to(device)
        obj = obj.to(device)
    return subj, obj
