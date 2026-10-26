"""Dense CLIP patch/token features for DCSM cosine maps.

layer='last'  — current DCSM/L3 behaviour (final encoder layer)
layer='penult' — one encoder layer earlier (hidden_states[-2])
"""

from __future__ import annotations

from typing import Literal, Sequence

import torch
import torch.nn.functional as F

DenseFeatLayer = Literal["last", "penult"]


def normalize_dense_feat_layer(layer: str | None) -> DenseFeatLayer:
    if layer is None or str(layer).strip() == "":
        return "last"
    key = str(layer).strip().lower()
    if key in ("last", "final", "-1"):
        return "last"
    if key in ("penult", "penultimate", "-2"):
        return "penult"
    raise ValueError(f"dense_feat_layer must be 'last' or 'penult', got {layer!r}")


def normalize_dense_feat_layers(
    layers: Sequence[str] | str | None,
) -> list[DenseFeatLayer]:
    """Normalize a layer list for multi-layer DCSM (e.g. ['last','penult'])."""
    if layers is None:
        return ["last"]
    if isinstance(layers, str):
        # allow comma-separated config strings
        parts = [p.strip() for p in layers.split(",") if p.strip()]
        if not parts:
            return ["last"]
        return [normalize_dense_feat_layer(p) for p in parts]
    out = [normalize_dense_feat_layer(p) for p in layers]
    if not out:
        raise ValueError("dense_feat_layers must be a non-empty list")
    return out


def encode_dense_vt(
    model,
    vision_inputs: dict,
    text_tokens: dict,
    *,
    layer: str = "last",
    vision_layer: str | None = None,
    text_layer: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return L2-normalized dense image patches and text tokens [B, S, D].

    Symmetric (default): both modalities use ``layer``.
    Asymmetric: set ``vision_layer`` / ``text_layer`` independently
    (e.g. vision=penult, text=last) while keeping a single DCSM channel.
    """
    v_layer = normalize_dense_feat_layer(
        vision_layer if vision_layer is not None else layer
    )
    t_layer = normalize_dense_feat_layer(
        text_layer if text_layer is not None else layer
    )
    want_v_hs = v_layer == "penult"
    want_t_hs = t_layer == "penult"

    vision_outputs = model.vision_model(
        **vision_inputs, output_hidden_states=want_v_hs
    )
    if v_layer == "penult":
        v_raw = vision_outputs.hidden_states[-2]
    else:
        v_raw = vision_outputs.last_hidden_state
    v_h = model.visual_projection(model.vision_model.post_layernorm(v_raw))
    v_h = F.normalize(v_h, p=2, dim=-1)

    text_outputs = model.text_model(**text_tokens, output_hidden_states=want_t_hs)
    if t_layer == "penult":
        # Mirror final-layer path: LN then projection (last_hidden_state already LN'd).
        t_raw = model.text_model.final_layer_norm(text_outputs.hidden_states[-2])
    else:
        t_raw = text_outputs.last_hidden_state
    t_h = model.text_projection(t_raw)
    t_h = F.normalize(t_h, p=2, dim=-1)
    return v_h, t_h


def encode_dense_vt_multi(
    model,
    vision_inputs: dict,
    text_tokens: dict,
    *,
    layers: Sequence[str] = ("last", "penult"),
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Encode multiple CLIP layers in one forward; return lists of [B,S,D] feats.

    Order of returned tensors matches ``normalize_dense_feat_layers(layers)``.
    """
    layers_n = normalize_dense_feat_layers(layers)
    need_penult = "penult" in layers_n
    vision_outputs = model.vision_model(
        **vision_inputs, output_hidden_states=need_penult
    )
    text_outputs = model.text_model(
        **text_tokens, output_hidden_states=need_penult
    )

    v_list: list[torch.Tensor] = []
    t_list: list[torch.Tensor] = []
    for ly in layers_n:
        if ly == "penult":
            v_raw = vision_outputs.hidden_states[-2]
            t_raw = model.text_model.final_layer_norm(text_outputs.hidden_states[-2])
        else:
            v_raw = vision_outputs.last_hidden_state
            t_raw = text_outputs.last_hidden_state
        v_h = model.visual_projection(model.vision_model.post_layernorm(v_raw))
        v_h = F.normalize(v_h, p=2, dim=-1)
        t_h = F.normalize(model.text_projection(t_raw), p=2, dim=-1)
        v_list.append(v_h)
        t_list.append(t_h)
    return v_list, t_list
