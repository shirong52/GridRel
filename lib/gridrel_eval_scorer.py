"""DCSM scorer wrapper that reuses a shared frozen CLIP (avoids tokenizer reload)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

from src.evaluating.dcsm_scorer import load_dcsm_checkpoint
from lib.clip_dense_feats import (
    encode_dense_vt,
    encode_dense_vt_multi,
    normalize_dense_feat_layer,
    normalize_dense_feat_layers,
)
from src.util.swin_scorer import build_scorer, load_scorer_state_dict, normalize_scorer_type
from src.util.dcsm_channels import num_input_channels, stack_dcsm_channels
from src.training.training_util import standardize
import torch.nn as nn
import torch.nn.functional as F


class SharedClipDCSMScorer:
    """GridRel / DCSM eval scorer sharing one frozen CLIP."""
    """API-compatible with DCSMScorer.score_image_captions, one CLIP instance."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        model: CLIPModel,
        tokenizer: CLIPTokenizer,
        img_proc: CLIPImageProcessor,
        device: str,
        text_seq: int = 30,
        text_prefix: str = "",
        dense_feat_layer_override: str | None = None,
    ) -> None:
        self.device = device
        self.text_seq = text_seq
        self.text_prefix = text_prefix
        self.model = model
        self.tokenizer = tokenizer
        self.img_proc = img_proc
        self.special_words = [
            "above", "below", "left", "right", "small", "big",
            "not", "no", "without", "but", "absent",
        ]

        state = load_dcsm_checkpoint(checkpoint, map_location=device)
        raw_scorer_type = str(state.get("scorer_type", "cnn"))
        scorer_type = normalize_scorer_type(raw_scorer_type)
        self._topology_scorer = False
        _cm = state.get("channel_mode", None)
        if _cm is None or str(_cm) in ("None", ""):
            self.channel_mode = "mgr" if scorer_type == "swin" else "m"
        else:
            self.channel_mode = str(_cm)
        if self.channel_mode == "gridrel":
            self.channel_mode = "mgr_a"
        self.use_gr_channels = scorer_type == "swin" or (
            scorer_type == "cnn" and self.channel_mode in ("mg", "mr", "mgr", "mrr")
        )
        # FR: prefer checkpoint metadata. Legacy multi-channel ckpts without flags stay FR off.
        if "use_functional_rows" in state:
            self.use_functional_rows = bool(state["use_functional_rows"])
        elif "no_replacements" in state:
            self.use_functional_rows = not bool(state["no_replacements"])
        elif self.use_gr_channels:
            self.use_functional_rows = False
        else:
            # Classic paper M-only defaults to FR on when metadata is missing.
            self.use_functional_rows = True

        in_ch = None
        self.shuffle_g = bool(state.get("shuffle_g", False))
        self.shuffle_g_seed = int(state.get("shuffle_g_seed", 15))
        self.random_r = bool(state.get("random_r", False))
        self.random_r_seed = int(state.get("random_r_seed", 15))
        self.pad_to_mgr = bool(state.get("pad_to_mgr", False))
        self.pad_to_mrr = bool(state.get("pad_to_mrr", False))
        hidden_dim = int(state.get("hidden_dim", 128))

        if self.use_gr_channels or self.pad_to_mgr or self.pad_to_mrr:
            self.use_gr_channels = True
            in_ch = num_input_channels(
                self.channel_mode,
                pad_to_mgr=self.pad_to_mgr,
                pad_to_mrr=self.pad_to_mrr,
            )

        # Resolve dense feature layers (multi-layer DCSM or single-layer legacy).
        # Eval-time override for layer×head cross ablation (P0.1) forces a single layer.
        self.dense_feat_vision = None
        self.dense_feat_text = None
        if dense_feat_layer_override is not None and str(dense_feat_layer_override).strip():
            self.dense_feat_layers = [
                normalize_dense_feat_layer(dense_feat_layer_override)
            ]
            self._dense_feat_layer_overridden = True
        elif state.get("dense_feat_layers") is not None:
            self.dense_feat_layers = normalize_dense_feat_layers(
                state.get("dense_feat_layers")
            )
            self._dense_feat_layer_overridden = False
        elif state.get("dense_feat_vision") is not None or state.get("dense_feat_text") is not None:
            base = state.get("dense_feat_layer", "last")
            self.dense_feat_vision = normalize_dense_feat_layer(
                state.get("dense_feat_vision", base)
            )
            self.dense_feat_text = normalize_dense_feat_layer(
                state.get("dense_feat_text", base)
            )
            self.dense_feat_layers = [self.dense_feat_vision]
            self._dense_feat_layer_overridden = False
        else:
            self.dense_feat_layers = [
                normalize_dense_feat_layer(state.get("dense_feat_layer", "last"))
            ]
            self._dense_feat_layer_overridden = False
        self.dense_feat_layer = self.dense_feat_layers[0]
        n_dense = len(self.dense_feat_layers)
        # Classic M-only multi-layer: in_channels = number of dense maps.
        if self.channel_mode == "ma":
            in_ch = 2
        elif in_ch is None and not self.use_gr_channels and n_dense > 1:
            in_ch = n_dense

        patch_grid = int(state.get("patch_grid", 14))
        img_seq = patch_grid * patch_grid + 1
        import src.util.dcsm_channels as dch

        dch.PATCH_GRID = patch_grid

        _topology_types = {
            "reshape2d", "reshape2d_m", "conv3d", "conv3d_m", "reshape3d", "reshape3d_m",
            "reshape3d_mlp", "conv3d_mlp", "reshape3d_mlp_m", "conv3d_mlp_m",
            "reshape3d_mlp_matched", "conv3d_mlp_matched",
            "conv3d_kt1", "reshape3d_kt1",
            "spatial_factor_joint", "spatial_factor_joint_m",
        }
        if raw_scorer_type in _topology_types:
            from pathlib import Path as _Path
            import sys as _sys

            chain3d = _Path(__file__).resolve().parents[1]
            if str(chain3d) not in _sys.path:
                _sys.path.insert(0, str(chain3d))
            from lib.topology import set_patch_grid

            set_patch_grid(patch_grid)
            from lib.conv3d_scorer import build_topology_scorer, load_topology_scorer_state

            self._topology_scorer = True
            if n_dense > 1:
                raise ValueError(
                    "multi-layer DCSM not supported for topology scorers in SharedClipDCSMScorer"
                )
            self.task_net = build_topology_scorer(
                raw_scorer_type,
                text_seq=text_seq,
                hidden_dim=hidden_dim,
                dropout=0.0,
                channel_mode=self.channel_mode,
            ).to(device)
            load_topology_scorer_state(self.task_net, state["model_state_dict"])
        else:
            self.task_net = build_scorer(
                scorer_type=scorer_type,
                img_seq=img_seq,
                text_seq=text_seq,
                hidden_dim=hidden_dim,
                dropout_prob=0.0,
                channel_mode=self.channel_mode,
                in_channels=in_ch,
            ).to(device)
            load_scorer_state_dict(self.task_net, state["model_state_dict"])
        self.bilinear = None
        if state.get("alignment_kind") == "low_rank_bilinear":
            from pathlib import Path as _Path
            import sys as _sys

            chain3d = _Path(__file__).resolve().parents[1]
            if str(chain3d) not in _sys.path:
                _sys.path.insert(0, str(chain3d))
            from lib.bilinear_metric import LowRankBilinear

            rank = int(state.get("alignment_rank", state["alignment_hidden"]))
            self.bilinear = LowRankBilinear(
                dim=int(state["alignment_dim"]), rank=rank
            ).to(device)
            self.bilinear.load_state_dict(state["alignment_state_dict"])
            self.bilinear.eval()
        self.task_net.eval()

        ext = state.get("external_logic_token_embeddings")
        lookup = state.get("learned_lookup")
        if lookup:
            self.special_words = list(lookup)
        elif ext is not None and hasattr(ext, "num_embeddings"):
            n = int(ext.num_embeddings)
            self.special_words = [
                "above", "below", "left", "right", "small", "big",
                "not", "no", "without", "but", "absent",
            ][:n]
        if ext is None:
            ext = nn.Embedding(len(self.special_words), 14 * 14 + 1).to(device)
            with torch.no_grad():
                ext.weight.copy_((F.normalize(ext.weight, p=2, dim=1) + 1) / 3)
        self.ext_emb = ext.to(device)

    def _format_texts(self, captions: Sequence[str]) -> list[str]:
        if self.use_gr_channels or getattr(self, "_topology_scorer", False):
            base = [self.text_prefix + c for c in captions]
        else:
            base = [self.text_prefix + c for c in captions]
        return [
            t.replace(" behind ", " above ")
            .replace(" in front of ", " below ")
            .replace(" bottom", " below")
            .replace(" top", " above")
            for t in base
        ]

    @torch.no_grad()
    def _dense_feats(self, images: Sequence[Image.Image], texts: Sequence[str]):
        v_in = self.img_proc(images=list(images), return_tensors="pt").to(self.device)
        t_in = self.tokenizer(
            list(texts), padding="max_length", max_length=self.text_seq,
            truncation=True, return_tensors="pt",
        ).to(self.device)
        if len(self.dense_feat_layers) == 1:
            if self.dense_feat_vision is not None or self.dense_feat_text is not None:
                return encode_dense_vt(
                    self.model,
                    v_in,
                    t_in,
                    vision_layer=self.dense_feat_vision,
                    text_layer=self.dense_feat_text,
                )
            return encode_dense_vt(
                self.model, v_in, t_in, layer=self.dense_feat_layers[0]
            )
        return encode_dense_vt_multi(
            self.model, v_in, t_in, layers=self.dense_feat_layers
        )

    def _apply_fr_4d(
        self, cossim: torch.Tensor, texts: Sequence[str], n_img: int
    ) -> torch.Tensor:
        """Apply FR on a single-channel map (B1,B2,H,W)."""
        if not self.use_functional_rows:
            return cossim
        new_maps = []
        for k, txt in enumerate(texts):
            sent = txt.split(" ")
            temp = cossim[:, k, :, :]
            for sp_w in self.special_words:
                if sp_w in sent:
                    idx = sent.index(sp_w)
                    mid = self.ext_emb(
                        torch.tensor([self.special_words.index(sp_w)], device=self.device)
                    )
                    mid = mid.unsqueeze(2).repeat(n_img, 1, 1)
                    temp = torch.cat(
                        [temp[:, :, :idx], mid, temp[:, :, idx + 1 :]], dim=-1
                    )
                    temp = temp[:, :, : self.text_seq]
            new_maps.append(temp)
        return torch.stack(new_maps, dim=1)

    @torch.no_grad()
    def score_image_captions(
        self,
        image: Union[Image.Image, Sequence[Image.Image]],
        captions: Sequence[str],
    ) -> torch.Tensor:
        images = image if isinstance(image, (list, tuple)) else [image]
        texts = self._format_texts(captions)
        n_img, n_txt = len(images), len(texts)
        feats = self._dense_feats(images, texts)

        aligned = None
        if len(self.dense_feat_layers) == 1:
            v_h, t_h = feats
            if getattr(self, "bilinear", None) is not None:
                aligned = self.bilinear.pairwise_map(v_h, t_h)
                aligned = standardize(aligned)
            cossim = torch.einsum(
                "bqie,lpte->bpit", v_h.unsqueeze(1), t_h.unsqueeze(0)
            )
            cossim = self._apply_fr_4d(cossim, texts, n_img)
            cossim = standardize(cossim)
        else:
            v_list, t_list = feats
            maps = []
            for v_h, t_h in zip(v_list, t_list):
                cm = torch.einsum(
                    "bqie,lpte->bpit", v_h.unsqueeze(1), t_h.unsqueeze(0)
                )
                cm = self._apply_fr_4d(cm, texts, n_img)
                maps.append(standardize(cm))
            # (B1,B2,L,H,W)
            cossim = torch.stack(maps, dim=2)

        scores = []
        for i in range(n_img):
            row = []
            for j in range(n_txt):
                mat = cossim[i : i + 1, j : j + 1]
                if self._topology_scorer:
                    texts_arg = [texts[j]] if self.channel_mode != "m" else None
                    mat_a = None if aligned is None else aligned[i : i + 1, j : j + 1]
                    row.append(
                        self.task_net(mat, texts_arg, aligned_cossim=mat_a).reshape(())
                    )
                elif self.channel_mode == "ma":
                    if aligned is None:
                        raise RuntimeError("channel_mode=ma requires the frozen bilinear map")
                    mat_a = aligned[i : i + 1, j : j + 1]
                    stacked = torch.cat([mat.unsqueeze(2), mat_a.unsqueeze(2)], dim=2)
                    row.append(self.task_net(stacked).reshape(()))
                elif self.use_gr_channels:
                    # GR stacking expects single-channel (B1,B2,H,W)
                    if mat.dim() == 5:
                        raise ValueError("multi-layer + G/R channels not supported")
                    mat = stack_dcsm_channels(
                        mat,
                        [texts[j]],
                        channel_mode=self.channel_mode,
                        shuffle_g=self.shuffle_g,
                        shuffle_g_seed=self.shuffle_g_seed,
                        random_r=self.random_r,
                        random_r_seed=self.random_r_seed,
                        pad_to_mgr=self.pad_to_mgr,
                        pad_to_mrr=self.pad_to_mrr,
                    )
                    row.append(self.task_net(mat).reshape(()))
                else:
                    row.append(self.task_net(mat).reshape(()))
            scores.append(torch.stack(row))
        out = torch.stack(scores)
        return out.reshape(len(images), len(captions))

    @torch.no_grad()
    def score_pair(self, image: Image.Image, caption: str) -> float:
        return float(self.score_image_captions(image, [caption]).reshape(-1)[0].item())
