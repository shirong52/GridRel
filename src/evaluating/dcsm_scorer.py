"""DCSM + frozen CLIP scorer for standard benchmark evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

from src.training.training_util import standardize
from src.util.dcsm_channels import num_input_channels, stack_dcsm_channels
from src.util.swin_scorer import build_scorer, load_scorer_state_dict, normalize_scorer_type
from lib.clip_dense_feats import (
    encode_dense_vt,
    encode_dense_vt_multi,
    normalize_dense_feat_layer,
    normalize_dense_feat_layers,
)

DEFAULT_CLIP_PATH = "/root/autodl-tmp/model/clip_vit_b"
SPECIAL_WORDS = [
    "above", "below", "left", "right", "small", "big",
    "not", "no", "without", "but", "absent",
]

# Training uses render_caption / ObjaverseDataset: "A picture of a {subj} {rel} a {obj}".
# For benchmarks whose captions lack that template, prepend this at eval time.
DCSM_EVAL_TEXT_PREFIX = "A picture of "


def load_clip(path: str, device: str) -> tuple[CLIPModel, CLIPTokenizer, CLIPImageProcessor]:
    model = CLIPModel.from_pretrained(path).eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model, CLIPTokenizer.from_pretrained(path), CLIPImageProcessor.from_pretrained(path)


def _patch_transformers_trie_for_unpickle() -> None:
    """Patch older DCSM checkpoint pickling incompatibilities with newer transformers.

    Older checkpoints embed CLIPTokenizer which references classes that have been
    removed or moved in transformers>=5:
      - tokenization_utils.Trie -> tokenization_python.Trie
      - tokenization_clip.BasicTokenizer (removed in newer transformers)

    DCSMScorer does not use the pickled tokenizer (it reloads CLIP from clip_path),
    so dummy compatibility aliases are sufficient.
    """
    import sys

    # --- Trie compat ---
    try:
        from transformers.tokenization_python import Trie as _Trie
    except Exception:  # pragma: no cover
        class _Trie:  # noqa: N801
            def __init__(self, *args, **kwargs):
                pass

            def __setstate__(self, state):
                pass

    import transformers.tokenization_utils as tu
    tu.Trie = _Trie  # type: ignore[attr-defined]
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if "tokenization_utils" in mod_name and not hasattr(mod, "Trie"):
            try:
                setattr(mod, "Trie", _Trie)
            except Exception:
                pass

    # --- BasicTokenizer compat (removed from tokenization_clip in newer transformers) ---
    class _BasicTokenizer:  # noqa: N801
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

    try:
        import transformers.models.clip.tokenization_clip as _tc
        if not hasattr(_tc, "BasicTokenizer"):
            _tc.BasicTokenizer = _BasicTokenizer  # type: ignore[attr-defined]
    except Exception:
        pass


def _load_with_skip_tokenizer_pickle(
    checkpoint: str | Path, map_location: str | torch.device
) -> dict:
    """Load ckpt while stubbing transformers objects embedded in old pickles."""
    import pickle

    class _SkipTokUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith(("transformers", "tokenizers")):

                def _init(self, *args, **kwargs):
                    return None

                def _setstate(self, state):
                    return None

                return type(
                    name,
                    (),
                    {"__init__": _init, "__setstate__": _setstate},
                )
            return super().find_class(module, name)

    class _SkipTokPickle:
        Unpickler = _SkipTokUnpickler
        load = staticmethod(lambda f: _SkipTokUnpickler(f).load())
        dump = pickle.dump
        dumps = pickle.dumps
        Pickler = pickle.Pickler

    return torch.load(
        checkpoint,
        map_location=map_location,
        weights_only=False,
        pickle_module=_SkipTokPickle,
    )


def load_dcsm_checkpoint(checkpoint: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """torch.load DCSM training ckpt with transformers Trie compatibility.

    Prefer a clean load first. Only monkey-patch transformers Trie/BasicTokenizer when
    unpickling fails — permanent patching breaks subsequent CLIPTokenizer.from_pretrained.
    """
    try:
        return torch.load(checkpoint, map_location=map_location, weights_only=False)
    except Exception:
        pass

    import transformers.tokenization_utils as tu

    orig_trie = getattr(tu, "Trie", None)
    _patch_transformers_trie_for_unpickle()
    try:
        return torch.load(checkpoint, map_location=map_location, weights_only=False)
    except Exception:
        pass
    finally:
        if orig_trie is not None:
            tu.Trie = orig_trie  # type: ignore[attr-defined]

    return _load_with_skip_tokenizer_pickle(checkpoint, map_location)


def clip_feature_tensor(out) -> torch.Tensor:
    """Normalize CLIP get_*_features output across transformers versions.

    - transformers<=4: returns a Tensor
    - transformers>=5: returns BaseModelOutputWithPooling; projected feats in pooler_output
    """
    if torch.is_tensor(out):
        return out
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None and torch.is_tensor(pooled):
        return pooled
    if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
        return out[0]
    raise TypeError(f"Unexpected CLIP feature type: {type(out)}")


class ClipCosineScorer:
    """Standard CLIP image-text cosine similarity."""

    def __init__(
        self,
        model: CLIPModel,
        tokenizer: CLIPTokenizer,
        img_proc: CLIPImageProcessor,
        device: str,
        *,
        text_prefix: str = "",
        text_max_length: int = 77,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.img_proc = img_proc
        self.device = device
        self.text_prefix = text_prefix
        self.text_max_length = text_max_length

    @torch.no_grad()
    def score_image_captions(
        self,
        image: Union[Image.Image, Sequence[Image.Image]],
        captions: Sequence[str],
    ) -> torch.Tensor:
        images = image if isinstance(image, (list, tuple)) else [image]
        texts = [self.text_prefix + c for c in captions]
        v_in = self.img_proc(images=images, return_tensors="pt").to(self.device)
        img_feat = F.normalize(clip_feature_tensor(self.model.get_image_features(**v_in)), dim=-1)
        t_in = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.text_max_length, return_tensors="pt",
        ).to(self.device)
        txt_feat = F.normalize(clip_feature_tensor(self.model.get_text_features(**t_in)), dim=-1)
        return (img_feat @ txt_feat.T).squeeze(0)

    @torch.no_grad()
    def score_pair(self, image: Image.Image, caption: str) -> float:
        return float(self.score_image_captions(image, [caption])[0].item())


class DCSMScorer:
    """Author DCSM: dense map + functional rows + CNN or Swin-style scorer."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        clip_path: str = DEFAULT_CLIP_PATH,
        device: str = "cuda",
        text_seq: int = 30,
        hidden_dim: int = 128,
        text_prefix: str = DCSM_EVAL_TEXT_PREFIX,
        use_functional_rows: bool = True,
        dense_feat_layer_override: str | None = None,
    ) -> None:
        self.device = device
        self.text_seq = text_seq
        self.text_prefix = text_prefix
        self.use_functional_rows = use_functional_rows
        self.special_words = SPECIAL_WORDS

        self.model, self.tokenizer, self.img_proc = load_clip(clip_path, device)
        state = load_dcsm_checkpoint(checkpoint, map_location=device)

        scorer_type = normalize_scorer_type(state.get("scorer_type", "cnn"))
        self.scorer_type = scorer_type
        # Default: Swin → mgr; classic CNN → m (single-channel). Explicit key wins.
        _cm = state.get("channel_mode", None)
        if _cm is None or str(_cm) in ("None", ""):
            self.channel_mode = "mgr" if scorer_type == "swin" else "m"
        else:
            self.channel_mode = str(_cm)
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
        # else keep constructor default (True for classic paper M+FR)

        in_ch = None
        self.bilinear = None
        if state.get("alignment_kind") == "low_rank_bilinear":
            from lib.bilinear_metric import LowRankBilinear

            rank = int(state.get("alignment_rank", state["alignment_hidden"]))
            self.bilinear = LowRankBilinear(
                dim=int(state["alignment_dim"]), rank=rank
            ).to(device)
            self.bilinear.load_state_dict(state["alignment_state_dict"])
            self.bilinear.eval()
        self.shuffle_g = bool(state.get("shuffle_g", False))
        self.shuffle_g_seed = int(state.get("shuffle_g_seed", 15))
        self.random_r = bool(state.get("random_r", False))
        self.random_r_seed = int(state.get("random_r_seed", 15))
        self.pad_to_mgr = bool(state.get("pad_to_mgr", False))
        self.pad_to_mrr = bool(state.get("pad_to_mrr", False))
        hidden_dim = int(state.get("hidden_dim", hidden_dim))

        if self.use_gr_channels or self.pad_to_mgr or self.pad_to_mrr:
            self.use_gr_channels = True
            in_ch = num_input_channels(
                self.channel_mode,
                pad_to_mgr=self.pad_to_mgr,
                pad_to_mrr=self.pad_to_mrr,
            )

        # Resolve dense feature layers (multi-layer DCSM or single-layer legacy).
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
        if self.channel_mode == "ma":
            in_ch = num_input_channels(self.channel_mode)
        elif in_ch is None and not self.use_gr_channels and n_dense > 1:
            in_ch = n_dense

        patch_grid = int(state.get("patch_grid", 14))
        img_seq = patch_grid * patch_grid + 1
        import src.util.dcsm_channels as dch

        dch.PATCH_GRID = patch_grid
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
        self.task_net.eval()

        ext = state.get("external_logic_token_embeddings")
        if ext is None:
            ext = nn.Embedding(len(self.special_words), img_seq).to(device)
            with torch.no_grad():
                ext.weight.copy_((F.normalize(ext.weight, p=2, dim=1) + 1) / 3)
        self.ext_emb = ext.to(device)

    def _net_input(
        self,
        cossim: torch.Tensor,
        texts: Sequence[str],
        aligned: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.channel_mode == "ma" or self.use_gr_channels:
            return stack_dcsm_channels(
                cossim,
                texts,
                channel_mode=self.channel_mode,
                shuffle_g=self.shuffle_g,
                shuffle_g_seed=self.shuffle_g_seed,
                random_r=self.random_r,
                random_r_seed=self.random_r_seed,
                pad_to_mgr=self.pad_to_mgr,
                pad_to_mrr=self.pad_to_mrr,
                aligned_cossim=aligned,
            )
        return cossim

    def _score_map(
        self,
        cossim: torch.Tensor,
        texts: Sequence[str],
        aligned: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.task_net(self._net_input(cossim, texts, aligned))

    def _format_texts(self, captions: Sequence[str]) -> List[str]:
        if self.use_gr_channels:
            return [self.text_prefix + c for c in captions]
        return [
            self.text_prefix
            + c.replace(" behind ", " above ")
            .replace(" in front of ", " below ")
            .replace(" bottom", " below")
            .replace(" top", " above")
            for c in captions
        ]

    @torch.no_grad()
    def _dense_feats(
        self,
        images: Sequence[Image.Image],
        texts: Sequence[str],
    ):
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

    @torch.no_grad()
    def encode_images(self, images: Sequence[Image.Image], batch_size: int = 32) -> torch.Tensor:
        if len(self.dense_feat_layers) > 1:
            raise NotImplementedError(
                "encode_images not supported for multi-layer DCSM; use score_image_captions"
            )
        chunks = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            v_h, _ = self._dense_feats(chunk, ["."] * len(chunk))
            chunks.append(v_h)
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def encode_texts(self, captions: Sequence[str], batch_size: int = 32) -> torch.Tensor:
        if len(self.dense_feat_layers) > 1:
            raise NotImplementedError(
                "encode_texts not supported for multi-layer DCSM; use score_image_captions"
            )
        texts = self._format_texts(captions)
        # need one dummy image for API; text path ignores image content when only encoding text
        dummy = Image.new("RGB", (224, 224))
        chunks = []
        for i in range(0, len(texts), batch_size):
            _, t_h = self._dense_feats([dummy], texts[i : i + batch_size])
            chunks.append(t_h)
        return torch.cat(chunks, dim=0)

    def _apply_functional_rows(
        self, cossim: torch.Tensor, texts: Sequence[str], n_img: int
    ) -> torch.Tensor:
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
    def retrieval_matrix(
        self,
        img_feats: torch.Tensor,
        txt_feats: torch.Tensor,
        formatted_texts: Sequence[str],
        *,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Full cross-modal score matrix [n_img, n_txt] using pre-encoded feats."""
        n_img, n_txt = img_feats.shape[0], txt_feats.shape[0]
        scores = torch.zeros(n_img, n_txt, device="cpu")
        for j in range(n_txt):
            t_h = txt_feats[j : j + 1].unsqueeze(0)
            for i in range(0, n_img, batch_size):
                v_h = img_feats[i : i + batch_size].unsqueeze(1)
                cossim = torch.einsum("bqie,lpte->bpit", v_h, t_h)
                cossim = self._apply_functional_rows(
                    cossim, [formatted_texts[j]], v_h.shape[0]
                )
                cossim = standardize(cossim)
                aligned = None
                if self.bilinear is not None:
                    aligned = standardize(
                        self.bilinear.pairwise_map(
                            img_feats[i : i + batch_size], txt_feats[j : j + 1]
                        )
                    )
                for bi in range(cossim.shape[0]):
                    mat = cossim[bi : bi + 1, 0:1]
                    mat_a = None if aligned is None else aligned[bi : bi + 1, 0:1]
                    scores[i + bi, j] = self._score_map(
                        mat, [formatted_texts[j]], mat_a
                    ).reshape(()).cpu()
        return scores

    @torch.no_grad()
    def _dcsm_matrix(
        self,
        images: Sequence[Image.Image],
        captions: Sequence[str],
    ) -> torch.Tensor:
        """Return CNN scores [n_images, n_captions]."""
        texts = self._format_texts(captions)
        n_img, n_txt = len(images), len(texts)
        feats = self._dense_feats(images, texts)

        aligned = None
        if len(self.dense_feat_layers) == 1:
            v_h, t_h = feats
            cossim = torch.einsum(
                "bqie,lpte->bpit", v_h.unsqueeze(1), t_h.unsqueeze(0)
            )
            cossim = self._apply_functional_rows(cossim, texts, n_img)
            cossim = standardize(cossim)
            if self.bilinear is not None:
                aligned = standardize(self.bilinear.pairwise_map(v_h, t_h))
        else:
            v_list, t_list = feats
            maps = []
            for v_h, t_h in zip(v_list, t_list):
                cm = torch.einsum(
                    "bqie,lpte->bpit", v_h.unsqueeze(1), t_h.unsqueeze(0)
                )
                cm = self._apply_functional_rows(cm, texts, n_img)
                maps.append(standardize(cm))
            cossim = torch.stack(maps, dim=2)  # (B1,B2,L,H,W)
            if self.channel_mode == "ma":
                raise ValueError("ma requires a single dense layer")

        scores = []
        for i in range(n_img):
            row = []
            for j in range(n_txt):
                mat = cossim[i : i + 1, j : j + 1]
                mat_a = None if aligned is None else aligned[i : i + 1, j : j + 1]
                row.append(self._score_map(mat, [texts[j]], mat_a).reshape(()))
            scores.append(torch.stack(row))
        out = torch.stack(scores)
        if out.dim() == 1:
            return out.unsqueeze(0)
        return out

    @torch.no_grad()
    def score_image_captions(
        self,
        image: Union[Image.Image, Sequence[Image.Image]],
        captions: Sequence[str],
    ) -> torch.Tensor:
        images = image if isinstance(image, (list, tuple)) else [image]
        return self._dcsm_matrix(images, captions).reshape(len(images), len(captions))

    @torch.no_grad()
    def score_pair(self, image: Image.Image, caption: str) -> float:
        return float(self.score_image_captions(image, [caption])[0].item())
