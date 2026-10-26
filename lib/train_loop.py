"""GridRel training loop: topology scorer (CLS + Conv3d) + frozen LowRankBilinear M^A.

Paper path: scorer_type=conv3d, channel_mode=mgr_a (alias: gridrel), no_replacements=true.
"""

from __future__ import annotations

import gc
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizer

# Package root: /root/autodl-tmp/GridRel
CHAIN3D = Path(__file__).resolve().parents[1]
# Data/config default; override via paths.repo_root in config
REPO = Path("/root/autodl-tmp/DCSM_Ideal_CLIP")
sys.path.insert(0, str(CHAIN3D))

from lib.conv3d_scorer import (  # noqa: E402
    _RESHAPE2D_TYPES,
    build_topology_scorer,
    is_topology_scorer_type,
)
from lib.clip_dense_feats import (  # noqa: E402
    encode_dense_vt,
    encode_dense_vt_multi,
    normalize_dense_feat_layer,
    normalize_dense_feat_layers,
)
from lib.bilinear_metric import LowRankBilinear  # noqa: E402
from lib.topology import (  # noqa: E402
    dcsm_stacked_to_train_3d_stacked,
    dcsm_to_train_3d,
    dcsm_to_train_spatial,
)
from lib.train_logging import TrainLogger, default_log_path  # noqa: E402
from src.training.training_util import make_and_return_train_data, standardize  # noqa: E402
from src.util import vis  # noqa: E402
from src.util.util import augment_all_img_lists, save_train_output  # noqa: E402

SPECIAL_WORDS = [
    "above", "below", "left", "right", "small", "big",
    "not", "no", "without", "but", "absent",
]

BILINEAR_MODES = frozenset({
    "mgr_a",
    "gridrel",  # alias; normalized to mgr_a before training
    "mgr_a_perm",
    "mgr_ashuf",
    "ma_gr",
    "ma",
    "mr_a",
    "mg_a",
})


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_path(repo_root: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else repo_root / path


def _apply_functional_rows(
    cossim_mat: torch.Tensor,
    texts: list[str],
    n_img: int,
    ext_emb: nn.Embedding,
    device: torch.device,
    text_seq: int,
) -> torch.Tensor:
    new_maps = []
    for k, txt in enumerate(texts):
        sent = txt.split(" ")
        temp = cossim_mat[:, k, :, :]
        for sp_w in SPECIAL_WORDS:
            if sp_w in sent:
                idx = sent.index(sp_w)
                chosen = SPECIAL_WORDS.index(sp_w)
                mid = ext_emb(torch.tensor([chosen], device=device))
                mid = mid.unsqueeze(2).repeat(n_img, 1, 1)
                temp = torch.cat(
                    [temp[:, :, :idx], mid, temp[:, :, idx + 1 :]],
                    dim=-1,
                )
        new_maps.append(temp)
    out = torch.stack(new_maps, dim=1)
    return out[:, :, :, :text_seq]


def run_training_3d(cfg: dict[str, Any], *, logger: TrainLogger | None = None) -> None:
    paths = cfg["paths"]
    train_cfg = cfg["train"]
    repo_root = Path(paths.get("repo_root", REPO))

    if logger is None:
        logger = TrainLogger(default_log_path(cfg, repo_root))

    def log(msg: str = "") -> None:
        logger.log(msg)

    device = train_cfg.get("device", "cuda")
    seed = int(train_cfg.get("seed", 15))
    _set_seed(seed)

    data_path = str(_resolve_path(repo_root, paths["data_path"]))
    model_save = _resolve_path(repo_root, paths["model_save_path"])
    figure_save = _resolve_path(repo_root, paths["figure_save_path"])
    model_save.mkdir(parents=True, exist_ok=True)
    figure_save.mkdir(parents=True, exist_ok=True)

    batch_size = int(train_cfg["batch_size"])
    lr = float(train_cfg["learning_rate"])
    epochs = int(train_cfg["epochs"])
    dropout = float(train_cfg.get("dropout", 0.1))
    hidden_dim = int(train_cfg.get("hidden_dim", 128))
    scorer_type = str(train_cfg["scorer_type"])
    channel_mode = str(train_cfg.get("channel_mode", "m"))
    # gridrel is an alias for mgr_a (paper GridRel channel stack)
    if channel_mode == "gridrel":
        channel_mode = "mgr_a"
    no_replacements = bool(train_cfg.get("no_replacements", False))
    max_batches = train_cfg.get("max_batches")
    max_batches = int(max_batches) if max_batches is not None else None
    # Dense feature layer policy (priority):
    # 1) dense_feat_layers → multi-channel stack
    # 2) dense_feat_vision / dense_feat_text → asymmetric single-channel
    # 3) dense_feat_layer → symmetric single-channel
    dense_feat_vision = None
    dense_feat_text = None
    if train_cfg.get("dense_feat_layers") is not None:
        dense_feat_layers = normalize_dense_feat_layers(train_cfg.get("dense_feat_layers"))
        dense_feat_layer = dense_feat_layers[0]
    elif (
        train_cfg.get("dense_feat_vision") is not None
        or train_cfg.get("dense_feat_text") is not None
    ):
        base = train_cfg.get("dense_feat_layer", "last")
        dense_feat_vision = normalize_dense_feat_layer(
            train_cfg.get("dense_feat_vision", base)
        )
        dense_feat_text = normalize_dense_feat_layer(
            train_cfg.get("dense_feat_text", base)
        )
        # Single channel; legacy field records vision layer only.
        dense_feat_layers = [dense_feat_vision]
        dense_feat_layer = dense_feat_vision
    else:
        dense_feat_layers = [
            normalize_dense_feat_layer(train_cfg.get("dense_feat_layer", "last"))
        ]
        dense_feat_layer = dense_feat_layers[0]
    n_dense_channels = len(dense_feat_layers)
    train_datasets = list(train_cfg["datasets"])
    clip_model_path = paths["clip_model"]
    text_seq = 30

    if not is_topology_scorer_type(scorer_type):
        raise ValueError(
            f"GridRel train_loop only supports topology scorers "
            f"(conv3d/reshape3d family); got {scorer_type!r}"
        )

    log("Loading CLIP...")
    model = CLIPModel.from_pretrained(clip_model_path).eval().to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_path)
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_path)
    img_processor = CLIPImageProcessor.from_pretrained(clip_model_path)
    vision_cfg = model.config.vision_config
    patch_grid = int(vision_cfg.image_size) // int(vision_cfg.patch_size)
    from lib.topology import set_patch_grid
    import src.util.dcsm_channels as dch

    set_patch_grid(patch_grid)
    dch.PATCH_GRID = patch_grid
    image_seq = patch_grid * patch_grid + 1
    log(
        f"CLIP loaded. patch_grid={patch_grid} image_seq={image_seq} "
        f"image_size={int(vision_cfg.image_size)}"
    )

    bilinear_metric = None
    # Membership uses post-normalize channel_mode; BILINEAR_MODES also lists
    # "gridrel" so pre-normalize configs remain recognized if re-checked.
    if channel_mode in BILINEAR_MODES:
        adapter_path = _resolve_path(repo_root, paths["alignment_checkpoint"])
        ckpt = torch.load(adapter_path, map_location=device)
        if ckpt.get("kind") != "low_rank_bilinear":
            raise ValueError(f"expected a frozen low-rank bilinear checkpoint, got {adapter_path}")
        bilinear_metric = LowRankBilinear(dim=int(ckpt["dim"]), rank=int(ckpt["rank"])).to(device)
        bilinear_metric.load_state_dict(ckpt["state_dict"])
        bilinear_metric.eval()
        for param in bilinear_metric.parameters():
            param.requires_grad = False
        log(
            f"Frozen bilinear metric: {adapter_path} "
            f"alpha={float(bilinear_metric.alpha.detach()):.4f} "
            "(weights are not updated)"
        )

    embed_dim = int(getattr(model.config, "projection_dim", 512))
    if bilinear_metric is not None and int(bilinear_metric.dim) != embed_dim:
        raise ValueError(
            f"bilinear dim {int(bilinear_metric.dim)} != CLIP projection_dim {embed_dim}. "
            "Train a new aligner on this backbone; do not reuse the ViT-B/16 checkpoint."
        )
    fr_dim = image_seq
    ext_emb = nn.Embedding(len(SPECIAL_WORDS), fr_dim).to(device)
    with torch.no_grad():
        ext_emb.weight.copy_((F.normalize(ext_emb.weight, p=2, dim=1) + 1) / 3)

    log("Building dataloader...")
    combined_dataloader_train = make_and_return_train_data(
        data_path, False, batch_size, datasets=train_datasets
    )
    num_datasets = len(train_datasets)
    log(f"train loader batches: {len(combined_dataloader_train)}")

    task_network = build_topology_scorer(
        scorer_type,
        text_seq=text_seq,
        hidden_dim=hidden_dim,
        dropout=dropout,
        channel_mode=channel_mode,
    ).to(device)
    n_params = sum(p.numel() for p in task_network.parameters())
    log(
        f"Scorer params: {n_params:,} (CLS + spatial, S=S1+alpha*S2, "
        f"channel_mode={channel_mode}, type={scorer_type})"
    )
    if dense_feat_vision is not None or dense_feat_text is not None:
        log(
            f"DCSM asymmetric layers: vision={dense_feat_vision}, "
            f"text={dense_feat_text} (single-channel)"
        )
    else:
        log(f"DCSM dense_feat_layers: {dense_feat_layers}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(task_network.parameters(), lr=lr)

    train_loss_list: list[float] = []
    train_acc_list: list[float] = []
    train_loss_dict = {f"dataset_{i}": [] for i in range(num_datasets)}
    train_acc_dict = {f"dataset_{i}": [] for i in range(num_datasets)}

    run_id = cfg.get("run_id", scorer_type)
    ckpt_tag = f"{run_id}_"
    start_epoch = 0
    resume_from = train_cfg.get("resume_from")
    if resume_from:
        resume_path = _resolve_path(repo_root, resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume_from not found: {resume_path}")
        log(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        task_network.load_state_dict(ckpt["model_state_dict"], strict=True)
        if ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("external_logic_token_embeddings") is not None:
            ext_emb.load_state_dict(
                ckpt["external_logic_token_embeddings"].state_dict()
                if hasattr(ckpt["external_logic_token_embeddings"], "state_dict")
                else ckpt["external_logic_token_embeddings"]
            )
        train_loss_list = list(ckpt.get("train_loss") or [])
        # Acc history is not stored in ckpt; pad with zeros to keep plot length aligned.
        train_acc_list = [0.0] * len(train_loss_list)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        if start_epoch >= epochs:
            log(
                f"Resume epoch {start_epoch} >= configured epochs={epochs}; nothing to train."
            )
            logger.close()
            return
        log(
            f"Loaded weights from ep{int(ckpt.get('epoch', -1))}; "
            f"continuing epochs [{start_epoch}, {epochs})"
        )

    for epoch in range(start_epoch, epochs):
        task_network.train()
        log("-----------------")
        log(f"Starting epoch {epoch} | {run_id}")
        log("-----------------")

        running_loss = 0.0
        running_correct = 0.0
        running_samplect = 0
        running_s2_std = 0.0
        s2_std_count = 0
        straggler = 0
        batch_idx = 0

        for batch in combined_dataloader_train:
            optimizer.zero_grad()
            if len(batch["dataset_0"]) < batch_size:
                straggler = len(batch["dataset_0"])
                continue

            for dataset_key, data in batch.items():
                train_loss_dict[dataset_key].append(0.0)
                train_acc_dict[dataset_key].append(0.0)

                for version in (0, 1):
                    images = [
                        data[qq][version]["image_options"] for qq in range(batch_size)
                    ] + [
                        data[qq][version]["opposite_image_options"]
                        for qq in range(batch_size)
                    ]
                    allowed_crops = [
                        transforms.RandomResizedCrop(224, (0.96, 1.0))
                        for _ in range(len(images))
                    ]
                    images = augment_all_img_lists(images, allowed_crops)

                    texts = [
                        data[qq][version]["caption_options"] for qq in range(batch_size)
                    ] + [
                        data[qq][version]["opposite_caption_options"]
                        for qq in range(batch_size)
                    ]

                    vision_inputs = img_processor(images=images, return_tensors="pt").to(device)
                    text_tokens = tokenizer(
                        text=texts,
                        padding="max_length",
                        max_length=text_seq,
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)

                    # DCSM cosine: single- or multi-layer patch/token features.
                    # Multi-layer: each map is FR'd + standardized, then stacked
                    # on channel dim -> (B1,B2,L,H,W). Topology scorers require L=1.
                    aligned_cossim = None
                    if n_dense_channels == 1:
                        # CLIP is frozen. ViT-L/14-336 keeps a 32-image graph that
                        # fills a 24GB GPU before the scorer runs.
                        with torch.no_grad():
                            if dense_feat_vision is not None or dense_feat_text is not None:
                                image_features, text_features = encode_dense_vt(
                                    model,
                                    vision_inputs,
                                    text_tokens,
                                    vision_layer=dense_feat_vision,
                                    text_layer=dense_feat_text,
                                )
                            else:
                                image_features, text_features = encode_dense_vt(
                                    model,
                                    vision_inputs,
                                    text_tokens,
                                    layer=dense_feat_layers[0],
                                )
                        image_features = image_features.unsqueeze(1)
                        text_features = text_features.unsqueeze(0)
                        cossim_mat = torch.einsum(
                            "bqie,lpte->bpit", image_features, text_features
                        ).to(device)
                        if not no_replacements:
                            cossim_mat = _apply_functional_rows(
                                cossim_mat,
                                texts,
                                len(images),
                                ext_emb,
                                device,
                                text_seq,
                            )
                        cossim_mat = standardize(cossim_mat)
                        if bilinear_metric is not None:
                            with torch.no_grad():
                                aligned_cossim = bilinear_metric.pairwise_map(
                                    image_features.squeeze(1),
                                    text_features.squeeze(0),
                                )
                            aligned_cossim = standardize(aligned_cossim.detach())
                    else:
                        with torch.no_grad():
                            v_list, t_list = encode_dense_vt_multi(
                                model,
                                vision_inputs,
                                text_tokens,
                                layers=dense_feat_layers,
                            )
                        maps = []
                        for v_h, t_h in zip(v_list, t_list):
                            cm = torch.einsum(
                                "bqie,lpte->bpit",
                                v_h.unsqueeze(1),
                                t_h.unsqueeze(0),
                            ).to(device)
                            if not no_replacements:
                                cm = _apply_functional_rows(
                                    cm,
                                    texts,
                                    len(images),
                                    ext_emb,
                                    device,
                                    text_seq,
                                )
                            maps.append(standardize(cm))
                        # (B1,B2,L,H,W)
                        cossim_mat = torch.stack(maps, dim=2)

                    shuffle_bs1 = torch.randperm(cossim_mat.shape[0])
                    shuffle_bs2 = torch.randperm(cossim_mat.shape[1])
                    cossim_mat = cossim_mat[shuffle_bs1][:, shuffle_bs2]
                    if aligned_cossim is not None:
                        aligned_cossim = aligned_cossim[shuffle_bs1][:, shuffle_bs2]
                    col_texts = [texts[int(j)] for j in shuffle_bs2.tolist()]

                    if n_dense_channels > 1:
                        raise ValueError(
                            "multi-layer DCSM not supported for topology scorers yet"
                        )
                    task_output = task_network(
                        cossim_mat, col_texts, aligned_cossim=aligned_cossim
                    ).squeeze()
                    quicklabels = torch.eye(task_output.shape[0], device=device)
                    quicklabels = quicklabels[shuffle_bs1][:, shuffle_bs2]

                    loss = criterion(task_output, quicklabels) + criterion(
                        task_output.t(), quicklabels.t()
                    )

                    running_loss += loss.item()
                    train_loss_dict[dataset_key][-1] += loss.item()
                    loss.backward()

                    with torch.no_grad():
                        num_correct = (
                            torch.sum(
                                torch.argmax(task_output, dim=0)
                                == torch.argmax(quicklabels, dim=0)
                            )
                            + torch.sum(
                                torch.argmax(task_output, dim=1)
                                == torch.argmax(quicklabels, dim=1)
                            )
                        ).item() / 2
                        running_correct += num_correct
                        running_samplect += task_output.shape[0]
                        train_acc_dict[dataset_key][-1] += (
                            num_correct / task_output.shape[0]
                        )
                        if channel_mode != "m":
                            from src.util.dcsm_channels import stack_dcsm_channels

                            stacked = stack_dcsm_channels(
                                cossim_mat,
                                col_texts,
                                channel_mode=channel_mode,
                                aligned_cossim=aligned_cossim,
                            )
                            spatial_in = dcsm_stacked_to_train_3d_stacked(stacked)
                        elif scorer_type in _RESHAPE2D_TYPES:
                            spatial_in = dcsm_to_train_spatial(cossim_mat)
                        else:
                            spatial_in = dcsm_to_train_3d(cossim_mat)
                        s2 = task_network.spatial_branch(spatial_in)
                        running_s2_std += s2.std().item()
                        s2_std_count += 1
                        log(f"Training Accuracy: {num_correct / task_output.shape[0]:.4f}")
                        log("----------")

            for param in task_network.parameters():
                if param.grad is not None:
                    param.grad.data /= num_datasets
            torch.nn.utils.clip_grad_norm_(task_network.parameters(), 1.0)
            optimizer.step()

            batch_idx += 1
            log("----------")
            log(f"Training Loss (batch {batch_idx}): {loss.item():.6f}")
            log("----------")

            del cossim_mat, task_output, quicklabels, loss
            gc.collect()
            torch.cuda.empty_cache()

            if max_batches is not None and batch_idx >= max_batches:
                log(f"Stopping early after {batch_idx} batches (train.max_batches)")
                break

        steps = batch_idx if batch_idx > 0 else 1
        if max_batches is None:
            steps = max(len(combined_dataloader_train) - int(straggler == 0), 1)
        train_loss_list.append(running_loss / (2 * steps))
        train_acc_list.append(running_correct / max(running_samplect, 1))

        savename_plot = os.path.join(
            str(figure_save),
            f"{ckpt_tag}ep{epoch}_bs{batch_size}_lr{lr}_drop{dropout}_seed{seed}.png",
        )
        vis.plot_training_justlist(
            train_loss_list, [], train_acc_list, [],
            total_epochs=epoch, losstype="CE", savename=savename_plot,
        )

        ckpt_name = os.path.join(
            str(model_save),
            f"{ckpt_tag}ep{epoch}_bs{batch_size}_lr{lr}_drop{dropout}_seed{seed}.pt",
        )
        save_train_output(
            model_state=task_network.state_dict(),
            optimizer_state=optimizer.state_dict(),
            clip_model=model.state_dict() if patch_grid == 14 else None,
            tokenizer=tokenizer,
            epoch=epoch,
            train_loss=train_loss_list,
            valid_loss=[],
            external_logic_token_embeddings=ext_emb,
            grads={},
            scorer_type=scorer_type,
            channel_mode=channel_mode,
            hidden_dim=hidden_dim,
            no_replacements=no_replacements,
            use_functional_rows=not no_replacements,
            learned_lookup=SPECIAL_WORDS,
            savename=ckpt_name,
            d_model=None,
            n_heads=None,
            num_relation_classes=None,
            dense_feat_layer=dense_feat_layer,
            dense_feat_layers=(
                list(dense_feat_layers) if n_dense_channels > 1 else None
            ),
            dense_feat_vision=dense_feat_vision,
            dense_feat_text=dense_feat_text,
            alignment_state_dict=(
                {k: v.detach().cpu() for k, v in bilinear_metric.state_dict().items()}
                if bilinear_metric is not None
                else None
            ),
            alignment_alpha=(
                float(bilinear_metric.alpha.detach()) if bilinear_metric is not None else None
            ),
            alignment_dim=(int(bilinear_metric.dim) if bilinear_metric is not None else None),
            alignment_hidden=(int(bilinear_metric.rank) if bilinear_metric is not None else None),
            alignment_kind=("low_rank_bilinear" if bilinear_metric is not None else None),
            alignment_rank=(int(bilinear_metric.rank) if bilinear_metric is not None else None),
            patch_grid=patch_grid,
        )
        log(f"Saved checkpoint: {ckpt_name}")
        log(f"Saved figure: {savename_plot}")
        epoch_s2_std = running_s2_std / max(s2_std_count, 1)
        if hasattr(task_network, "alpha"):
            alpha_val = float(task_network.alpha.detach())
            log(
                f"Epoch {epoch} train_loss={train_loss_list[-1]:.4f} "
                f"train_acc={train_acc_list[-1]:.4f} s2_std={epoch_s2_std:.6f} alpha={alpha_val:.4f}"
            )
        else:
            log(
                f"Epoch {epoch} train_loss={train_loss_list[-1]:.4f} "
                f"train_acc={train_acc_list[-1]:.4f}"
            )

    logger.close()


run_training_gridrel = run_training_3d
