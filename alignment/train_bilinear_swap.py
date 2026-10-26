"""Train a shared low-rank metric with a same-image swap hinge.

Patches and tokens stay frozen. The score is
s(v, t) = v·t + alpha*(A^T v)·(B^T t), with alpha starting at 0.
A swap pair is two objects in one caption.
The loss asks token A to score higher on A's exclusive patches than on
B's, and the same the other way. It does not ask the box to beat the
background.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

GRIDREL = Path(__file__).resolve().parents[1]
ROOT = GRIDREL
sys.path.insert(0, str(GRIDREL))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from coco_region_utils import (  # noqa: E402
    canvas_box,
    find_phrase,
    git_hash,
    load_config,
    patch_inside_mask,
    token_indices_for_span,
)
from lib.bilinear_metric import LowRankBilinear, swap_loss  # noqa: E402
from lib.clip_dense_feats import encode_dense_vt  # noqa: E402


def _exclusive_pair(matched: list[tuple], min_exclusive: int):
    """First name-sorted pair whose boxes do not cover the same patches."""
    for i in range(len(matched)):
        for j in range(i + 1, len(matched)):
            left, right = matched[i], matched[j]
            only_a = left[0]["mask"] & ~right[0]["mask"]
            only_b = right[0]["mask"] & ~left[0]["mask"]
            if int(only_a.sum()) < min_exclusive or int(only_b.sum()) < min_exclusive:
                continue
            return left, right, only_a, only_b
    return None


def iter_swap_pairs(
    instances: dict,
    captions: dict,
    cfg_data: dict,
    image_dir: Path,
    grid: int = 14,
    patch: int = 16,
    canvas: int = 224,
):
    """One pair per image: two objects named once in the same caption."""
    names = {c["id"]: c["name"] for c in instances["categories"]}
    meta = {im["id"]: im for im in instances["images"]}
    by_image = defaultdict(list)
    min_area = float(cfg_data["min_box_area"])
    for ann in instances["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        if float(ann.get("area", w * h)) < min_area or w * h < min_area:
            continue
        by_image[ann["image_id"]].append(ann)
    caps = defaultdict(list)
    for ann in captions["annotations"]:
        caps[ann["image_id"]].append(ann["caption"])

    min_pos = int(cfg_data["min_pos_patches"])
    min_neg = int(cfg_data["min_neg_patches"])
    min_exclusive = int(cfg_data["min_exclusive_patches"])
    for image_id, anns in by_image.items():
        info = meta.get(image_id)
        if info is None:
            continue
        path = image_dir / info["file_name"]
        if not path.is_file():
            continue
        grouped = defaultdict(list)
        for ann in anns:
            grouped[ann["category_id"]].append(ann)
        objects = []
        for cat_id, group in grouped.items():
            if len(group) != 1:
                continue
            box = canvas_box(group[0]["bbox"], info["width"], info["height"], canvas, canvas)
            if box is None:
                continue
            mask = patch_inside_mask(box, grid, patch)
            n_pos = int(mask.sum())
            n_neg = int(mask.size - n_pos)
            if n_pos < min_pos or n_neg < min_neg:
                continue
            objects.append(
                {
                    "name": names[cat_id],
                    "category_id": int(cat_id),
                    "bbox_xywh": [float(v) for v in group[0]["bbox"]],
                    "canvas_box": [float(v) for v in box],
                    "mask": mask,
                }
            )
        if len(objects) < 2:
            continue
        best = None
        for caption in caps.get(image_id, []):
            matched = []
            for obj in objects:
                found = find_phrase(caption, obj["name"])
                if found is None:
                    continue
                start, end, kind = found
                matched.append((obj, start, end, kind))
            if len(matched) < 2:
                continue
            matched.sort(key=lambda item: item[0]["name"])
            chosen = _exclusive_pair(matched, min_exclusive)
            if chosen is None:
                continue
            key = (-len(matched), len(caption), caption)
            if best is None or key < best[0]:
                best = (key, caption, chosen)
        if best is None:
            continue
        _, caption, (left, right, only_a, only_b) = best
        yield {
            "image_id": int(image_id),
            "image_path": str(path),
            "caption": caption,
            "category_a": left[0]["name"],
            "category_b": right[0]["name"],
            "char_start_a": int(left[1]),
            "char_end_a": int(left[2]),
            "char_start_b": int(right[1]),
            "char_end_b": int(right[2]),
            "match_form_a": left[3],
            "match_form_b": right[3],
            "mask_a": only_a,
            "mask_b": only_b,
            "mask_a_full": left[0]["mask"],
            "mask_b_full": right[0]["mask"],
        }


def attach_swap_tokens(pairs: list[dict], tokenizer, text_seq: int) -> tuple[list[dict], int]:
    kept = []
    dropped = 0
    for pair in pairs:
        encoded = tokenizer(
            pair["caption"],
            padding="max_length",
            max_length=text_seq,
            truncation=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        special = [bool(x) for x in encoded["special_tokens_mask"]]
        hits_a = token_indices_for_span(
            encoded["offset_mapping"], special, pair["char_start_a"], pair["char_end_a"]
        )
        hits_b = token_indices_for_span(
            encoded["offset_mapping"], special, pair["char_start_b"], pair["char_end_b"]
        )
        if not hits_a or not hits_b or set(hits_a) & set(hits_b):
            dropped += 1
            continue
        row = dict(pair)
        row["token_indices_a"] = hits_a
        row["token_indices_b"] = hits_b
        kept.append(row)
    return kept, dropped


def sample_swap_images(pairs: list[dict], n_images: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    ids = [pair["image_id"] for pair in pairs]
    rng.shuffle(ids)
    chosen = set(ids[:n_images])
    return [pair for pair in pairs if pair["image_id"] in chosen]


def split_train_select(pairs: list[dict], n_select: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    ids = [pair["image_id"] for pair in pairs]
    rng.shuffle(ids)
    select_ids = set(ids[:n_select])
    select, train = [], []
    for pair in pairs:
        if pair["image_id"] in select_ids:
            select.append(pair)
        else:
            train.append(pair)
    return train, select


def encode_swaps(pairs, model, tokenizer, img_proc, cfg, device: str) -> list[dict]:
    text_seq = int(cfg["model"]["text_seq"])
    layer = cfg["model"]["dense_feat_layer"]
    batch_size = int(cfg["model"]["clip_batch_size"])
    items = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        from PIL import Image

        images = [Image.open(p["image_path"]).convert("RGB") for p in batch]
        vision = img_proc(images=images, return_tensors="pt").to(device)
        text = tokenizer(
            [p["caption"] for p in batch],
            padding="max_length",
            max_length=text_seq,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        v_h, t_h = encode_dense_vt(model, vision, text, layer=layer)
        for i, pair in enumerate(batch):
            n_patch = int(v_h.shape[1] - 1)
            if int(pair["mask_a"].shape[0]) != n_patch:
                raise ValueError(
                    f"mask length {pair['mask_a'].shape[0]} != CLIP patches {n_patch}"
                )
            items.append(
                {
                    "image_id": pair["image_id"],
                    "category_a": pair["category_a"],
                    "category_b": pair["category_b"],
                    "patches": v_h[i, 1:].detach().float().cpu(),
                    "tokens": t_h[i].detach().float().cpu(),
                    "index_a": torch.tensor(pair["token_indices_a"], dtype=torch.long),
                    "index_b": torch.tensor(pair["token_indices_b"], dtype=torch.long),
                    "mask_a": torch.from_numpy(pair["mask_a"]).bool(),
                    "mask_b": torch.from_numpy(pair["mask_b"]).bool(),
                }
            )
        print(f"encoded {min(start + batch_size, len(pairs))}/{len(pairs)}", flush=True)
    return items


def _gaps(cell, item) -> tuple[torch.Tensor, torch.Tensor]:
    from lib.bilinear_metric import region_mean

    q_aa = region_mean(cell, item["index_a"], item["mask_a"])
    q_ab = region_mean(cell, item["index_a"], item["mask_b"])
    q_bb = region_mean(cell, item["index_b"], item["mask_b"])
    q_ba = region_mean(cell, item["index_b"], item["mask_a"])
    return q_aa - q_ab, q_bb - q_ba


def batch_loss(metric, batch, margin: float, device: str) -> torch.Tensor:
    losses = []
    for item in batch:
        patches = item["patches"].to(device)
        tokens = item["tokens"].to(device)
        cell = metric.scores(patches, tokens)
        losses.append(
            swap_loss(
                cell,
                item["index_a"].to(device),
                item["index_b"].to(device),
                item["mask_a"].to(device),
                item["mask_b"].to(device),
                margin,
            )
        )
    return torch.stack(losses).mean()


@torch.no_grad()
def eval_split(metric, items, device: str) -> dict:
    metric.eval()
    gap_a, gap_b = [], []
    for item in items:
        cell = metric.scores(item["patches"].to(device), item["tokens"].to(device))
        moved = {
            "index_a": item["index_a"].to(device),
            "index_b": item["index_b"].to(device),
            "mask_a": item["mask_a"].to(device),
            "mask_b": item["mask_b"].to(device),
        }
        da, db = _gaps(cell, moved)
        gap_a.append(float(da))
        gap_b.append(float(db))
    both = sum(a > 0 and b > 0 for a, b in zip(gap_a, gap_b)) / len(gap_a)
    mean_a = sum(gap_a) / len(gap_a)
    mean_b = sum(gap_b) / len(gap_b)
    norm_a, norm_b, alpha = metric.factor_norm()
    return {
        "n_pairs": len(items),
        "mean_gap_a": mean_a,
        "mean_gap_b": mean_b,
        "mean_gap": 0.5 * (mean_a + mean_b),
        "frac_both_correct": both,
        "frac_a_correct": sum(a > 0 for a in gap_a) / len(gap_a),
        "frac_b_correct": sum(b > 0 for b in gap_b) / len(gap_b),
        "factor_norm_a": norm_a,
        "factor_norm_b": norm_b,
        "alpha": alpha,
    }


def load_annotations(coco: Path, split: str) -> tuple[dict, dict]:
    instances = json.loads((coco / "annotations" / f"instances_{split}.json").read_text())
    captions = json.loads((coco / "annotations" / f"captions_{split}.json").read_text())
    return instances, captions


def collect_split(coco: Path, split: str, cfg) -> list[dict]:
    instances, captions = load_annotations(coco, split)
    model_cfg = cfg.get("model", {})
    pairs = list(
        iter_swap_pairs(
            instances,
            captions,
            cfg["data"],
            coco / split,
            grid=int(model_cfg.get("patch_grid", 14)),
            patch=int(model_cfg.get("patch_size", 16)),
            canvas=int(model_cfg.get("image_size", 224)),
        )
    )
    del instances, captions
    gc.collect()
    return pairs


def write_notes(path: Path, summary: dict) -> None:
    lines = [
        "# Result memo",
        "",
        f"Run ID: {summary['run_id']}",
        "Question: 共享低秩度量能否让同一句里的两个物体词各自更偏向自己的区域？",
        "Controlled change: 分数从 v·t 换成 v·t + alpha*(A^T v)·(B^T t)，alpha 从 0 开始。损失只有同一张图上的 swap hinge。patch 不改。",
        (
            f"Dataset/split: COCO，train images={summary['n_train_images']}，"
            f"select images={summary['n_select_images']}，val images={summary['n_val_images']}。"
            "两个物体必须出现在同一条 caption 里，区域是彼此不重叠的 patch。"
        ),
        "Baseline: 同一批配对上 A=B=0，也就是原始点积。",
        f"Result: {summary['reading']}",
        f"Diagnostics: {json.dumps(summary['select_history'], ensure_ascii=False)}",
        "What this supports: 只说明这批 COCO 双物体配对上，低秩度量有没有拉开交叉偏好。",
        "What this does NOT support: Reshape3D+MGR 的空间关系会因此提高。没有接 scorer。",
        f"Suspicious signs: {summary['stop_reasons'] or '无'}",
        f"Next step: {summary['next_action']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config") / "train_bilinear_swap.yaml",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    repo = Path(cfg["paths"]["repo_root"])
    coco = Path(cfg["paths"]["coco_dir"])
    out = repo / cfg["paths"]["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    device = cfg["model"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("config requests cuda but CUDA is not available")
    seed = int(cfg["data"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)

    from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast

    tokenizer = CLIPTokenizerFast.from_pretrained(cfg["paths"]["clip_model"])
    text_seq = int(cfg["model"]["text_seq"])
    print("collecting train swap pairs", flush=True)
    train_pool = collect_split(coco, cfg["data"]["train_split"], cfg)
    train_pool, dropped_train = attach_swap_tokens(train_pool, tokenizer, text_seq)
    print(f"train eligible images {len(train_pool)} dropped_tokens {dropped_train}", flush=True)
    sampled = sample_swap_images(train_pool, int(cfg["data"]["n_images"]), seed)
    del train_pool
    gc.collect()
    train_pairs, select_pairs = split_train_select(sampled, int(cfg["data"]["n_select"]), seed)
    print(
        f"train {len(train_pairs)} select {len(select_pairs)}",
        flush=True,
    )
    print("collecting val swap pairs", flush=True)
    val_pairs = collect_split(coco, cfg["data"]["val_split"], cfg)
    val_pairs, dropped_val = attach_swap_tokens(val_pairs, tokenizer, text_seq)
    print(f"val images {len(val_pairs)} dropped_tokens {dropped_val}", flush=True)

    model = CLIPModel.from_pretrained(cfg["paths"]["clip_model"]).to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    img_proc = CLIPImageProcessor.from_pretrained(cfg["paths"]["clip_model"])
    print("encoding", flush=True)
    train_items = encode_swaps(train_pairs, model, tokenizer, img_proc, cfg, device)
    select_items = encode_swaps(select_pairs, model, tokenizer, img_proc, cfg, device)
    val_items = encode_swaps(val_pairs, model, tokenizer, img_proc, cfg, device)
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    dim = train_items[0]["patches"].shape[-1]
    rank = int(cfg["model"]["rank"])
    metric = LowRankBilinear(dim=dim, rank=rank).to(device)
    frozen = eval_split(metric, select_items, device)
    print(
        f"select frozen gap={frozen['mean_gap']:.4f} "
        f"both={frozen['frac_both_correct']:.3f}",
        flush=True,
    )
    opt = torch.optim.Adam(
        metric.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    margin = float(cfg["train"]["margin"])
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    history = []
    reasons: list[str] = []
    completed = 0
    for epoch in range(epochs):
        order = list(range(len(train_items)))
        random.Random(seed + epoch).shuffle(order)
        metric.train()
        total = 0.0
        n_batches = 0
        for start in range(0, len(order), batch_size):
            batch = [train_items[i] for i in order[start : start + batch_size]]
            opt.zero_grad(set_to_none=True)
            loss = batch_loss(metric, batch, margin, device)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n_batches += 1
        completed = epoch + 1
        row = eval_split(metric, select_items, device)
        row["epoch"] = epoch
        row["train_loss"] = total / max(n_batches, 1)
        history.append(row)
        print(
            f"epoch={epoch} loss={row['train_loss']:.4f} "
            f"gap={row['mean_gap']:.4f} both={row['frac_both_correct']:.3f} "
            f"||A||={row['factor_norm_a']:.3f} ||B||={row['factor_norm_b']:.3f} "
            f"alpha={row['alpha']:.4f}",
            flush=True,
        )
        if completed >= int(cfg["train"]["min_epochs_before_gap_check"]) and row["mean_gap"] <= frozen["mean_gap"]:
            reasons.append(
                f"select swap gap {row['mean_gap']:.4f} 没有高于冻结时的 {frozen['mean_gap']:.4f}"
            )
            print("STOP " + reasons[-1], flush=True)
            break

    val_before = eval_split(LowRankBilinear(dim=dim, rank=rank).to(device), val_items, device)
    val_after = eval_split(metric, val_items, device)
    improved = (
        not reasons
        and val_after["mean_gap"] > val_before["mean_gap"]
        and val_after["frac_both_correct"] > val_before["frac_both_correct"]
    )
    if reasons:
        reading = "训练在 select 上触发中断：" + "；".join(reasons)
        next_action = "wait_for_user"
    elif improved:
        reading = (
            f"val swap gap {val_before['mean_gap']:.4f} -> {val_after['mean_gap']:.4f}，"
            f"两个词都选对的比例 {val_before['frac_both_correct']:.3f} -> {val_after['frac_both_correct']:.3f}。"
            "还没有接 Reshape3D。"
        )
        next_action = "user_decides_scorer"
    else:
        reading = (
            f"val swap gap {val_before['mean_gap']:.4f} -> {val_after['mean_gap']:.4f}，"
            f"两个词都选对的比例 {val_before['frac_both_correct']:.3f} -> {val_after['frac_both_correct']:.3f}。"
            "低秩度量没有同时提高间隔和选对比例。"
        )
        next_action = "wait_for_user"

    ckpt = out / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in metric.state_dict().items()},
            "dim": dim,
            "rank": rank,
            "kind": "low_rank_bilinear",
            "stopped": bool(reasons),
        },
        ckpt / "bilinear_metric.pt",
    )
    summary = {
        "run_id": cfg["run_id"],
        "git_hash": git_hash(repo),
        "config": str(args.config),
        "seed": seed,
        "rank": rank,
        "margin": margin,
        "n_train_images": len(train_items),
        "n_select_images": len(select_items),
        "n_val_images": len(val_items),
        "dropped_train_tokens": dropped_train,
        "dropped_val_tokens": dropped_val,
        "completed_epochs": completed,
        "stopped": bool(reasons),
        "stop_reasons": reasons,
        "passed_gates": improved,
        "select_before": frozen,
        "select_history": history,
        "val_before": val_before,
        "val_after": val_after,
        "reading": reading,
        "next_action": next_action,
    }
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_notes(out / "notes.md", summary)
    print(json.dumps({
        "stopped": summary["stopped"],
        "passed_gates": improved,
        "select_before": frozen,
        "select_history": history,
        "val_before": val_before,
        "val_after": val_after,
        "reading": reading,
        "next_action": next_action,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
