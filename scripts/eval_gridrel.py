#!/usr/bin/env python3
"""GridRel full evaluation: Winoground / WhatsUp / SugarCrepe / ARO / controlled_pairs.

Protocols (original papers / realign-clip):
  - Winoground: 2×2 group / text / image accuracy (all examples)
  - WhatsUp: 6-split 4-way MCQ (gold=0), macro accuracy
  - SugarCrepe: 7-subset binary choice, macro accuracy
  - controlled_pairs: test split full-matrix retrieval R@1 (T→I / I→T)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

GRIDREL = Path(__file__).resolve().parents[1]
REPO = GRIDREL  # config paths resolve relative to GridRel
REALIGN = Path("/root/realign-clip")
sys.path.insert(0, str(GRIDREL))
sys.path.insert(0, str(REALIGN / "src"))

from src.evaluating.dcsm_scorer import (  # noqa: E402
    ClipCosineScorer,
    DCSMScorer,
    DCSM_EVAL_TEXT_PREFIX,
    load_clip,
)
from relalign_clip.data.datasets import get_controlled_records_for_split  # noqa: E402
from relalign_clip.evaluation.composition_benchmarks import (  # noqa: E402
    ARO_TASKS,
    SUGARCREPE_SUBSETS,
    WHATSUP_SPLITS,
    CompositionBenchmarkEvaluator,
    _controlled_pair_set_accuracy,
    _crop_bbox,
    _infer_controlled_relation,
    _iter_parquet_rows,
    _parquet_num_rows,
    _pil_from_hf_image,
    _sample_metadata,
    pre_caption,
)
from relalign_clip.evaluation.retrieval import compute_recall_from_score_matrix  # noqa: E402

# Winoground helpers (inline to avoid extra deps)
import json as _json


class PairScorer(Protocol):
    def score_pair(self, image: Image.Image, caption: str) -> float: ...
    def score_image_captions(self, image, captions: Sequence[str]) -> torch.Tensor: ...


class _PrefixOverrideScorer:
    """Temporarily override DCSM text prefix (e.g. WhatsUp uses '' vs train-aligned prefix)."""

    def __init__(self, inner: PairScorer, text_prefix: str) -> None:
        self.inner = inner
        self.text_prefix = text_prefix

    def score_image_captions(self, image, captions: Sequence[str]) -> torch.Tensor:
        if hasattr(self.inner, "text_prefix"):
            old = self.inner.text_prefix
            self.inner.text_prefix = self.text_prefix
            try:
                return self.inner.score_image_captions(image, captions)
            finally:
                self.inner.text_prefix = old
        return self.inner.score_image_captions(image, captions)

    def score_pair(self, image: Image.Image, caption: str) -> float:
        return float(self.score_image_captions(image, [caption]).reshape(-1)[0].item())


def load_winoground_records(data_dir: Path) -> List[Dict[str, Any]]:
    examples = data_dir / "data" / "examples.jsonl"
    images_dir = data_dir / "data" / "images"
    if not images_dir.exists():
        images_dir = data_dir / "images"
    records = []
    for line in examples.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = _json.loads(line)
        records.append({
            "caption_0": row["caption_0"],
            "caption_1": row["caption_1"],
            "image_0": str(images_dir / f"{row['image_0']}.png"),
            "image_1": str(images_dir / f"{row['image_1']}.png"),
            "tag": row.get("tag", ""),
        })
    return records


def eval_winoground(scorer: PairScorer, records: List[Dict[str, Any]]) -> Dict[str, float]:
    g_ok = t_ok = i_ok = 0
    for rec in tqdm(records, desc="winoground", leave=False):
        im0 = Image.open(rec["image_0"]).convert("RGB")
        im1 = Image.open(rec["image_1"]).convert("RGB")
        s00 = scorer.score_pair(im0, rec["caption_0"])
        s01 = scorer.score_pair(im0, rec["caption_1"])
        s10 = scorer.score_pair(im1, rec["caption_0"])
        s11 = scorer.score_pair(im1, rec["caption_1"])
        t_ok += int(s00 > s01 and s11 > s10)
        i_ok += int(s00 > s10 and s11 > s01)
        g_ok += int(s00 > s01 and s11 > s10 and s00 > s10 and s11 > s01)
    n = len(records)
    return {
        "group_accuracy": g_ok / n,
        "text_accuracy": t_ok / n,
        "image_accuracy": i_ok / n,
        "count": n,
    }


def _mcq_eval_records(
    records: List[Dict[str, Any]],
    scorer: PairScorer,
    correct_index: int = 0,
    desc: str = "mcq",
) -> Dict[str, Any]:
    correct = 0
    for rec in tqdm(records, desc=desc, leave=False):
        img = rec["image"]
        caps = rec["captions"]
        scores = scorer.score_image_captions(img, caps)
        if isinstance(scores, torch.Tensor):
            pred = int(scores.argmax().item())
        else:
            pred = int(np.argmax(scores))
        correct += int(pred == correct_index)
    n = len(records)
    return {"accuracy": correct / n if n else 0.0, "count": n}


def eval_whatsup(
    scorer: PairScorer,
    whatsup_dir: Path,
    *,
    splits: Sequence[str] | None = None,
) -> Dict[str, Any]:
    data_root = whatsup_dir / "data"
    results: Dict[str, Any] = {"splits": {}, "macro_accuracy": 0.0}
    accs = []
    chosen_splits = list(splits) if splits is not None else list(WHATSUP_SPLITS)
    # WhatsUp parquet captions already include "A photo/picture of ..."; no extra prefix.
    prefix = ""

    from lib.gridrel_eval_scorer import SharedClipDCSMScorer

    def _is_dcsm_scorer(scorer: PairScorer) -> bool:
        if isinstance(scorer, (DCSMScorer, SharedClipDCSMScorer)):
            return True
        if isinstance(scorer, _PrefixOverrideScorer):
            return isinstance(scorer.inner, (DCSMScorer, SharedClipDCSMScorer))
        return False

    class _Wrapped:
        def __init__(self, inner: PairScorer) -> None:
            self.inner = inner
            self._is_dcsm = _is_dcsm_scorer(inner)

        def score_image_captions(self, image, captions):
            if self._is_dcsm:
                return self.inner.score_image_captions(image, list(captions))
            caps = [prefix + c for c in captions]
            return self.inner.score_image_captions(image, caps)

        def score_pair(self, image, caption):
            return float(self.score_image_captions(image, [caption])[0])

    ws = _Wrapped(scorer)

    for split in chosen_splits:
        total = _parquet_num_rows(data_root, split_prefix=split)
        raw = []
        for row in _iter_parquet_rows(data_root, split_prefix=split):
            img_field = row.get("image_options") or row.get("image")
            img = _pil_from_hf_image(img_field[0] if isinstance(img_field, list) else img_field)
            caps = list(row["caption_options"])
            meta = {"captions": caps}
            if split.startswith("Controlled_Images_"):
                meta["relation"] = _infer_controlled_relation(caps[0], split.split("_")[-1])
            raw.append({"image": img, "captions": caps, "caption_options": caps, **meta})

        metrics = _mcq_eval_records(raw, ws, desc=split)
        if split.startswith("Controlled_Images_"):
            subset = split.split("_")[-1]
            preds = []
            for rec in raw:
                s = ws.score_image_captions(rec["image"], rec["captions"])
                preds.append(int(s.argmax().item()))
            mask = np.array([p == 0 for p in preds])
            metrics.update(_controlled_pair_set_accuracy(raw, mask, subset))
        metrics.update(_sample_metadata(total, None, 42))
        results["splits"][split] = metrics
        accs.append(metrics["accuracy"])
    results["macro_accuracy"] = float(np.mean(accs)) if accs else 0.0
    return results


def eval_sugarcrepe(
    scorer: PairScorer,
    composition_dir: Path,
    *,
    subsets: Sequence[str] | None = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"subsets": {}, "macro_accuracy": 0.0}
    accs = []
    chosen_subsets = list(subsets) if subsets is not None else list(SUGARCREPE_SUBSETS)
    for subset in chosen_subsets:
        data_dir = composition_dir / f"SugarCrepe_{subset}" / "data"
        total = _parquet_num_rows(data_dir)
        records = []
        for row in _iter_parquet_rows(data_dir):
            pos, neg = row["tested_labels"][0], row["tested_labels"][1]
            records.append({
                "image": _pil_from_hf_image(row["image"]),
                "captions": [pos, neg],
            })
        metrics = _mcq_eval_records(records, scorer, desc=f"sugarcrepe_{subset}")
        metrics.update(_sample_metadata(total, None, 42))
        results["subsets"][subset] = metrics
        accs.append(metrics["accuracy"])
    results["macro_accuracy"] = float(np.mean(accs)) if accs else 0.0
    return results


def eval_aro(
    scorer: PairScorer,
    composition_dir: Path,
    *,
    tasks: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """ARO: 2-choice (VG Rel/Attr) or 5-choice (COCO/Flickr order), per Yuksekgonul et al."""
    root = composition_dir
    run_tasks = list(tasks) if tasks is not None else list(ARO_TASKS)
    results: Dict[str, Any] = {"tasks": {}, "macro_accuracy": 0.0}
    accs = []
    for task in run_tasks:
        data_dir = root / f"ARO-{task}" / "data"
        total = _parquet_num_rows(data_dir)
        records = []
        if task in ("Visual-Relation", "Visual-Attribution"):
            correct_index = 1
            for row in _iter_parquet_rows(data_dir):
                img = _crop_bbox(_pil_from_hf_image(row["image"]), row["bbox"])
                records.append({
                    "image": img,
                    "captions": [row["false_caption"], row["true_caption"]],
                })
        else:
            correct_index = 0
            for row in _iter_parquet_rows(data_dir):
                img = _pil_from_hf_image(row["images"])
                caps = [
                    pre_caption(row["correct_caption"]),
                    pre_caption(row["hard_text_1"]),
                    pre_caption(row["hard_text_2"]),
                    pre_caption(row["hard_text_3"]),
                    pre_caption(row["hard_text_4"]),
                ]
                records.append({"image": img, "captions": caps})
        metrics = _mcq_eval_records(records, scorer, correct_index=correct_index, desc=f"aro_{task}")
        metrics.update(_sample_metadata(total, None, 42))
        results["tasks"][task] = metrics
        accs.append(metrics["accuracy"])
    results["macro_accuracy"] = float(np.mean(accs)) if accs else 0.0
    rel = [results["tasks"][t]["accuracy"] for t in ("Visual-Relation", "Visual-Attribution") if t in results["tasks"]]
    order = [results["tasks"][t]["accuracy"] for t in ("COCO-order", "Flickr-Order") if t in results["tasks"]]
    if rel:
        results["relation_attribution_macro"] = float(np.mean(rel))
    if order:
        results["order_macro"] = float(np.mean(order))
    return results


def eval_controlled_pairs(
    scorer: PairScorer,
    data_dir: Path,
    *,
    split: str = "test",
    batch_size: int = 32,
) -> Dict[str, Any]:
    records = get_controlled_records_for_split(str(data_dir), allowed_relations=None, split=split)
    root = Path(data_dir)
    img_paths = [str(root / str(r["image"])) for r in records]
    captions = [str(r["caption"]) for r in records]
    n = len(records)
    images = [Image.open(p).convert("RGB") for p in tqdm(img_paths, desc="load_ctrl_imgs", leave=False)]

    if isinstance(scorer, DCSMScorer):
        img_feats = scorer.encode_images(images, batch_size=batch_size)
        txt_feats = scorer.encode_texts(captions, batch_size=batch_size)
        fmt_texts = scorer._format_texts(captions)
        t2i = scorer.retrieval_matrix(img_feats, txt_feats, fmt_texts, batch_size=batch_size)
        i2t = t2i.T
    else:
        t2i = torch.zeros(n, n)
        for ti in tqdm(range(n), desc="controlled T2I", leave=False):
            text = captions[ti]
            for j in range(0, n, batch_size):
                chunk = images[j : j + batch_size]
                scores = scorer.score_image_captions(chunk, [text])
                if scores.dim() == 1:
                    scores = scores.unsqueeze(0)
                t2i[ti, j : j + batch_size] = scores[:, 0].cpu()
        i2t = torch.zeros(n, n)
        for ii in tqdm(range(n), desc="controlled I2T", leave=False):
            img = images[ii]
            for j in range(0, n, batch_size):
                caps = captions[j : j + batch_size]
                scores = scorer.score_image_captions(img, caps)
                if scores.dim() == 0:
                    scores = scores.unsqueeze(0)
                i2t[ii, j : j + batch_size] = scores.cpu()

    k = [1]
    return {
        "t2i": compute_recall_from_score_matrix(t2i, k),
        "i2t": compute_recall_from_score_matrix(i2t, k),
        "count": n,
        "split": split,
    }


def _baseline_name(cfg: Dict[str, Any]) -> str:
    return str(cfg["eval"].get("baseline_name", "CLIP"))


def build_scorers(cfg: Dict[str, Any], device: str) -> Dict[str, PairScorer]:
    """Build CLIP + DCSM scorers; share one frozen CLIP across DCSM checkpoints."""
    # Local import keeps validation_chain optional for other callers.
    from lib.gridrel_eval_scorer import SharedClipDCSMScorer

    clip_path = cfg["paths"]["clip_model"]
    text_seq = int(cfg["eval"]["text_seq"])
    dcsm_prefix = str(cfg["eval"].get("dcsm_text_prefix", DCSM_EVAL_TEXT_PREFIX))
    scorers: Dict[str, PairScorer] = {}

    model, tok, proc = load_clip(clip_path, device)
    if cfg["eval"].get("include_clip", True):
        scorers[_baseline_name(cfg)] = ClipCosineScorer(model, tok, proc, device)

    layer_override = cfg["eval"].get("dense_feat_layer_override", None)
    for name, ckpt in cfg["paths"].get("dcsm_checkpoints", {}).items():
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = REPO / ckpt_path
        scorers[f"DCSM-{name.upper()}"] = SharedClipDCSMScorer(
            ckpt_path,
            model=model,
            tokenizer=tok,
            img_proc=proc,
            device=device,
            text_seq=text_seq,
            text_prefix=dcsm_prefix,
            dense_feat_layer_override=layer_override,
        )
    return scorers


def _benchmarks_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    default = ["winoground", "whatsup", "sugarcrepe"]
    return list(cfg["eval"].get("benchmarks", default))


def _model_has_benchmarks(res: Dict[str, Any], benchmarks: List[str]) -> bool:
    return all(b in res for b in benchmarks)


def _run_benchmarks(
    scorer: PairScorer,
    cfg: Dict[str, Any],
    benchmarks: List[str],
    *,
    wino_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    if "winoground" in benchmarks:
        res["winoground"] = eval_winoground(scorer, wino_records)
        print(
            f"Winoground: group={res['winoground']['group_accuracy']:.4f} "
            f"text={res['winoground']['text_accuracy']:.4f} "
            f"image={res['winoground']['image_accuracy']:.4f}"
        )
    if "whatsup" in benchmarks:
        whatsup_prefix = cfg["eval"].get("dcsm_text_prefix_whatsup", "")
        whatsup_scorer: PairScorer = (
            _PrefixOverrideScorer(scorer, whatsup_prefix)
            if whatsup_prefix != getattr(scorer, "text_prefix", None)
            else scorer
        )
        res["whatsup"] = eval_whatsup(
            whatsup_scorer,
            Path(cfg["paths"]["whatsup_dir"]),
            splits=cfg["eval"].get("whatsup_splits"),
        )
        print(f"WhatsUp macro={res['whatsup']['macro_accuracy']:.4f}")
    if "sugarcrepe" in benchmarks:
        res["sugarcrepe"] = eval_sugarcrepe(
            scorer,
            Path(cfg["paths"]["composition_dir"]),
            subsets=cfg["eval"].get("sugarcrepe_subsets"),
        )
        print(f"SugarCrepe macro={res['sugarcrepe']['macro_accuracy']:.4f}")
    if "aro" in benchmarks:
        res["aro"] = eval_aro(scorer, Path(cfg["paths"]["composition_dir"]))
        print(f"ARO macro={res['aro']['macro_accuracy']:.4f}")
    if "aro_visual_relation" in benchmarks:
        aro_vr = eval_aro(
            scorer, Path(cfg["paths"]["composition_dir"]), tasks=["Visual-Relation"]
        )
        vr = aro_vr["tasks"]["Visual-Relation"]
        res["aro_visual_relation"] = {
            "accuracy": vr["accuracy"],
            "count": vr["count"],
            "task": "Visual-Relation",
        }
        print(f"ARO Visual-Relation={res['aro_visual_relation']['accuracy']:.4f}")
    if "controlled_pairs" in benchmarks:
        res["controlled_pairs"] = eval_controlled_pairs(
            scorer,
            Path(cfg["paths"]["controlled_pairs_dir"]),
            split=cfg["eval"].get("controlled_split", "test"),
            batch_size=int(cfg["eval"].get("batch_size", 32)),
        )
        print(
            f"controlled_pairs: T2I R@1={res['controlled_pairs']['t2i']['recall@1']:.4f} "
            f"I2T R@1={res['controlled_pairs']['i2t']['recall@1']:.4f}"
        )
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume", action="store_true", help="Skip models already in metrics JSON")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cand = GRIDREL / cfg_path
        cfg_path = cand if cand.is_file() else (REPO / cfg_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    device = args.device or cfg["eval"].get("device", "cuda")
    out_dir = Path(args.output_dir or cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmarks = _benchmarks_from_cfg(cfg)
    out_json = out_dir / "metrics_standard.json"

    wino_records = load_winoground_records(Path(cfg["paths"]["winoground_dir"])) if "winoground" in benchmarks else []

    all_results: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "clip_model": cfg["paths"]["clip_model"],
        "baseline_name": _baseline_name(cfg),
        "benchmarks": benchmarks,
        "whatsup_splits": cfg["eval"].get("whatsup_splits"),
        "sugarcrepe_subsets": cfg["eval"].get("sugarcrepe_subsets"),
        "dcsm_text_prefix": cfg["eval"].get("dcsm_text_prefix", DCSM_EVAL_TEXT_PREFIX),
        "dcsm_text_prefix_whatsup": cfg["eval"].get("dcsm_text_prefix_whatsup"),
        "models": {},
    }
    if args.resume and out_json.exists():
        prev = json.loads(out_json.read_text(encoding="utf-8"))
        all_results["models"] = prev.get("models", {})

    scorers = build_scorers(cfg, device)
    for model_name, scorer in scorers.items():
        existing = all_results["models"].get(model_name, {})
        if args.resume and _model_has_benchmarks(existing, benchmarks):
            print(f"\n========== {model_name} (skipped, already done) ==========")
            continue

        print(f"\n========== {model_name} ==========")
        res = dict(existing) if existing else {}
        res.update(_run_benchmarks(scorer, cfg, benchmarks, wino_records=wino_records))
        all_results["models"][model_name] = res
        all_results["timestamp"] = datetime.now(timezone.utc).isoformat()
        out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    _write_table(out_dir / "comparison_table.md", all_results, benchmarks)
    print(f"\nSaved → {out_json}")


def _write_table(path: Path, results: Dict[str, Any], benchmarks: List[str]) -> None:
    models = list(results["models"].keys())
    baseline = str(results.get("baseline_name", "CLIP"))

    all_rows = [
        ("Winoground", "group_acc", ("winoground", "group_accuracy")),
        ("Winoground", "text_acc", ("winoground", "text_accuracy")),
        ("Winoground", "image_acc", ("winoground", "image_accuracy")),
        ("WhatsUp", "macro_acc", ("whatsup", "macro_accuracy")),
        ("SugarCrepe", "macro_acc", ("sugarcrepe", "macro_accuracy")),
        ("ARO", "macro_acc", ("aro", "macro_accuracy")),
        ("ARO", "Visual-Relation", ("aro", "tasks", "Visual-Relation", "accuracy")),
        ("ARO-VR", "accuracy", ("aro_visual_relation", "accuracy")),
        ("ARO", "Visual-Attribution", ("aro", "tasks", "Visual-Attribution", "accuracy")),
        ("ARO", "COCO-order", ("aro", "tasks", "COCO-order", "accuracy")),
        ("ARO", "Flickr-Order", ("aro", "tasks", "Flickr-Order", "accuracy")),
        ("controlled_pairs", "T2I R@1", ("controlled_pairs", "t2i", "recall@1")),
        ("controlled_pairs", "I2T R@1", ("controlled_pairs", "i2t", "recall@1")),
    ]
    bench_keys = {
        "winoground": "Winoground",
        "whatsup": "WhatsUp",
        "sugarcrepe": "SugarCrepe",
        "aro": "ARO",
        "aro_visual_relation": "ARO-VR",
        "controlled_pairs": "controlled_pairs",
    }
    active = {bench_keys[b] for b in benchmarks if b in bench_keys}
    rows = [r for r in all_rows if r[0] in active]

    others = [m for m in models if m != baseline]
    lines = [
        f"# {baseline} vs DCSM — Standard Benchmark Comparison",
        "",
        f"Generated: {results.get('timestamp', '')}",
        f"Backbone: {results.get('clip_model', '')}",
        f"Benchmarks: {', '.join(benchmarks)}",
        "",
        "## Main metrics",
        "",
        "| Benchmark | Metric | " + baseline + " | " + " | ".join(others) + " |",
        "|-----------|--------|" + "|".join(["------"] * len(models)) + "|",
    ]

    for bench, metric, keys in rows:
        vals = []
        for m in models:
            d = results["models"][m]
            v = d
            for k in keys:
                if not isinstance(v, dict) or k not in v:
                    v = None
                    break
                v = v[k]
            vals.append(f"{float(v):.3f}" if v is not None else "N/A")
        lines.append(f"| {bench} | {metric} | " + " | ".join(vals) + " |")

    sugar_subsets = list(
        results.get("sugarcrepe_subsets")
        or (SUGARCREPE_SUBSETS if "sugarcrepe" in benchmarks else [])
    )
    if sugar_subsets:
        lines.extend([
            "",
            "## SugarCrepe 子集",
            "",
            "| 子集 | " + " | ".join(models) + " |",
            "|------|" + "|".join(["------"] * len(models)) + "|",
        ])
        for subset in sugar_subsets:
            vals = []
            for m in models:
                acc = (
                    results["models"][m]
                    .get("sugarcrepe", {})
                    .get("subsets", {})
                    .get(subset, {})
                    .get("accuracy")
                )
                vals.append(f"{float(acc):.3f}" if acc is not None else "N/A")
            lines.append(f"| {subset} | " + " | ".join(vals) + " |")

    whatsup_splits = list(
        results.get("whatsup_splits")
        or (WHATSUP_SPLITS if "whatsup" in benchmarks else [])
    )
    if whatsup_splits:
        lines.extend([
            "",
            "## WhatsUp 子集",
            "",
            "| 子集 | " + " | ".join(models) + " |",
            "|------|" + "|".join(["------"] * len(models)) + "|",
        ])
        for split in whatsup_splits:
            vals = []
            for m in models:
                acc = (
                    results["models"][m]
                    .get("whatsup", {})
                    .get("splits", {})
                    .get(split, {})
                    .get("accuracy")
                )
                vals.append(f"{float(acc):.3f}" if acc is not None else "N/A")
            lines.append(f"| {split} | " + " | ".join(vals) + " |")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
