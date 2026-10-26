#!/usr/bin/env python3
"""Stage 2: train the GridRel dual-branch scorer (CLS + Conv3d on 12-channel volume)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

os.environ.setdefault("PYTHONUNBUFFERED", "1")

GRIDREL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GRIDREL))

from lib.train_logging import TrainLogger, default_log_path, log_header, setup_unbuffered_io  # noqa: E402
from lib.train_loop import run_training_gridrel  # noqa: E402

setup_unbuffered_io()


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for item in overrides:
        key, _, val = item.partition("=")
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        if val.lower() in ("true", "false"):
            node[parts[-1]] = val.lower() == "true"
        else:
            try:
                node[parts[-1]] = int(val)
            except ValueError:
                try:
                    node[parts[-1]] = float(val)
                except ValueError:
                    node[parts[-1]] = val
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(GRIDREL / "configs" / "train_gridrel.yaml"),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="dotlist override, e.g. train.epochs=1",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cand = GRIDREL / cfg_path
        cfg_path = cand if cand.is_file() else cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if args.override:
        cfg = _apply_overrides(cfg, args.override)

    repo_root = Path(cfg.get("paths", {}).get("repo_root", "/root/autodl-tmp/DCSM_Ideal_CLIP"))
    # Log under GridRel run dir when model_save_path is absolute under GridRel.
    log_repo = GRIDREL
    log_path = default_log_path(cfg, log_repo)
    logger = TrainLogger(log_path)
    log_header(logger, cfg, cfg_path)
    logger.log(f"Log file: {log_path}")
    logger.log(f"Data repo_root: {repo_root}")

    try:
        run_training_gridrel(cfg, logger=logger)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
