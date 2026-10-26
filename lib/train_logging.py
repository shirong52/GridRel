"""Line-buffered stdout + file tee for training logs."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


class TrainLogger:
    def __init__(self, log_path: Path | None = None) -> None:
        self._log_path = log_path
        self._file: TextIO | None = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = log_path.open("a", encoding="utf-8")

    def log(self, msg: str = "") -> None:
        line = msg if msg.endswith("\n") else msg + "\n"
        sys.stdout.write(line)
        sys.stdout.flush()
        if self._file is not None:
            self._file.write(line)
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def setup_unbuffered_io() -> None:
    """Force line-buffered stdout/stderr (needed when output is piped to a file)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True, write_through=True)
        except Exception:
            pass


def default_log_path(cfg: dict, repo_root: Path) -> Path:
    run_id = cfg.get("run_id", "train")
    seed = int(cfg.get("train", {}).get("seed", 15))
    save_rel = cfg.get("paths", {}).get("model_save_path", f"validation_chain/3D/runs/{run_id}")
    run_dir = Path(save_rel)
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir
    return run_dir.parent / f"train_{run_id}_seed{seed}.log"


def log_header(logger: TrainLogger, cfg: dict, cfg_path: Path) -> None:
    train = cfg.get("train", {})
    paths = cfg.get("paths", {})
    logger.log(f"[{datetime.now(timezone.utc).isoformat()}] training start")
    logger.log(f"Config: {cfg_path}")
    logger.log(f"Run ID: {cfg.get('run_id')}")
    logger.log(f"Data: {paths.get('data_path')}")
    logger.log(f"Datasets: {train.get('datasets')}")
    logger.log(f"Scorer: {train.get('scorer_type')} | channel_mode: {train.get('channel_mode', 'm')}")
    fr_on = not bool(train.get('no_replacements', False))
    logger.log(f"FR: {'ON' if fr_on else 'OFF'}")
    logger.log(
        f"Seed: {train.get('seed')} | epochs: {train.get('epochs')} "
        f"| lr: {train.get('learning_rate')} | bs: {train.get('batch_size')} "
        f"| dropout: {train.get('dropout')}"
    )
