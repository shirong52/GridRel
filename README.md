# GridRel

Paper main-path implementation: multi-channel grid scoring on a frozen CLIP encoder.

```text
X = Concat(M, M^A, G, R) → reshape3d → CLS + Conv3d dual-branch scorer
```

This tree is extracted from `DCSM_Ideal_CLIP/validation_chain/3D` and keeps only GridRel. Training data and CLIP weights still point at the existing local paths; other projects are left unchanged.

## Layout

```text
configs/
  train_ma.yaml       # Stage 1: train calibrated channel M^A
  train_gridrel.yaml  # Stage 2: train GridRel scorer
  eval.yaml           # Full eval (WhatsUp / Winoground / … and subsets)
scripts/
  train_ma.py
  train_gridrel.py
  eval_gridrel.py
lib/                  # M^A, reshape3d, dual-branch scorer, training loop
src/                  # Channel construction, data, eval helpers
alignment/            # Stage 1 code and M^A checkpoint
```

## Training

```bash
# Stage 1 (skip if alignment/.../bilinear_metric.pt already exists)
python scripts/train_ma.py --config configs/train_ma.yaml

# Stage 2
python scripts/train_gridrel.py --config configs/train_gridrel.yaml
```

Optional overrides, e.g. `python scripts/train_gridrel.py --override train.epochs=1`.

## Evaluation

```bash
python scripts/eval_gridrel.py --config configs/eval.yaml
```

`eval.yaml` runs the full benchmark suite and writes overall metrics plus internal subsets (WhatsUp 6 splits, SugarCrepe subsets, etc.).

## Naming

In configs, `channel_mode: gridrel` is an alias of the previous `mgr_a` mode: `[M, M^A, G, R]` (12 channels).
