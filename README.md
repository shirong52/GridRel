# GridRel

GridRel is a framework for spatial relation understanding with a frozen CLIP backbone. This repository provides code for two-stage training and evaluation.

## Installation

Clone the repository:

```bash
git clone https://github.com/shirong52/GridRel.git
cd GridRel
```

Use Python 3.10 and install the core dependencies below. The PyTorch command is for CUDA 12.1; choose a suitable build for your environment from the [official instructions](https://pytorch.org/get-started/previous-versions/).

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.26.1 huggingface_hub==0.27.1 numpy==1.26.3 Pillow matplotlib tqdm PyYAML
```

## Project Structure

```text
GridRel/
├── configs/       # Training and evaluation configurations
├── scripts/       # Training and evaluation entry points
├── alignment/     # Stage 1 similarity calibration
├── lib/           # GridRel model and training loop
└── src/           # Dataset readers and shared utilities
```

## Data

### Stage 1: COCO

Download **COCO 2017** from the [official website](https://cocodataset.org/#download), including the training images, validation images, and train/val annotations. Both instance and caption annotations are required.

Organize the data under `data/coco/` with `train2017/`, `val2017/`, and `annotations/` subdirectories.

### Stage 2: Objaverse Spatial

Use the Objaverse spatial training data released by [DCSM / Ideal-CLIP](https://github.com/Raphoo/DCSM_Ideal_CLIP):

- **Original download:** [Google Drive](https://drive.google.com/drive/folders/1W5KNbdWM1vSMg7TYqf1RRvcL_uhZPbUP) — select the Objaverse spatial dataset.
- **Download mirror:** [Hugging Face](https://huggingface.co/datasets/lomlsr/objaverse_spatial).

Download `objaverse_train_cutmix_newrelations_withbg.zip` and extract it into `data/`. Keep the original filenames and the paired `CUTMIX_` images. The image directory should be `data/objaverse_train_cutmix_newrelations_withbg/`.

The Hugging Face repository is a mirror of the original training data. Please credit the original dataset providers when using it.

## Model Checkpoints

Download the pretrained checkpoints from [Hugging Face](https://huggingface.co/lomlsr/GridRel):

- `bilinear_metric.pt`: Stage 1 similarity calibration.
- `bilinear_mgr_a.pt`: Trained GridRel model.

Place downloaded checkpoints in `checkpoints/` and update the corresponding paths in the configuration files.

## Usage

Run all commands from the repository root. Before running, update the `paths` section of each YAML configuration for your environment. Set `repo_root` to your GridRel checkout and `clip_model` to `openai/clip-vit-base-patch16` or a local copy of that model.

### Stage 1: Similarity Calibration

Set `paths.coco_dir` in `configs/train_ma.yaml` to your COCO directory and choose `paths.output_dir` for the training outputs. Then run:

```bash
python scripts/train_ma.py --config configs/train_ma.yaml
```

The calibration checkpoint is saved as `checkpoints/bilinear_metric.pt` inside the configured output directory. You can skip this stage by downloading the pretrained calibration checkpoint.

### Stage 2: GridRel Training

In `configs/train_gridrel.yaml`, set `paths.data_path` to the parent of the Objaverse image directory, `paths.alignment_checkpoint` to the Stage 1 checkpoint, and the model/figure output paths to your preferred locations. Then run:

```bash
python scripts/train_gridrel.py --config configs/train_gridrel.yaml
```

Model checkpoints are saved to `paths.model_save_path`. Training settings such as batch size, learning rate, and number of epochs can be changed in the configuration file.

### Evaluation

Set the model checkpoint, CLIP model, benchmark data paths, and output directory in `configs/eval.yaml`. Then run:

```bash
python scripts/eval_gridrel.py --config configs/eval.yaml
```

The default configuration evaluates WhatsUp, Winoground, SugarCrepe (`replace_rel`), and ARO Visual-Relation. Results are saved to the configured output directory.

**Note:** the evaluation script requires the external `relalign_clip` helper package, which is not included in this repository. Make it importable before running evaluation.

## Acknowledgments

This project builds on [CLIP](https://github.com/openai/CLIP) and [DCSM / Ideal-CLIP](https://github.com/Raphoo/DCSM_Ideal_CLIP). We thank the original authors for sharing their code and datasets.
