# FreqSafe — Frequency Robustness Analysis of Medical Image Segmentation Models

Medical image segmentation models are typically evaluated on Dice score — and that's the problem.

A model that achieves high Dice on clean test data may completely collapse when the input quality drops even slightly. This project investigates whether Dice score actually predicts how robust a model is to frequency-domain perturbations (Gaussian blur), and does so systematically across multiple architectures and imaging modalities.

The core question: **if two models have similar Dice scores, do they fail at the same rate under degradation?**

The experiments here test three architectures — a pure CNN (UNet), a CNN-Transformer hybrid (TransUNet), and a pure Transformer (SwinUNet) — across four medical imaging modalities: dermoscopy, endoscopy, ultrasound, and retinal fundus. Each model is trained clean, then evaluated under progressive Gaussian blur (σ = 1 to 50), with 13 segmentation metrics tracked per image per sigma level.

Two robustness metrics are introduced:
- **CFT** (Critical Frequency Threshold) — the first blur level where Dice drops ≥10% from baseline
- **FRI** (Frequency Robustness Index) — normalized area under the Dice-vs-σ curve; higher = more robust

The findings reveal a consistent pattern across modalities that challenges how robustness is assumed to correlate with clean performance. Full analysis is in the accompanying paper.

---

## Models

| Key | Architecture | Params |
|---|---|---|
| `unet` | Classic encoder-decoder with skip connections | ~31M |
| `transunet` | ResNetV2 CNN encoder + ViT + CNN decoder | ~93M |
| `swin` | Swin Transformer encoder-decoder (Swin-UNet) | ~27M |

## Datasets

| Key | Modality | Dataset | Channels |
|---|---|---|---|
| `isic2016` | Dermoscopy | ISIC 2016 | 3 |
| `kvasir` | Endoscopy | Kvasir-SEG | 3 |
| `thyroid` | Ultrasound | TN3K | 1 |
| `refuge2` | Retinal fundus | REFUGE2 | 3 |

---

## Setup

```bash
git clone https://github.com/your-username/freqsafe.git
cd freqsafe
pip install -r requirements.txt
```

---

## Dataset folder structure

All datasets go under a single `Dataset/` folder. Each dataset has its own subdirectory with a `train/` and `test/` split, each containing an `images/` and `masks/` folder.

```
Dataset/
├── Dermoscopy_ISIC2016/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
│
├── Endoscopy_Kvasir/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
│
├── Ultrasound_Thyroid/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
│
└── Retinal_REFUGE2/
    ├── train/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/
```

**Rules for images and masks**

- Every image must have a corresponding mask with the **exact same filename** (e.g. `ISIC_0000020.png` → `ISIC_0000020.png`)
- Masks must be grayscale — pixel value > 127 is foreground, ≤ 127 is background
- Supported formats: `.png`, `.jpg`, `.jpeg`
- Thyroid (ultrasound) images are single-channel grayscale. Everything else is RGB. The pipeline handles this automatically based on `in_channels` in `config.yaml` — no manual conversion needed

The subdirectory names (`Dermoscopy_ISIC2016`, `Endoscopy_Kvasir`, etc.) are configured in `config.yaml` under `datasets → subdir`. You can rename them as long as you update the config to match.

---

## Configuration

`config.yaml` is the single source of truth for all paths, hyperparameters, and experiment settings.

```yaml
paths:
  dataset_root: "Dataset"    # root folder containing all dataset subdirectories
  output_dir:   "output"     # all weights, CSVs, and summaries go here

datasets:
  isic2016:
    subdir:      "Dermoscopy_ISIC2016"
    in_channels: 3
    norm_mean:   [0.485, 0.456, 0.406]
    norm_std:    [0.229, 0.224, 0.225]
  thyroid:
    subdir:      "Ultrasound_Thyroid"
    in_channels: 1           # grayscale — single channel
    norm_mean:   [0.5]
    norm_std:    [0.5]

experiments:
  - [unet, isic2016]
  - [transunet, kvasir]
  # ... all 12 combinations

training:
  img_size:    256
  epochs:      80
  batch_size:  32            # TransUNet is always forced to 8 (memory)
  lr:          1.0e-4
  seed:        42

inference:
  sigma_start:        1
  sigma_end:          50
  cft_drop_threshold: 10.0   # % Dice drop that defines the CFT

explainability:
  image_paths:              # one representative test image per dataset
    isic2016: "ISIC_0000020.png"
    thyroid:  "0000.png"
  experiments:               # [model, dataset, critical_sigma] — sigma from freq_summary.csv
    - [unet, isic2016, 25]
    - [swin, thyroid,  25]
    # ... one row per model+dataset you want explainability for
```

To use your own dataset: add a new entry under `datasets:`, set the correct `subdir`, `in_channels`, and normalization stats, then add it to the `experiments:` list.

---

## Training

**Train all 12 experiments sequentially**
```bash
python train.py
```

**Train a specific model and dataset**
```bash
python train.py --model unet --dataset isic2016
python train.py --model transunet --dataset kvasir
python train.py --model swin --dataset thyroid
```

**Override hyperparameters from command line**
```bash
python train.py --epochs 100 --lr 5e-4 --batch-size 16
python train.py --model unet --dataset refuge2 --epochs 50
```

**Resuming an interrupted run** is automatic — the script reads `epoch_metrics.csv` to find the last completed epoch, loads `last.pth`, and picks up from there. No flags needed.

**Multi-GPU** is also automatic. If `torch.cuda.device_count() > 1`, DataParallel is used. Single GPU and CPU work without any changes.

---

## Inference

After training, run the frequency sweep. This loads `best.pth` for each experiment, runs clean inference, then evaluates the model at every σ level from `sigma_start` to `sigma_end`.

**Run inference on all 12 experiments**
```bash
python inference.py
```

**Single model + dataset**
```bash
python inference.py --model unet --dataset isic2016
```

**Custom sigma range or CFT sensitivity**
```bash
python inference.py --sigma-start 1 --sigma-end 30
python inference.py --cft-threshold 5.0
python inference.py --model swin --dataset refuge2 --sigma-end 25 --cft-threshold 15.0
```

Inference skips any experiment where `best.pth` does not exist (i.e. not trained yet). It also skips individual output files that already exist, so partial runs are safe to resume.

---

## Explainability

For each `[model, dataset, sigma]` entry under `explainability.experiments` in `config.yaml`, this loads `best.pth`, runs the model on the configured representative image (`explainability.image_paths`) both clean and low-pass blurred at the given `sigma`, and saves the predicted mask overlay plus a saliency map — GradCAM for UNet/TransUNet, attention rollout for SwinUNet.

**Run all configured experiments**
```bash
python explainability.py
```

**Single model + dataset**
```bash
python explainability.py --model unet --dataset isic2016
```

The `sigma` for each entry is the CFT you found from that experiment's `freq_summary.csv` (or any blur level you want to inspect). Output goes to `output/{model}_{dataset}/explainability/`:

| File | Description |
|---|---|
| `1_original.png` | Clean image with predicted mask overlay |
| `2_lowpass_cft.png` | Image blurred at `sigma`, with predicted mask overlay |
| `3_gradcam_original.png` / `4_gradcam_lowpass_cft.png` | GradCAM heatmap, clean vs. blurred (UNet, TransUNet) |
| `3_attention_original.png` / `4_attention_lowpass_cft.png` | Swin attention rollout, clean vs. blurred (SwinUNet) |

Experiments are skipped if `best.pth` or the representative image doesn't exist.

---

## Output folder structure

Everything lands under `output/`, organized by `{model}_{dataset}/`.

```
output/
├── unet_isic2016/
│   ├── best.pth
│   ├── last.pth
│   ├── epoch_metrics.csv
│   ├── per_image_clean.csv
│   ├── per_image_blur.csv
│   ├── freq_sweep.csv
│   ├── freq_summary.csv
│   └── explainability/          ← only if listed in explainability.experiments
│       ├── 1_original.png
│       ├── 2_lowpass_cft.png
│       ├── 3_gradcam_original.png
│       └── 4_gradcam_lowpass_cft.png
│
├── transunet_isic2016/
│   └── (same structure)
│
├── swin_isic2016/
│   └── (same structure)
│
├── unet_kvasir/ ...
├── transunet_kvasir/ ...
│
└── master_freq_summary.csv     ← all 12 experiments in one file
```

### What each file contains

**`best.pth`**
Model weights at the epoch with the highest validation Dice. This is what inference uses.

**`last.pth`**
Model weights at the last completed epoch, along with optimizer state. Used only for resuming interrupted training.

---

**`epoch_metrics.csv`**
One row per training epoch. Tracks loss and all 13 validation metrics across the full training run.

| Column | Description |
|---|---|
| `epoch` | Epoch number |
| `train_total_loss` | Combined Dice + BCE loss on training set |
| `train_dice_loss` | Soft-Dice loss component |
| `train_bce_loss` | Binary cross-entropy component |
| `test_total_loss` | Combined loss on validation/test set |
| `test_dice_loss` | Dice loss on test set |
| `test_bce_loss` | BCE loss on test set |
| `dice_score` | Dice coefficient on test set |
| `iou` | Intersection over Union |
| `pixel_accuracy` | Overall pixel-level accuracy |
| `precision` | TP / (TP + FP) |
| `recall` | TP / (TP + FN) |
| `specificity` | TN / (TN + FP) |
| `f1_score` | Harmonic mean of precision and recall |
| `fnr` | False Negative Rate — fraction of lesion missed |
| `fpr` | False Positive Rate |
| `hd95` | 95th percentile Hausdorff Distance (boundary accuracy) |
| `assd` | Average Symmetric Surface Distance |
| `bf_score` | Boundary F1 score |
| `cldice` | Centerline Dice — topology-aware metric |

---

**`per_image_clean.csv`**
One row per test image, no perturbation. Contains all 13 metrics for every individual image at baseline (σ = 0).

| Column | Description |
|---|---|
| `image_name` | Filename stem of the test image |
| `dice_score` ... `cldice` | All 13 metrics for that image |

---

**`per_image_blur.csv`**
One row per test image, with metrics at every sigma level. Wide format — columns are named `{metric}_sigma{σ}`.

| Column | Description |
|---|---|
| `image_name` | Filename stem |
| `dice_score_sigma1` | Dice at σ = 1 |
| `dice_score_sigma2` | Dice at σ = 2 |
| ... | ... |
| `cldice_sigma50` | clDice at σ = 50 |

Total columns = 1 (image_name) + 13 metrics × 50 sigma levels = 651 columns.

---

**`freq_sweep.csv`**
One row per sigma level (including σ = 0 as baseline). Aggregated across all test images.

| Column | Description |
|---|---|
| `model` | Model name |
| `dataset` | Dataset name |
| `sigma` | Blur level (0 = clean) |
| `dice_score` ... `cldice` | Mean of each metric across all test images |
| `dice_drop_pct` | % drop in Dice relative to baseline |
| `cft_flag` | `True` if Dice drop ≥ CFT threshold at this sigma |

---

**`freq_summary.csv`**
One row — the per-experiment robustness summary.

| Column | Description |
|---|---|
| `model` | Model name |
| `dataset` | Dataset name |
| `baseline_dice` | Clean Dice (σ = 0) |
| `cft_sigma` | First σ where Dice drops ≥10% — `never` if it never drops |
| `fri` | Frequency Robustness Index (normalized AUC, 0–1) |
| `dice_at_sigma10/25/50` | Dice at three key checkpoints |
| `fnr_at_sigma10/25/50` | False Negative Rate at three key checkpoints |

---

**`master_freq_summary.csv`**
All 12 experiments combined into one file (same columns as `freq_summary.csv`). Generated at the end of `python inference.py`.

---

## Repository structure

```
freqsafe/
├── train.py              ← training pipeline with resume + multi-GPU
├── inference.py          ← frequency sweep inference
├── explainability.py     ← GradCAM / attention rollout on a representative image
├── config.yaml           ← all settings live here
├── requirements.txt
│
├── src/
│   ├── models/
│   │   ├── __init__.py   ← build_model() and load_checkpoint()
│   │   ├── unet.py
│   │   ├── transunet.py
│   │   └── swin_unet.py
│   ├── dataset.py        ← training and inference dataset classes
│   ├── losses.py         ← Soft-Dice + BCE combined loss
│   ├── metrics.py        ← all 13 segmentation metrics
│   ├── filters.py        ← Gaussian low-pass filter
│   ├── config_loader.py  ← resolves config.yaml into absolute paths
│   └── utils.py          ← seed, CSV helpers
│
├── Dataset/              ← your data (gitignored)
├── output/               ← generated outputs (gitignored)
└── old/                  ← original Kaggle notebook and rough scripts
```
