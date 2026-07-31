# Frequency Robustness of Medical Image Segmentation Models
### The Dice Trap: Why Clean Accuracy Fails to Predict Frequency Robustness in Medical Image Segmentation

> Status: manuscript under review, not yet published.

---
 
## The Problem
 
Segmentation models are ranked almost entirely by Dice score on clean, protocol-matched test data, a world the model never actually inhabits after deployment. Scanner hardware, reconstruction kernels, defocus, and patient motion reshape the spatial-frequency content of every image, and a clean-data leaderboard is blind to that axis by construction.
 
This project tests the Dice Trap hypothesis: a high clean Dice score does not just fail to predict robustness under frequency degradation, it can actively invert it. Three further questions follow:
 
- Does modality or architecture drive how fast and how badly a model collapses?
- What does that collapse cost clinically, measured in missed structures rather than an overlap score?
- Do these reliability properties hold under realistic conditions, or are they specific to one convenient setup?
 
---

## Approach

Three architecturally distinct segmentation paradigms are trained from scratch under an identical protocol, then stress-tested with a controlled, continuous frequency-domain perturbation. Holding architecture, training protocol, and evaluation identical across four imaging modalities isolates whether robustness is governed by the model or by the data.

| Model | Architecture | Params |
|---|---|---|
| `unet` | Classic CNN encoder-decoder with skip connections | ~31M |
| `transunet` | CNN-Transformer hybrid (ResNetV2 encoder + ViT) | ~93M |
| `swin` | Pure Transformer (Swin-Unet, shifted-window attention) | ~27M |

| Dataset | Modality | Channels | Frequency character |
|---|---|---|---|
| `isic2016` | Dermoscopy | 3 (RGB) | Texture-rich, high-frequency |
| `kvasir` | Endoscopy | 3 (RGB) | Shape-dominant, mid-frequency |
| `thyroid` | Ultrasound | 1 (grayscale) | Speckle-governed, sparse high-frequency |
| `refuge2` | Retinal fundus | 3 (RGB) | Fine vasculature, moderate-to-high frequency |

12 model x dataset combinations, each trained independently and evaluated across 51 blur levels (clean plus sigma 1 to 50), with 13 segmentation metrics tracked per image at every level spanning overlap (Dice, IoU), confusion structure (precision, recall, specificity, FNR, FPR), boundary accuracy (HD95, ASSD, boundary-F1), and topology (clDice).

| Component | Details |
|---|---|
| **Perturbation** | Isotropic Gaussian low-pass filtering, applied to the raw image before normalization to preserve its physical interpretation as progressive high-frequency information loss. |
| **CFT (Critical Frequency Threshold)** | The first blur level at which Dice drops at least 10% from that model's own clean baseline. Higher CFT means the model tolerates more degradation before meaningful failure. |
| **FRI (Frequency Robustness Index)** | Normalized area under the Dice-vs-sigma curve (0 to 1). Two models can share a CFT and still differ in how much performance they retain beyond it; FRI captures that. |
| **Clinical cost** | False Negative Rate (FNR) at representative blur levels, since a missed structure is the error that matters most in tumour, polyp, nodule, and optic-disc segmentation. |

Statistical testing, effect sizes, and the full result set live in the accompanying manuscript and are intentionally not included in this repository. What follows documents the pipeline itself: what it does, how it's configured, and exactly what goes in and comes out.

---

## Repository Structure

```
.
├── train.py               training pipeline with resume + multi-GPU
├── inference.py            frequency sweep inference
├── explainability.py       GradCAM / attention rollout on a representative image
├── config.yaml              all settings live here
├── requirements.txt
│
├── src/
│   ├── models/
│   │   ├── __init__.py    build_model() and load_checkpoint()
│   │   ├── unet.py
│   │   ├── transunet.py
│   │   └── swin_unet.py
│   ├── dataset.py          training and inference dataset classes
│   ├── losses.py            Soft-Dice + BCE combined loss
│   ├── metrics.py           all 13 segmentation metrics
│   ├── filters.py           Gaussian low-pass filter
│   ├── config_loader.py    resolves config.yaml into absolute paths
│   └── utils.py             seed, CSV helpers
│
├── Dataset/                your data (gitignored)
├── output/                 generated outputs (gitignored)
└── old/                    original Kaggle notebook and rough scripts
```

---

## Setup

```bash
git clone https://github.com/sarang-banakhede/dice-trap-freqsafe.git
cd dice-trap-freqsafe
pip install -r requirements.txt
```

### Dataset layout

All datasets go under a single `Dataset/` folder, each in its own subdirectory with a `train/` and `test/` split, each containing `images/` and `masks/`.

```
Dataset/
├── Dermoscopy_ISIC2016/
│   ├── train/{images,masks}/
│   └── test/{images,masks}/
├── Endoscopy_Kvasir/
│   ├── train/{images,masks}/
│   └── test/{images,masks}/
├── Ultrasound_Thyroid/
│   ├── train/{images,masks}/
│   └── test/{images,masks}/
└── Retinal_REFUGE2/
    ├── train/{images,masks}/
    └── test/{images,masks}/
```

| Rule | Detail |
|---|---|
| Filenames | Every image needs a mask with the exact same filename, e.g. `ISIC_0000020.png` to `ISIC_0000020.png`. |
| Mask encoding | Grayscale; pixel value > 127 is foreground, <= 127 is background. |
| Formats | `.png`, `.jpg`, `.jpeg`. |
| Channels | Thyroid (ultrasound) is single-channel grayscale; everything else is RGB. Handled automatically via `in_channels` in `config.yaml`, no manual conversion needed. |

Subdirectory names are configurable under `datasets -> subdir` in `config.yaml`; rename freely as long as the config matches.

---

## Configuration

`config.yaml` is the single source of truth for paths, per-dataset settings, hyperparameters, and the experiment matrix.

```yaml
paths:
  dataset_root: "Dataset"
  output_dir:   "output"

datasets:
  isic2016:
    subdir:      "Dermoscopy_ISIC2016"
    in_channels: 3
    norm_mean:   [0.485, 0.456, 0.406]
    norm_std:    [0.229, 0.224, 0.225]
  thyroid:
    subdir:      "Ultrasound_Thyroid"
    in_channels: 1
    norm_mean:   [0.5]
    norm_std:    [0.5]

experiments:
  - [unet, isic2016]
  - [transunet, kvasir]
  # ... all 12 combinations

training:
  img_size:    256
  epochs:      80
  batch_size:  32     # TransUNet is always forced to 8 (memory)
  lr:          1.0e-4
  seed:        42

inference:
  sigma_start:        1
  sigma_end:          50
  cft_drop_threshold: 10.0

explainability:
  image_paths:               # one representative test image per dataset
    isic2016: "ISIC_0000020.png"
    thyroid:  "0000.png"
  experiments:                # [model, dataset, critical_sigma]
    - [unet, isic2016, 25]
    - [swin, thyroid,  25]
```

To add a new dataset, add an entry under `datasets:` with the correct `subdir`, `in_channels`, and normalization stats, then reference it in `experiments:`.

---

## Usage

### 1. Training

```bash
python train.py                                       # all 12 experiments
python train.py --model unet --dataset isic2016        # a single combination
python train.py --epochs 100 --lr 5e-4 --batch-size 16 # override hyperparameters
```

Interrupted runs resume automatically: `train.py` reads `epoch_metrics.csv` to find the last completed epoch, reloads `last.pth`, and continues. Multi-GPU is automatic via `DataParallel` when `torch.cuda.device_count() > 1`.

### 2. Inference (frequency sweep)

```bash
python inference.py                                                # all 12 experiments
python inference.py --model unet --dataset isic2016
python inference.py --sigma-start 1 --sigma-end 30 --cft-threshold 5.0
```

Loads `best.pth`, runs clean inference, then sweeps sigma from `sigma_start` to `sigma_end`. Skips experiments without a trained `best.pth`, and skips output files that already exist, so partial runs are safe to resume.

### 3. Explainability

```bash
python explainability.py                              # all configured entries
python explainability.py --model unet --dataset isic2016
```

For each `[model, dataset, sigma]` entry under `explainability.experiments`, this loads `best.pth`, runs the model on the configured representative image both clean and low-pass blurred at the given sigma, and saves the predicted mask overlay plus a saliency map: GradCAM for U-Net and TransUNet, attention rollout for Swin-Unet. The `sigma` used is typically that experiment's CFT from `freq_summary.csv`, but any blur level can be inspected. Skipped if `best.pth` or the representative image doesn't exist.

---

## Output

Everything lands under `output/{model}_{dataset}/`:

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
│   └── explainability/         only if listed in explainability.experiments
│       ├── 1_original.png
│       ├── 2_lowpass_cft.png
│       ├── 3_gradcam_original.png
│       └── 4_gradcam_lowpass_cft.png
├── transunet_isic2016/  (same structure)
├── swin_isic2016/       (same structure)
├── ...
└── master_freq_summary.csv     all 12 experiments combined
```

| File | Granularity | Contents |
|---|---|---|
| `best.pth` | - | Weights at the epoch with highest validation Dice. Used for inference. |
| `last.pth` | - | Weights + optimizer state at the last completed epoch. Used only to resume training. |
| `epoch_metrics.csv` | 1 row / epoch | Train/test loss components (total, Dice, BCE) plus all 13 validation metrics. |
| `per_image_clean.csv` | 1 row / test image | All 13 metrics at sigma = 0 (no perturbation). |
| `per_image_blur.csv` | 1 row / test image | All 13 metrics at every sigma, wide format (`{metric}_sigma{sigma}`), 651 columns total. |
| `freq_sweep.csv` | 1 row / sigma level | All 13 metrics averaged across test images, plus `dice_drop_pct` and `cft_flag`. |
| `freq_summary.csv` | 1 row | Per-experiment summary: `baseline_dice`, `cft_sigma`, `fri`, Dice/FNR at sigma = 10/25/50. |
| `master_freq_summary.csv` | 1 row / experiment | All 12 `freq_summary.csv` rows combined. |
| `explainability/*.png` | 1 set / configured entry | Mask overlays and saliency maps, clean vs. blurred at the configured sigma. |

### The 13 metrics (tracked at every granularity above)

`dice_score` - `iou` - `pixel_accuracy` - `precision` - `recall` - `specificity` - `f1_score` - `fnr` - `fpr` - `hd95` - `assd` - `bf_score` - `cldice`

---

## Notes

- This repository documents the pipeline, not the findings. Statistical testing, effect sizes, and discussion live in the manuscript, which is currently under review and not included here.

## Citation

A citation entry will be added once the manuscript is accepted.
