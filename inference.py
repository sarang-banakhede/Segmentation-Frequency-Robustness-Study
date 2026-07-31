#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config_loader import load_config
from src.dataset import TestDataset, _norm
from src.filters import apply_low_pass
from src.metrics import (_boundary_f1, _cldice, _hd95_assd,
                          compute_single_metrics, confusion_counts, metrics_from_confusion)
from src.models import build_model, load_checkpoint
from src.utils import set_seed

warnings.filterwarnings("ignore")


METRIC_NAMES = [
    "dice_score", "iou", "pixel_accuracy",
    "precision", "recall", "specificity", "f1_score",
    "fnr", "fpr", "hd95", "assd", "bf_score", "cldice",
]

CLEAN_COLS = ["image_name"] + METRIC_NAMES

SWEEP_AGG_COLS = (
    ["model", "dataset", "sigma"] + METRIC_NAMES + ["dice_drop_pct", "cft_flag"]
)

SUMMARY_COLS = [
    "model", "dataset", "baseline_dice", "cft_sigma", "fri",
    "dice_at_sigma10", "dice_at_sigma25", "dice_at_sigma50",
    "fnr_at_sigma10",  "fnr_at_sigma25",  "fnr_at_sigma50",
]


def _sweep_per_img_cols(sigmas):
    return ["image_name"] + [f"{m}_sigma{s}" for s in sigmas for m in METRIC_NAMES]


def _safe_mean(lst):
    v = [x for x in lst if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def _csv_write(path: Path, cols: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def _compute_agg(pred_bin: np.ndarray, masks: np.ndarray) -> dict:
    tp = fp = fn = tn = 0.0
    hd95_l, assd_l, bf_l, cld_l = [], [], [], []
    for i in range(pred_bin.shape[0]):
        _tp, _fp, _fn, _tn = confusion_counts(pred_bin[i, 0], masks[i, 0])
        tp += _tp; fp += _fp; fn += _fn; tn += _tn
        h, a = _hd95_assd(pred_bin[i, 0], masks[i, 0])
        hd95_l.append(h); assd_l.append(a)
        bf_l.append(_boundary_f1(pred_bin[i, 0], masks[i, 0]))
        cld_l.append(_cldice(pred_bin[i, 0], masks[i, 0]))
    out = metrics_from_confusion(tp, fp, fn, tn)
    out.update(hd95=_safe_mean(hd95_l), assd=_safe_mean(assd_l),
               bf_score=_safe_mean(bf_l), cldice=_safe_mean(cld_l))
    return out


def _compute_cft(dice_by_sigma: dict, baseline: float, thr: float):
    for s in sorted(dice_by_sigma):
        if ((baseline - dice_by_sigma[s]) / (baseline + 1e-9)) * 100 >= thr:
            return s
    return None


def _compute_fri(dice_by_sigma: dict, baseline: float, max_s: int) -> float:
    sigs = sorted(dice_by_sigma)
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(_trapz([dice_by_sigma[s] for s in sigs], sigs)) / (baseline * max_s + 1e-9)


@torch.no_grad()
def _infer(model, loader, device, sigma: float, norm_mean, norm_std):
    model.eval()
    preds, masks_out = [], []
    for imgs_raw, masks in loader:
        x = imgs_raw.to(device)
        if sigma > 0:
            x = apply_low_pass(x, sigma)
        if norm_mean and norm_std:
            x = _norm(x, norm_mean, norm_std)
        pred_bin = (torch.sigmoid(model(x)) > 0.5).squeeze(1).cpu().numpy()
        mask_np  = masks.squeeze(1).cpu().numpy()
        for i in range(pred_bin.shape[0]):
            preds.append(pred_bin[i])
            masks_out.append(mask_np[i])
    return preds, masks_out


def run_experiment(model_name: str, dataset_name: str,
                   cfg: dict, device: torch.device,
                   sigmas: list, cft_thr: float) -> None:
    exp_tag     = f"{model_name}_{dataset_name}"
    out_dir     = Path(cfg["paths"]["output_dir"]) / exp_tag
    weights     = out_dir / "best.pth"

    if not weights.exists():
        print(f"  [{exp_tag}] SKIP — best.pth not found (train first)")
        return

    clean_csv   = out_dir / "per_image_clean.csv"
    blur_csv    = out_dir / "per_image_blur.csv"
    sweep_csv   = out_dir / "freq_sweep.csv"
    summary_csv = out_dir / "freq_summary.csv"

    if all(p.exists() for p in [clean_csv, blur_csv, sweep_csv, summary_csv]):
        print(f"  [{exp_tag}] SKIP — all outputs already exist")
        return

    ds_cfg    = cfg["datasets"][dataset_name]
    icfg      = cfg["inference"]
    in_ch     = ds_cfg["in_channels"]
    norm_mean = ds_cfg.get("norm_mean")
    norm_std  = ds_cfg.get("norm_std")
    pin       = device.type == "cuda"
    nw        = icfg["num_workers"] if device.type == "cuda" else 0

    print(f"\n  {'='*64}")
    print(f"  INFERENCE: {exp_tag}")

    model = build_model(model_name, in_ch, cfg["training"]["img_size"]).to(device)
    ckpt  = load_checkpoint(model, str(weights), device)
    model.eval()
    print(f"  Loaded best.pth (epoch {ckpt.get('epoch', '?')})")

    ds = TestDataset(
        ds_cfg["test_images"], ds_cfg["test_masks"],
        in_channels=in_ch, img_size=cfg["training"]["img_size"],
        norm_mean=None, norm_std=None, normalize=False)
    loader = DataLoader(ds, batch_size=icfg["batch_size"], shuffle=False,
                        num_workers=nw, pin_memory=pin, drop_last=False)

    img_names = [Path(p).stem for p in ds.image_paths]
    N         = len(img_names)

    if not clean_csv.exists():
        print(f"  Clean inference ({N} images) …")
        c_preds, c_masks = _infer(model, loader, device, 0.0, norm_mean, norm_std)
        _csv_write(clean_csv, CLEAN_COLS,
                   [{"image_name": img_names[i],
                     **compute_single_metrics(c_preds[i], c_masks[i])} for i in range(N)])
        baseline_agg  = _compute_agg(np.stack(c_preds)[:, np.newaxis],
                                     np.stack(c_masks)[:, np.newaxis])
        baseline_dice = baseline_agg["dice_score"]
    else:
        print("  Clean CSV exists — reading baseline from sweep …")
        c_preds = c_masks = None
        if sweep_csv.exists():
            with open(sweep_csv) as f:
                row0 = next(r for r in csv.DictReader(f) if int(r["sigma"]) == 0)
            baseline_dice = float(row0["dice_score"])
        else:
            c_preds, c_masks = _infer(model, loader, device, 0.0, norm_mean, norm_std)
            baseline_agg  = _compute_agg(np.stack(c_preds)[:, np.newaxis],
                                         np.stack(c_masks)[:, np.newaxis])
            baseline_dice = baseline_agg["dice_score"]

    print(f"  Baseline Dice = {baseline_dice:.4f}")

    if not (blur_csv.exists() and sweep_csv.exists()):
        print(f"  Blur sweep σ={sigmas[0]}→{sigmas[-1]} …")

        if c_preds is None:
            c_preds, c_masks = _infer(model, loader, device, 0.0, norm_mean, norm_std)
            baseline_agg     = _compute_agg(np.stack(c_preds)[:, np.newaxis],
                                            np.stack(c_masks)[:, np.newaxis])

        per_img   = [{"image_name": img_names[i]} for i in range(N)]
        agg_rows  = [dict(model=model_name, dataset=dataset_name, sigma=0,
                          dice_drop_pct=0.0, cft_flag=False, **baseline_agg)]
        dice_by_s: dict[int, float] = {}
        fnr_by_s:  dict[int, float] = {}

        for sigma in tqdm(sigmas, desc="    blur", ncols=70):
            s_preds, s_masks = _infer(model, loader, device, float(sigma), norm_mean, norm_std)
            for i in range(N):
                m = compute_single_metrics(s_preds[i], s_masks[i])
                for mn in METRIC_NAMES:
                    per_img[i][f"{mn}_sigma{sigma}"] = m.get(mn, float("nan"))
            agg      = _compute_agg(np.stack(s_preds)[:, np.newaxis],
                                    np.stack(s_masks)[:, np.newaxis])
            drop_pct = ((baseline_dice - agg["dice_score"]) / (baseline_dice + 1e-9)) * 100.0
            agg_rows.append(dict(model=model_name, dataset=dataset_name, sigma=sigma,
                                 dice_drop_pct=round(drop_pct, 4),
                                 cft_flag=(drop_pct >= cft_thr), **agg))
            dice_by_s[sigma] = agg["dice_score"]
            fnr_by_s[sigma]  = agg["fnr"]

        _csv_write(blur_csv,  _sweep_per_img_cols(sigmas), per_img)
        _csv_write(sweep_csv, SWEEP_AGG_COLS, agg_rows)
    else:
        print("  Blur CSVs already exist — reading sigma data …")
        dice_by_s = {}; fnr_by_s = {}
        with open(sweep_csv) as f:
            for row in csv.DictReader(f):
                s = int(row["sigma"])
                if s > 0:
                    dice_by_s[s] = float(row["dice_score"])
                    fnr_by_s[s]  = float(row["fnr"])

    if not summary_csv.exists():
        cft = _compute_cft(dice_by_s, baseline_dice, cft_thr)
        fri = _compute_fri(dice_by_s, baseline_dice, max(sigmas))
        print(f"  CFT={cft}  FRI={fri:.4f}")

        def _at(d, s): return round(d.get(s, float("nan")), 4)
        _csv_write(summary_csv, SUMMARY_COLS, [dict(
            model=model_name, dataset=dataset_name,
            baseline_dice=round(baseline_dice, 4),
            cft_sigma=cft if cft is not None else "never",
            fri=round(fri, 4),
            dice_at_sigma10=_at(dice_by_s, 10), dice_at_sigma25=_at(dice_by_s, 25),
            dice_at_sigma50=_at(dice_by_s, 50),
            fnr_at_sigma10=_at(fnr_by_s, 10),  fnr_at_sigma25=_at(fnr_by_s, 25),
            fnr_at_sigma50=_at(fnr_by_s, 50),
        )])

    del model, loader, ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def build_master_summary(cfg: dict) -> None:
    out_dir     = Path(cfg["paths"]["output_dir"])
    master_path = out_dir / "master_freq_summary.csv"
    rows = []
    for m, d in cfg["experiments"]:
        p = out_dir / f"{m}_{d}" / "freq_summary.csv"
        if not p.exists():
            continue
        with open(p) as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        print("  No freq_summary.csv found — run inference first."); return
    with open(master_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n  Master summary → {master_path}  ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="FreqSafe — inference & frequency sweep")
    parser.add_argument("--model",           choices=["unet", "transunet", "swin"],
                        help="run a single model (default: all)")
    parser.add_argument("--dataset",         choices=["isic2016", "kvasir", "thyroid", "refuge2"],
                        help="run on a single dataset (default: all)")
    parser.add_argument("--sigma-start",     type=int,   default=None, dest="sigma_start",
                        help="override sigma_start from config")
    parser.add_argument("--sigma-end",       type=int,   default=None, dest="sigma_end",
                        help="override sigma_end from config")
    parser.add_argument("--cft-threshold",   type=float, default=None, dest="cft_threshold",
                        help="override cft_drop_threshold from config")
    parser.add_argument("--config",          default="config.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    Path(cfg["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    icfg   = cfg["inference"]

    sigma_start = args.sigma_start  or icfg["sigma_start"]
    sigma_end   = args.sigma_end    or icfg["sigma_end"]
    cft_thr     = args.cft_threshold or icfg["cft_drop_threshold"]
    sigmas      = list(range(sigma_start, sigma_end + 1))

    experiments = cfg["experiments"]
    if args.model or args.dataset:
        experiments = [
            (m, d) for m, d in experiments
            if (args.model   is None or m == args.model)
            and (args.dataset is None or d == args.dataset)
        ]

    print(f"\n  FreqSafe — Inference Pipeline")
    print(f"  Device: {device}  |  σ={sigma_start}→{sigma_end}  |  CFT threshold={cft_thr}%")
    print(f"  Experiments: {len(experiments)}\n")

    for m, d in experiments:
        run_experiment(m, d, cfg, device, sigmas, cft_thr)

    print("\n  Building master summary …")
    build_master_summary(cfg)
    print("  All done.\n")


if __name__ == "__main__":
    main()
