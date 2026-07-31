#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config_loader import load_config
from src.dataset import MedicalSegDataset, TestDataset
from src.losses import CombinedLoss
from src.metrics import (_boundary_f1, _cldice, _hd95_assd,
                          confusion_counts, metrics_from_confusion)
from src.models import build_model
from src.utils import csv_append, csv_init, set_seed

warnings.filterwarnings("ignore")


TRAIN_COLUMNS = [
    "epoch",
    "train_total_loss", "train_dice_loss", "train_bce_loss",
    "test_total_loss",  "test_dice_loss",  "test_bce_loss",
    "dice_score", "iou", "pixel_accuracy",
    "precision", "recall", "specificity", "f1_score",
    "fnr", "fpr", "bf_score", "cldice", "hd95", "assd",
]

TRANSUNET_BATCH = 8


def _completed_epochs(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        return int(rows[-1]["epoch"]) if rows else 0
    except Exception:
        return 0


def _safe_mean(lst: list) -> float:
    v = [x for x in lst if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    tot = dice_l = bce_l = n = 0.0
    for imgs, masks in tqdm(loader, desc="  train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=(scaler is not None)):
            prob = torch.sigmoid(model(imgs))
            total, dl, bl = criterion(prob, masks)
        if scaler:
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            optimizer.step()
        b = imgs.size(0)
        tot += total.item() * b; dice_l += dl.item() * b; bce_l += bl.item() * b; n += b
    n = max(n, 1)
    return dict(train_total_loss=tot / n, train_dice_loss=dice_l / n, train_bce_loss=bce_l / n)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    tot = dice_l = bce_l = n = 0.0
    tp = fp = fn = tn = 0.0
    hd95_l, assd_l, bf_l, cld_l = [], [], [], []

    for imgs, masks in tqdm(loader, desc="   eval", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        prob = torch.sigmoid(model(imgs))
        total, dl, bl = criterion(prob, masks)
        b = imgs.size(0)
        tot += total.item() * b; dice_l += dl.item() * b; bce_l += bl.item() * b; n += b

        pb_np = (prob > 0.5).float().cpu().numpy()
        mk_np = masks.cpu().numpy()
        _tp, _fp, _fn, _tn = confusion_counts(pb_np, mk_np)
        tp += _tp; fp += _fp; fn += _fn; tn += _tn
        for i in range(b):
            h, a = _hd95_assd(pb_np[i, 0], mk_np[i, 0])
            hd95_l.append(h); assd_l.append(a)
            bf_l.append(_boundary_f1(pb_np[i, 0], mk_np[i, 0]))
            cld_l.append(_cldice(pb_np[i, 0], mk_np[i, 0]))

    n = max(n, 1)
    m = dict(test_total_loss=tot / n, test_dice_loss=dice_l / n, test_bce_loss=bce_l / n)
    m.update(metrics_from_confusion(tp, fp, fn, tn))
    m.update(bf_score=_safe_mean(bf_l), cldice=_safe_mean(cld_l),
             hd95=_safe_mean(hd95_l), assd=_safe_mean(assd_l))
    return m


def run_experiment(model_name: str, dataset_name: str,
                   cfg: dict, device: torch.device,
                   num_gpus: int, scaler, args) -> None:
    exp_tag   = f"{model_name}_{dataset_name}"
    out_dir   = Path(cfg["paths"]["output_dir"]) / exp_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path  = out_dir / "epoch_metrics.csv"
    best_path = out_dir / "best.pth"
    last_path = out_dir / "last.pth"

    tcfg   = cfg["training"]
    epochs = args.epochs     or tcfg["epochs"]
    lr     = args.lr         or tcfg["lr"]
    ds_cfg = cfg["datasets"][dataset_name]
    in_ch  = ds_cfg["in_channels"]

    start_epoch = _completed_epochs(csv_path)
    if start_epoch >= epochs:
        print(f"  [{exp_tag}] SKIP — already complete ({epochs} epochs logged)")
        return

    eff_bs = TRANSUNET_BATCH if model_name == "transunet" else (args.batch_size or tcfg["batch_size"])
    pin    = device.type == "cuda"
    nw     = tcfg["num_workers"] if device.type == "cuda" else 0

    train_ds = MedicalSegDataset(
        ds_cfg["train_images"], ds_cfg["train_masks"],
        in_channels=in_ch, img_size=tcfg["img_size"],
        norm_mean=ds_cfg.get("norm_mean"), norm_std=ds_cfg.get("norm_std"),
        aug_cfg=tcfg["augmentation"])
    test_ds = TestDataset(
        ds_cfg["test_images"], ds_cfg["test_masks"],
        in_channels=in_ch, img_size=tcfg["img_size"],
        norm_mean=ds_cfg.get("norm_mean"), norm_std=ds_cfg.get("norm_std"),
        normalize=True)

    train_loader = DataLoader(train_ds, batch_size=eff_bs, shuffle=True,
                              num_workers=nw, pin_memory=pin, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=eff_bs, shuffle=False,
                              num_workers=nw, pin_memory=pin, drop_last=False)

    model = build_model(model_name, in_ch, tcfg["img_size"]).to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model)

    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  {'='*64}")
    print(f"  {exp_tag}  |  {params:,} params  |  in_ch={in_ch}  |  bs={eff_bs}")
    print(f"  epochs={epochs}  lr={lr}  device={device}  gpus={num_gpus}  amp={scaler is not None}")

    if start_epoch > 0 and last_path.exists():
        print(f"  Resuming from epoch {start_epoch + 1} …")
        ckpt  = torch.load(last_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        (model.module if hasattr(model, "module") else model).load_state_dict(state, strict=True)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        for _ in range(start_epoch):
            scheduler.step()

    if start_epoch == 0:
        csv_init(csv_path, TRAIN_COLUMNS)

    best_dice = -1.0
    if start_epoch > 0 and csv_path.exists():
        try:
            with open(csv_path) as f:
                best_dice = max(float(r["dice_score"]) for r in csv.DictReader(f) if r.get("dice_score"))
        except Exception:
            best_dice = -1.0

    t_start = time.time()
    for epoch in range(start_epoch + 1, epochs + 1):
        t0      = time.time()
        train_m = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        eval_m  = eval_epoch(model, test_loader, criterion, device)
        scheduler.step()
        csv_append(csv_path, TRAIN_COLUMNS, {"epoch": epoch, **train_m, **eval_m})

        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save({"epoch": epoch, "model_state": state,
                    "optimizer_state": optimizer.state_dict()}, last_path)

        cur_dice = eval_m["dice_score"]
        if cur_dice > best_dice:
            best_dice = cur_dice
            torch.save({"epoch": epoch, "dice_score": best_dice, "model_state": state}, best_path)

        if epoch % 10 == 0 or epoch == start_epoch + 1:
            print(f"  Ep {epoch:03d}/{epochs}  "
                  f"Loss={train_m['train_total_loss']:.4f}  "
                  f"Dice={cur_dice:.4f}  FNR={eval_m['fnr']:.4f}  "
                  f"HD95={eval_m['hd95']:.2f}  ({time.time() - t0:.1f}s)")

    elapsed = time.time() - t_start
    print(f"  DONE  |  Best Dice={best_dice:.4f}  |  {elapsed / 60:.1f} min")
    print(f"  {'='*64}\n")

    del model, optimizer, scheduler, criterion, train_loader, test_loader, train_ds, test_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="FreqSafe — training pipeline")
    parser.add_argument("--model",      choices=["unet", "transunet", "swin"],
                        help="train a single model (default: all three)")
    parser.add_argument("--dataset",    choices=["isic2016", "kvasir", "thyroid", "refuge2"],
                        help="train on a single dataset (default: all four)")
    parser.add_argument("--epochs",     type=int,   default=None, help="override config epochs")
    parser.add_argument("--lr",         type=float, default=None, help="override learning rate")
    parser.add_argument("--batch-size", type=int,   default=None, dest="batch_size",
                        help="override batch size (TransUNet is always 8 regardless)")
    parser.add_argument("--config",     default="config.yaml", help="path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    Path(cfg["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    scaler   = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    experiments = cfg["experiments"]
    if args.model or args.dataset:
        experiments = [
            (m, d) for m, d in experiments
            if (args.model   is None or m == args.model)
            and (args.dataset is None or d == args.dataset)
        ]

    print(f"\n  FreqSafe — Training Pipeline")
    print(f"  Device: {device}  |  GPUs: {num_gpus}  |  AMP: {scaler is not None}")
    print(f"  Experiments: {len(experiments)}\n")

    for m, d in experiments:
        run_experiment(m, d, cfg, device, num_gpus, scaler, args)

    print(f"  All done. Outputs → {cfg['paths']['output_dir']}/")


if __name__ == "__main__":
    main()
