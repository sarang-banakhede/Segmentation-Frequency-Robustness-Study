from __future__ import annotations
import argparse
import gc
import math
import warnings
from pathlib import Path

import cv2
import matplotlib; matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.config_loader import load_config
from src.dataset import _norm
from src.filters import apply_low_pass
from src.models import (
    SwinTransformerSys, _window_partition, _window_reverse,
    build_model, load_checkpoint,
)

warnings.filterwarnings("ignore")


def load_image(path: str, in_channels: int, img_size: int = 256):
    mode = "L" if in_channels == 1 else "RGB"
    pil  = Image.open(path).convert(mode).resize((img_size, img_size), Image.BILINEAR)
    arr  = np.array(pil, dtype=np.float32) / 255.0
    if in_channels == 1:
        tensor  = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        display = np.stack([arr, arr, arr], axis=-1)
    else:
        tensor  = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        display = arr
    return display.clip(0, 1), tensor


def tensor_to_display(t: torch.Tensor, in_channels: int) -> np.ndarray:
    arr = t.squeeze(0).cpu().numpy()
    arr = np.stack([arr[0]] * 3, axis=-1) if in_channels == 1 else arr.transpose(1, 2, 0)
    return arr.clip(0.0, 1.0)


def save_img(arr: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def overlay_heatmap(img_np: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (alpha * heatmap + (1 - alpha) * img_np).clip(0.0, 1.0)


def overlay_mask(img_np: np.ndarray, mask: np.ndarray, color=(0, 1, 0), alpha: float = 0.35) -> np.ndarray:
    out = img_np.copy()
    for c, col in enumerate(color):
        out[:, :, c] = np.where(mask > 0.5, alpha * col + (1 - alpha) * img_np[:, :, c], img_np[:, :, c])
    return out.clip(0.0, 1.0)


@torch.no_grad()
def predict(model: nn.Module, tensor_raw: torch.Tensor, device: torch.device,
            sigma: float, norm_mean: list | None, norm_std: list | None) -> np.ndarray:
    x = tensor_raw.to(device)
    if sigma > 0:
        x = apply_low_pass(x, sigma)
    if norm_mean and norm_std:
        x = _norm(x, norm_mean, norm_std)
    prob = torch.sigmoid(model(x))
    return (prob > 0.5).squeeze().cpu().numpy().astype(np.float32)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module, img_size: int):
        self.model, self.img_size = model, img_size
        self.grads = self.acts = None
        self._hooks = [
            target_layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o.detach())),
            target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0].detach())),
        ]

    def remove(self):
        for h in self._hooks:
            h.remove()

    def __call__(self, inp: torch.Tensor) -> np.ndarray:
        self.model.zero_grad()
        inp = inp.requires_grad_(True)
        torch.sigmoid(self.model(inp)).sum().backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self.acts).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        mn, mx  = cam.min(), cam.max()
        return ((cam - mn) / (mx - mn + 1e-8)).astype(np.float32)


GRADCAM_TARGETS = {
    "unet":      lambda m: m.enc4.net,
    "transunet": lambda m: m.emb.hybrid.body.block3,
}


def run_gradcam(model_name: str, model: nn.Module, tensor_raw: torch.Tensor, device: torch.device,
                sigma: float, norm_mean, norm_std, img_size: int) -> np.ndarray:
    x = tensor_raw.to(device)
    if sigma > 0:
        x = apply_low_pass(x, sigma)
    if norm_mean and norm_std:
        x = _norm(x, norm_mean, norm_std)
    gcam = GradCAM(model, GRADCAM_TARGETS[model_name](model), img_size)
    cam  = gcam(x)
    gcam.remove()
    return cam


@torch.no_grad()
def run_swin_attention(model: SwinTransformerSys, tensor_raw: torch.Tensor, device: torch.device,
                       sigma: float, norm_mean, norm_std, img_size: int) -> np.ndarray:
    model.eval()
    x = tensor_raw.to(device)
    if sigma > 0:
        x = apply_low_pass(x, sigma)
    if norm_mean and norm_std:
        x = _norm(x, norm_mean, norm_std)

    x = model.patch_embed(x)
    if model.ape:
        x = x + model.absolute_pos_embed
    x = model.pos_drop(x)

    last_attn = None
    for layer_idx, layer in enumerate(model.layers):
        if layer_idx != len(model.layers) - 1:
            x = layer(x)
            continue
        for blk in layer.blocks:
            H, W     = blk.input_resolution
            B, L, C  = x.shape
            shortcut = x
            xn = blk.norm1(x).view(B, H, W, C)
            sx = torch.roll(xn, (-blk.shift_size, -blk.shift_size), (1, 2)) if blk.shift_size > 0 else xn
            xw = _window_partition(sx, blk.window_size).view(-1, blk.window_size * blk.window_size, C)
            B_, N, C_ = xw.shape
            qkv = blk.attn.qkv(xw).float().reshape(
                B_, N, 3, blk.attn.num_heads, C_ // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q * blk.attn.scale) @ k.transpose(-2, -1)
            rb = blk.attn.relative_position_bias_table[
                blk.attn.relative_position_index.view(-1)].view(
                blk.window_size * blk.window_size, blk.window_size * blk.window_size, -1)
            attn = attn + rb.permute(2, 0, 1).float().unsqueeze(0)
            if blk.attn_mask is not None:
                nW   = blk.attn_mask.shape[0]
                attn = (attn.view(B_ // nW, nW, blk.attn.num_heads, N, N)
                        + blk.attn_mask.unsqueeze(1).unsqueeze(0).float())
                attn = attn.view(-1, blk.attn.num_heads, N, N)
            attn_w    = blk.attn.softmax(attn)
            last_attn = attn_w.mean(dim=1)

            aw_out = (blk.attn.attn_drop(attn_w) @ v.float()).to(x.dtype)
            aw_out = aw_out.transpose(1, 2).reshape(B_, N, C_)
            aw_out = blk.attn.proj_drop(blk.attn.proj(aw_out))
            sx2 = _window_reverse(aw_out.view(-1, blk.window_size, blk.window_size, C_), blk.window_size, H, W)
            x2  = torch.roll(sx2, (blk.shift_size, blk.shift_size), (1, 2)) if blk.shift_size > 0 else sx2
            x   = shortcut + blk.drop_path(x2.view(B, L, C_))
            x   = x + blk.drop_path(blk.mlp(blk.norm2(x)))
        if layer.downsample:
            x = layer.downsample(x)

    if last_attn is None:
        return np.zeros((img_size, img_size), dtype=np.float32)

    attn_map = last_attn.mean(dim=1)
    ws       = int(math.sqrt(attn_map.shape[1]))
    attn_map = attn_map.view(-1, ws, ws)
    gh, gw   = model.patches_resolution[0] // 8, model.patches_resolution[1] // 8
    if attn_map.shape[0] == gh * gw:
        attn_grid = attn_map.view(gh, gw, ws, ws).permute(0, 2, 1, 3).reshape(gh * ws, gw * ws)
    else:
        attn_grid = attn_map.mean(dim=0)
    attn_np = cv2.resize(attn_grid.cpu().float().numpy(), (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    mn, mx  = attn_np.min(), attn_np.max()
    return ((attn_np - mn) / (mx - mn + 1e-8)).astype(np.float32)


def process_experiment(model_name: str, dataset_name: str, sigma: float,
                       cfg: dict, device: torch.device, out_root: Path) -> None:
    exp_tag = f"{model_name}_{dataset_name}"
    weights = out_root / exp_tag / "best.pth"
    out_dir = out_root / exp_tag / "explainability"

    if not weights.exists():
        print(f"  [{exp_tag}] SKIP — best.pth not found (train first)")
        return

    ds_cfg    = cfg["datasets"][dataset_name]
    in_ch     = ds_cfg["in_channels"]
    norm_mean = ds_cfg.get("norm_mean")
    norm_std  = ds_cfg.get("norm_std")
    img_size  = cfg["training"]["img_size"]
    img_path  = cfg["explainability"]["image_paths"][dataset_name]

    if not Path(img_path).exists():
        print(f"  [{exp_tag}] SKIP — image not found: {img_path}")
        return

    print(f"\n  [{exp_tag}]  sigma={sigma}  img={Path(img_path).name}")

    np_orig, tensor_raw = load_image(img_path, in_ch, img_size)
    np_lp = tensor_to_display(apply_low_pass(tensor_raw, float(sigma)), in_ch)

    model = build_model(model_name, in_ch, img_size).to(device)
    load_checkpoint(model, str(weights), device)
    model.eval()

    mask_orig = predict(model, tensor_raw, device, 0.0, norm_mean, norm_std)
    mask_lp   = predict(model, tensor_raw, device, float(sigma), norm_mean, norm_std)
    save_img(overlay_mask(np_orig, mask_orig), out_dir / "1_original.png")
    save_img(overlay_mask(np_lp,   mask_lp),   out_dir / "2_lowpass_cft.png")

    if model_name in GRADCAM_TARGETS:
        cam_orig = run_gradcam(model_name, model, tensor_raw, device, 0.0, norm_mean, norm_std, img_size)
        cam_lp   = run_gradcam(model_name, model, tensor_raw, device, float(sigma), norm_mean, norm_std, img_size)
        save_img(overlay_heatmap(np_orig, cam_orig), out_dir / "3_gradcam_original.png")
        save_img(overlay_heatmap(np_lp,   cam_lp),   out_dir / "4_gradcam_lowpass_cft.png")
    else:
        attn_orig = run_swin_attention(model, tensor_raw, device, 0.0, norm_mean, norm_std, img_size)
        attn_lp   = run_swin_attention(model, tensor_raw, device, float(sigma), norm_mean, norm_std, img_size)
        save_img(overlay_heatmap(np_orig, attn_orig), out_dir / "3_attention_original.png")
        save_img(overlay_heatmap(np_lp,   attn_lp),   out_dir / "4_attention_lowpass_cft.png")

    print(f"    saved -> {out_dir}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="FreqSafe — explainability")
    parser.add_argument("--model",   choices=["unet", "transunet", "swin"])
    parser.add_argument("--dataset", choices=["isic2016", "kvasir", "thyroid", "refuge2"])
    parser.add_argument("--config",  default="config.yaml")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(cfg["explainability"].get("output_dir", cfg["paths"]["output_dir"]))

    experiments = cfg["explainability"]["experiments"]
    if args.model or args.dataset:
        experiments = [
            (m, d, s) for m, d, s in experiments
            if (args.model   is None or m == args.model)
            and (args.dataset is None or d == args.dataset)
        ]

    print(f"\n  FreqSafe — Explainability Pipeline")
    print(f"  Device: {device}  |  Experiments: {len(experiments)}\n")

    for model_name, dataset_name, sigma in experiments:
        process_experiment(model_name, dataset_name, sigma, cfg, device, out_root)

    print(f"\n  All done. Outputs -> {out_root}/<model>_<dataset>/explainability/\n")


if __name__ == "__main__":
    main()
