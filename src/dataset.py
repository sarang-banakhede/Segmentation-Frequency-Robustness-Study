from __future__ import annotations

import os
import random
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else _BILINEAR
_NEAREST  = Image.Resampling.NEAREST  if hasattr(Image, "Resampling") else _NEAREST


def _norm(tensor: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    m = torch.tensor(mean, dtype=torch.float32, device=tensor.device).view(-1, 1, 1)
    s = torch.tensor(std,  dtype=torch.float32, device=tensor.device).view(-1, 1, 1)
    return (tensor - m) / s


class MedicalSegDataset(Dataset):
    def __init__(
        self,
        image_dir:   str,
        mask_dir:    str,
        in_channels: int         = 3,
        img_size:    int         = 256,
        norm_mean:   list[float] | None = None,
        norm_std:    list[float] | None = None,
        aug_cfg:     dict        | None = None,
    ):
        self.mask_dir    = mask_dir
        self.in_channels = in_channels
        self.img_size    = img_size
        self.norm_mean   = norm_mean
        self.norm_std    = norm_std
        self.aug_cfg     = aug_cfg

        exts = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
        paths: list[str] = []
        for e in exts:
            paths.extend(glob(os.path.join(image_dir, e)))
        self.image_paths = sorted(set(paths))

        if not self.image_paths:
            raise RuntimeError(f"No images found in '{image_dir}'")
        print(f"  [Dataset] {len(self.image_paths):>5} images  ←  {image_dir}  "
              f"({'augmented' if aug_cfg else 'no aug'})")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path  = self.image_paths[idx]
        stem      = Path(img_path).stem
        mask_path = self._find_mask(stem)

        mode  = "L" if self.in_channels == 1 else "RGB"
        image = Image.open(img_path).convert(mode).resize(
            (self.img_size, self.img_size), _BILINEAR)
        mask  = Image.open(mask_path).convert("L").resize(
            (self.img_size, self.img_size), _NEAREST)

        if self.aug_cfg:
            image, mask = self._augment(image, mask)

        img_np  = np.array(image, dtype=np.float32) / 255.0
        mask_np = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)

        img_t = torch.from_numpy(img_np)
        img_t = img_t.unsqueeze(0) if self.in_channels == 1 else img_t.permute(2, 0, 1)

        if self.norm_mean and self.norm_std:
            img_t = _norm(img_t, self.norm_mean, self.norm_std)

        return img_t, torch.from_numpy(mask_np).unsqueeze(0)

    def _find_mask(self, stem: str) -> str:
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
            p = os.path.join(self.mask_dir, stem + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Mask not found for '{stem}' in {self.mask_dir}")

    def _augment(self, image: Image.Image, mask: Image.Image):
        cfg = self.aug_cfg
        if cfg.get("hflip") and random.random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if cfg.get("vflip") and random.random() < 0.5:
            image, mask = TF.vflip(image), TF.vflip(mask)
        rot = float(cfg.get("rotation_deg", 0))
        if rot > 0 and random.random() < 0.5:
            angle = random.uniform(-rot, rot)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mask  = TF.rotate(mask,  angle, interpolation=InterpolationMode.NEAREST,  fill=0)
        brt = float(cfg.get("brightness", 0.0))
        ctr = float(cfg.get("contrast",   0.0))
        if (brt > 0 or ctr > 0) and random.random() < 0.5:
            if brt > 0:
                image = TF.adjust_brightness(image, random.uniform(1 - brt, 1 + brt))
            if ctr > 0:
                image = TF.adjust_contrast(image, random.uniform(1 - ctr, 1 + ctr))
        return image, mask


class TestDataset(Dataset):
    def __init__(
        self,
        image_dir:   str,
        mask_dir:    str,
        in_channels: int          = 3,
        img_size:    int          = 256,
        norm_mean:   list[float] | None = None,
        norm_std:    list[float] | None = None,
        normalize:   bool         = True,
    ):
        self.mask_dir    = mask_dir
        self.in_channels = in_channels
        self.img_size    = img_size
        self.norm_mean   = norm_mean
        self.norm_std    = norm_std
        self.normalize   = normalize

        exts = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]
        paths: list[str] = []
        for e in exts:
            paths.extend(glob(os.path.join(image_dir, e)))
        self.image_paths = sorted(set(paths))

        if not self.image_paths:
            raise RuntimeError(f"No images found in '{image_dir}'")
        print(f"  [TestDataset] {len(self.image_paths)} images  ←  {image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path  = self.image_paths[idx]
        stem      = Path(img_path).stem
        mask_path = self._find_mask(stem)

        mode  = "L" if self.in_channels == 1 else "RGB"
        image = Image.open(img_path).convert(mode).resize(
            (self.img_size, self.img_size), _BILINEAR)
        mask  = Image.open(mask_path).convert("L").resize(
            (self.img_size, self.img_size), _NEAREST)

        img_np  = np.array(image, dtype=np.float32) / 255.0
        mask_np = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)

        img_t = torch.from_numpy(img_np)
        img_t = img_t.unsqueeze(0) if self.in_channels == 1 else img_t.permute(2, 0, 1)

        if self.normalize and self.norm_mean and self.norm_std:
            img_t = _norm(img_t, self.norm_mean, self.norm_std)

        return img_t, torch.from_numpy(mask_np).unsqueeze(0)

    def _find_mask(self, stem: str) -> str:
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
            p = os.path.join(self.mask_dir, stem + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Mask not found for '{stem}' in {self.mask_dir}")
