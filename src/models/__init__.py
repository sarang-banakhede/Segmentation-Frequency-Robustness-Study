import torch
import torch.nn as nn

from .unet import UNet
from .transunet import TransUNet
from .swin_unet import SwinTransformerSys, _window_partition, _window_reverse


def build_model(name: str, in_ch: int, img_size: int = 256) -> nn.Module:
    name = name.lower()
    if name == "unet":
        return UNet(in_channels=in_ch, out_channels=1)
    if name == "transunet":
        return TransUNet(img_size=img_size, num_classes=1)
    if name == "swin":
        return SwinTransformerSys(
            img_size=img_size, patch_size=4, in_chans=in_ch, num_classes=1,
            embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
            window_size=8)
    raise ValueError(f"Unknown model '{name}'. Choose: unet | transunet | swin")


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> dict:
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return ckpt
