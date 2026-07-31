import copy
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        return F.conv2d(x, (w - m) / torch.sqrt(v + 1e-5),
                        self.bias, self.stride, self.padding, self.dilation, self.groups)


def _conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, 3, stride=stride, padding=1, bias=bias, groups=groups)

def _conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, 1, stride=stride, padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin; cmid = cmid or cout // 4
        self.gn1  = nn.GroupNorm(32, cmid, eps=1e-6); self.conv1 = _conv1x1(cin,  cmid)
        self.gn2  = nn.GroupNorm(32, cmid, eps=1e-6); self.conv2 = _conv3x3(cmid, cmid, stride)
        self.gn3  = nn.GroupNorm(32, cout, eps=1e-6); self.conv3 = _conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or cin != cout:
            self.downsample = _conv1x1(cin, cout, stride)
            self.gn_proj    = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = self.gn_proj(self.downsample(x)) if hasattr(self, "downsample") else x
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        return self.relu(residual + self.gn3(self.conv3(y)))


class ResNetV2(nn.Module):
    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor); self.width = width
        self.root = nn.Sequential(OrderedDict([
            ("conv", StdConv2d(3, width, 7, stride=2, bias=False, padding=3)),
            ("gn",   nn.GroupNorm(32, width, eps=1e-6)),
            ("relu", nn.ReLU(inplace=True))]))
        self.body = nn.Sequential(OrderedDict([
            ("block1", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(width, width * 4, width))] +
                [(f"unit{i}", PreActBottleneck(width * 4, width * 4, width))
                 for i in range(2, block_units[0] + 1)]))),
            ("block2", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(width * 4, width * 8, width * 2, stride=2))] +
                [(f"unit{i}", PreActBottleneck(width * 8, width * 8, width * 2))
                 for i in range(2, block_units[1] + 1)]))),
            ("block3", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(width * 8, width * 16, width * 4, stride=2))] +
                [(f"unit{i}", PreActBottleneck(width * 16, width * 16, width * 4))
                 for i in range(2, block_units[2] + 1)])))]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x); features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, :x.size()[2], :x.size()[3]] = x
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


class ViTAttention(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.vis           = vis
        self.num_heads     = config.transformer["num_heads"]
        self.head_size     = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out   = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.softmax      = nn.Softmax(dim=-1)

    def _transpose(self, x):
        return x.view(x.size()[:-1] + (self.num_heads, self.head_size)).permute(0, 2, 1, 3)

    def forward(self, x):
        q = self._transpose(self.query(x))
        k = self._transpose(self.key(x))
        v = self._transpose(self.value(x))
        scores  = self.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size))
        weights = scores if self.vis else None
        ctx     = self.attn_dropout(scores) @ v
        ctx     = ctx.permute(0, 2, 1, 3).contiguous().view(ctx.size(0), -1, self.all_head_size)
        return self.proj_dropout(self.out(ctx)), weights


class ViTMlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1  = nn.Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2  = nn.Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.drop = nn.Dropout(config.transformer["dropout_rate"])
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6);  nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class ViTEmbeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super().__init__()
        img_size = _pair(img_size)
        self.hybrid = config.patches.get("grid") is not None
        if self.hybrid:
            grid_size       = config.patches["grid"]
            patch_size      = (img_size[0] // 16 // grid_size[0],
                               img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches       = ((img_size[0] // patch_size_real[0]) *
                               (img_size[1] // patch_size_real[1]))
            self.hybrid_model = ResNetV2(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        else:
            patch_size = _pair(config.patches["size"])
            n_patches  = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings    = nn.Conv2d(in_channels, config.hidden_size,
                                             kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout             = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        features = None
        if self.hybrid:
            x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x).flatten(2).transpose(-1, -2)
        return self.dropout(x + self.position_embeddings), features


class ViTBlock(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn  = ViTAttention(config, vis)
        self.ffn   = ViTMlp(config)

    def forward(self, x):
        h = x; x, w = self.attn(self.norm1(x)); x = x + h
        h = x; x    = self.ffn(self.norm2(x));  x = x + h
        return x, w


class ViTEncoder(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.vis          = vis
        self.layer        = nn.ModuleList(
            [copy.deepcopy(ViTBlock(config, vis)) for _ in range(config.transformer["num_layers"])])
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, x):
        weights = []
        for blk in self.layer:
            x, w = blk(x)
            if self.vis: weights.append(w)
        return self.encoder_norm(x), weights


class ViTTransformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super().__init__()
        self.embeddings = ViTEmbeddings(config, img_size)
        self.encoder    = ViTEncoder(config, vis)

    def forward(self, x):
        emb, features = self.embeddings(x)
        enc, weights  = self.encoder(emb)
        return enc, weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0, stride=1, use_batchnorm=True):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                      bias=not use_batchnorm),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_ch + skip_ch, out_ch, 3, padding=1)
        self.conv2 = Conv2dReLU(out_ch, out_ch, 3, padding=1)
        self.up    = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class SegmentationHead(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, upsampling=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity())


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config    = config
        self.conv_more = Conv2dReLU(config.hidden_size, 512, kernel_size=3, padding=1)
        dec_ch  = config.decoder_channels
        in_ch   = [512] + list(dec_ch[:-1])
        skip_ch = list(config.skip_channels)
        for i in range(4 - config.n_skip):
            skip_ch[3 - i] = 0
        self.blocks = nn.ModuleList(
            [DecoderBlock(i, o, s) for i, o, s in zip(in_ch, dec_ch, skip_ch)])

    def forward(self, x, features=None):
        B, n_patch, hidden = x.size()
        h = w = int(np.sqrt(n_patch))
        x = self.conv_more(x.permute(0, 2, 1).contiguous().view(B, hidden, h, w))
        for i, blk in enumerate(self.blocks):
            skip = features[i] if (features is not None and i < self.config.n_skip) else None
            x    = blk(x, skip=skip)
        return x


class TransUNet(nn.Module):
    def __init__(self, config, img_size=256, num_classes=1, vis=False):
        super().__init__()
        self.config      = config
        self.transformer = ViTTransformer(config, img_size, vis)
        self.decoder     = DecoderCup(config)
        self.seg_head    = SegmentationHead(config.decoder_channels[-1], num_classes, kernel_size=3)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x, _, features = self.transformer(x)
        return self.seg_head(self.decoder(x, features))


def _get_r50_b16_config():
    cfg = type("C", (), {})()
    cfg.patches     = {"grid": (16, 16)}
    cfg.hidden_size = 768
    cfg.transformer = {"mlp_dim": 3072, "num_heads": 12, "num_layers": 12,
                       "attention_dropout_rate": 0.0, "dropout_rate": 0.1}
    cfg.classifier  = "seg"
    cfg.resnet      = type("R", (), {})()
    cfg.resnet.num_layers   = (3, 4, 9)
    cfg.resnet.width_factor = 1
    cfg.decoder_channels    = (256, 128, 64, 16)
    cfg.skip_channels       = [512, 256, 64, 16]
    cfg.n_skip              = 3
    return cfg
