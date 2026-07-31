from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v = w.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
        return F.conv2d(
            x, (w - w.mean(dim=[1, 2, 3], keepdim=True)) / torch.sqrt(v + 1e-5),
            self.bias, self.stride, self.padding, self.dilation, self.groups)


class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4
        self.gn1 = nn.GroupNorm(min(32, cmid), cmid)
        self.conv1 = StdConv2d(cin, cmid, 1, bias=False)
        self.gn2 = nn.GroupNorm(min(32, cmid), cmid)
        self.conv2 = StdConv2d(cmid, cmid, 3, stride=stride, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(min(32, cout), cout)
        self.conv3 = StdConv2d(cmid, cout, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        if cin != cout or stride != 1:
            self.downsample = nn.Sequential(
                StdConv2d(cin, cout, 1, stride=stride, bias=False),
                nn.GroupNorm(min(32, cout), cout))
        else:
            self.downsample = None

    def forward(self, x):
        r = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        return self.relu(self.gn3(self.conv3(y)) + r)


class ResNetV2(nn.Module):
    def __init__(self, block_units, width_factor):
        super().__init__()
        wf = width_factor
        self.width = 16 * wf
        root_ch = 16 * wf
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, root_ch, 7, stride=2, padding=3, bias=False)),
            ('gn', nn.GroupNorm(min(32, root_ch), root_ch)),
            ('relu', nn.ReLU(inplace=True))]))
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(16 * wf, 64 * wf, 16 * wf))] +
                [(f'unit{i}', PreActBottleneck(64 * wf, 64 * wf, 16 * wf))
                 for i in range(2, block_units[0] + 1)]))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(64 * wf, 128 * wf, 32 * wf, stride=2))] +
                [(f'unit{i}', PreActBottleneck(128 * wf, 128 * wf, 32 * wf))
                 for i in range(2, block_units[1] + 1)]))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(128 * wf, 256 * wf, 64 * wf, stride=2))] +
                [(f'unit{i}', PreActBottleneck(256 * wf, 256 * wf, 64 * wf))
                 for i in range(2, block_units[2] + 1)]))),
        ]))

    def forward(self, x):
        x = self.root(x)
        features = []
        children = list(self.body.children())
        for i, blk in enumerate(children):
            x = blk(x)
            if i != len(children) - 1:
                features.append(x)
        return x, features


class ViTAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        nh = cfg['num_heads']
        hs = cfg['hidden_size'] // nh
        self.nh = nh
        self.hs = hs
        self.qkv = nn.Linear(cfg['hidden_size'], cfg['hidden_size'] * 3)
        self.proj = nn.Linear(cfg['hidden_size'], cfg['hidden_size'])
        self.drop = nn.Dropout(cfg['attn_drop'])

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.nh, self.hs).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.drop(F.softmax((q @ k.transpose(-2, -1)) * (self.hs ** -0.5), dim=-1))
        return (attn @ v).transpose(1, 2).reshape(B, N, C)


class ViTMlp(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg['hidden_size'], cfg['mlp_dim'])
        self.fc2 = nn.Linear(cfg['mlp_dim'], cfg['hidden_size'])
        self.drop = nn.Dropout(cfg['drop'])

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class ViTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg['hidden_size'], eps=1e-6)
        self.norm2 = nn.LayerNorm(cfg['hidden_size'], eps=1e-6)
        self.attn = ViTAttention(cfg)
        self.ffn = ViTMlp(cfg)
        self.use_checkpoint = False

    def _fwd(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))

    def forward(self, x):
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._fwd, x, use_reentrant=False)
        return self._fwd(x)


class ViTEmbeddings(nn.Module):
    def __init__(self, cfg, img_size):
        super().__init__()
        feat_map = img_size // 8
        ps = (1, 1)
        n_patches = feat_map * feat_map
        self.hybrid = ResNetV2((3, 4, 9), 1)
        self.patch_emb = nn.Conv2d(256, cfg['hidden_size'], kernel_size=ps, stride=ps)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_patches, cfg['hidden_size']))
        self.drop = nn.Dropout(cfg['drop'])

    def forward(self, x):
        x, feats = self.hybrid(x)
        x = self.patch_emb(x).flatten(2).transpose(-1, -2)
        return self.drop(x + self.pos_emb), feats


class TransUNetDecBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=0):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.c1 = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.c2(self.c1(x))


class TransUNet(nn.Module):
    def __init__(self, img_size=256, num_classes=1):
        super().__init__()
        cfg = {'hidden_size': 768, 'num_heads': 12, 'mlp_dim': 3072,
               'attn_drop': 0.0, 'drop': 0.1, 'num_layers': 12}
        self.emb = ViTEmbeddings(cfg, img_size)
        self.encoder = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg['num_layers'])])
        for blk in self.encoder:
            blk.use_checkpoint = True
        self.enc_norm = nn.LayerNorm(cfg['hidden_size'], eps=1e-6)
        self.conv_more = nn.Sequential(
            nn.Conv2d(768, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        skip_chs = [0, 0, 0]
        dec_chs = [256, 128, 16]
        in_chs = [512, 256, 128]
        self.dec_blocks = nn.ModuleList(
            [TransUNetDecBlock(i, o, s) for i, o, s in zip(in_chs, dec_chs, skip_chs)])
        self.seg_head = nn.Conv2d(dec_chs[-1], num_classes, 3, padding=1)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x, feats = self.emb(x)
        for blk in self.encoder:
            x = blk(x)
        x = self.enc_norm(x)
        B, N, C = x.shape
        h = w = int(N ** 0.5)
        x = self.conv_more(x.permute(0, 2, 1).contiguous().view(B, C, h, w))
        for blk in self.dec_blocks:
            x = blk(x, None)
        return self.seg_head(x)
