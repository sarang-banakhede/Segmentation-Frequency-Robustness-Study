import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.modules.utils import _pair


def _to_2tuple(x):
    return (x, x) if isinstance(x, int) else x


def _trunc_normal_(tensor, mean=0., std=1.):
    with torch.no_grad():
        return tensor.normal_(mean, std)


def _window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def _window_reverse(windows, ws, H, W):
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x.div(keep_prob) * (keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)).floor_()


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords   = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten  = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        _trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax   = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        orig_dtype = x.dtype
        qkv = self.qkv(x).float().reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        attn = attn + rel_bias.permute(2, 0, 1).contiguous().float().unsqueeze(0)
        if mask is not None:
            nW   = mask.shape[0]
            attn = (attn.view(B_ // nW, nW, self.num_heads, N, N)
                    + mask.unsqueeze(1).unsqueeze(0).float()).view(-1, self.num_heads, N, N)
        attn = self.attn_drop(self.softmax(attn))
        x    = (attn @ v).to(orig_dtype).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim              = dim
        self.input_resolution = input_resolution
        self.window_size      = window_size
        self.shift_size       = shift_size
        if min(input_resolution) <= window_size:
            self.shift_size  = 0
            self.window_size = min(input_resolution)
        self.norm1     = norm_layer(dim)
        self.attn      = WindowAttention(
            dim, window_size=_to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        self.mlp       = Mlp(dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        if self.shift_size > 0:
            H, W     = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in h_slices:
                    img_mask[:, h, w, :] = cnt; cnt += 1
            mw = _window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
            attn_mask = mw.unsqueeze(1) - mw.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W    = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        shifted_x = torch.roll(x, (-self.shift_size, -self.shift_size), (1, 2)) if self.shift_size > 0 else x
        x_windows    = _window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        shifted_x    = _window_reverse(attn_windows.view(-1, self.window_size, self.window_size, C),
                                       self.window_size, H, W)
        x = (torch.roll(shifted_x, (self.shift_size, self.shift_size), (1, 2)) if self.shift_size > 0 else shifted_x)
        x = shortcut + self.drop_path(x.view(B, L, C))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim       = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm      = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                        x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.reduction(self.norm(x.view(B, -1, 4 * C)))


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim    = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm   = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        x = rearrange(x.view(B, H, W, C), "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        return self.norm(x.view(B, -1, C // 4))


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim       = dim
        self.dim_scale = dim_scale
        self.expand    = nn.Linear(dim, 16 * dim, bias=False)
        self.norm      = norm_layer(dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        x = rearrange(x.view(B, H, W, C),
                      "b h w (p1 p2 c) -> b (h p1) (w p2) c",
                      p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        return self.norm(x.view(B, -1, self.dim))


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample:
            x = self.downsample(x)
        return x


class BasicLayer_up(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])
        self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2,
                                    norm_layer=norm_layer) if upsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size   = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm:
            x = self.norm(x)
        return x


class SwinTransformerSys(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1,
                 embed_dim=96, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 final_upsample="expand_first", **kwargs):
        super().__init__()
        self.num_layers     = len(depths)
        self.embed_dim      = embed_dim
        self.ape            = ape
        self.num_features   = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio      = mlp_ratio
        self.final_upsample = final_upsample

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None)
        patches_resolution      = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches, embed_dim))
            _trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList([
            BasicLayer(
                dim=int(embed_dim * 2 ** i),
                input_resolution=(patches_resolution[0] // 2 ** i,
                                   patches_resolution[1] // 2 ** i),
                depth=depths[i], num_heads=num_heads[i], window_size=window_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i < self.num_layers - 1 else None)
            for i in range(self.num_layers)])

        self.layers_up       = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i in range(self.num_layers):
            idx = self.num_layers - 1 - i
            self.concat_back_dim.append(
                nn.Linear(2 * int(embed_dim * 2 ** idx), int(embed_dim * 2 ** idx))
                if i > 0 else nn.Identity())
            if i == 0:
                layer_up = PatchExpand(
                    (patches_resolution[0] // 2 ** idx, patches_resolution[1] // 2 ** idx),
                    dim=int(embed_dim * 2 ** idx), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(
                    dim=int(embed_dim * 2 ** idx),
                    input_resolution=(patches_resolution[0] // 2 ** idx,
                                       patches_resolution[1] // 2 ** idx),
                    depth=depths[idx], num_heads=num_heads[idx],
                    window_size=window_size, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:idx]):sum(depths[:idx + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if i < self.num_layers - 1 else None)
            self.layers_up.append(layer_up)

        self.norm    = norm_layer(self.num_features)
        self.norm_up = norm_layer(embed_dim)

        if final_upsample == "expand_first":
            self.up = FinalPatchExpand_X4(
                (img_size // patch_size, img_size // patch_size),
                dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(embed_dim, num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
        x = self.norm(x)
        for i, layer_up in enumerate(self.layers_up):
            if i == 0:
                x = layer_up(x)
            else:
                x = self.concat_back_dim[i](torch.cat([x, x_downsample[3 - i]], -1))
                x = layer_up(x)
        x = self.norm_up(x)
        H, W = self.patches_resolution
        B, L, C = x.shape
        x = self.up(x)
        x = x.view(B, 4 * H, 4 * W, -1).permute(0, 3, 1, 2)
        return self.output(x)
