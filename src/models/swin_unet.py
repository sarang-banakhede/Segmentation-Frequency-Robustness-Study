import torch
import torch.nn as nn
from einops import rearrange


def _to_2tuple(x):
    return (x, x) if isinstance(x, int) else x


def _trunc_normal_(t, mean=0., std=1.):
    with torch.no_grad():
        return t.normal_(mean, std)


def _window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def _window_reverse(windows, ws, H, W):
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class SwinDropPath(nn.Module):
    def __init__(self, p=0.):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0. or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x.div(kp) * (kp + torch.rand(shape, dtype=x.dtype, device=x.device)).floor_()


class SwinMlp(nn.Module):
    def __init__(self, in_f, hid_f=None, out_f=None, drop=0.):
        super().__init__()
        out_f = out_f or in_f
        hid_f = hid_f or in_f
        self.fc1 = nn.Linear(in_f, hid_f)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_f, out_f)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class WindowAttention(nn.Module):
    def __init__(self, dim, ws, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.ws = ws
        self.num_heads = num_heads
        hd = dim // num_heads
        self.scale = qk_scale or hd ** -0.5
        self.rpbt = nn.Parameter(torch.zeros((2 * ws[0] - 1) * (2 * ws[1] - 1), num_heads))
        ch = torch.arange(ws[0])
        cw = torch.arange(ws[1])
        coords = torch.stack(torch.meshgrid([ch, cw], indexing="ij"))
        cf = torch.flatten(coords, 1)
        rc = cf[:, :, None] - cf[:, None, :]
        rc = rc.permute(1, 2, 0).contiguous()
        rc[:, :, 0] += ws[0] - 1
        rc[:, :, 1] += ws[1] - 1
        rc[:, :, 0] *= 2 * ws[1] - 1
        self.register_buffer("rpi", rc.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        _trunc_normal_(self.rpbt, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        od = x.dtype
        qkv = self.qkv(x).float().reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        rb = self.rpbt[self.rpi.view(-1)].view(self.ws[0] * self.ws[1], self.ws[0] * self.ws[1], -1)
        attn = attn + rb.permute(2, 0, 1).contiguous().float().unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0).float()
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).to(od).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, res, num_heads, ws=7, shift=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., dp=0.):
        super().__init__()
        self.dim = dim
        self.res = res
        self.ws = ws
        self.shift = shift
        if min(res) <= ws:
            self.shift = 0
            self.ws = min(res)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, _to_2tuple(self.ws), num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.dp = SwinDropPath(dp) if dp > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwinMlp(dim, int(dim * mlp_ratio), drop=drop)
        if self.shift > 0:
            H, W = res
            img_mask = torch.zeros(1, H, W, 1)
            h_slices = (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None))
            w_slices = (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mw = _window_partition(img_mask, self.ws).view(-1, self.ws * self.ws)
            attn_mask = mw.unsqueeze(1) - mw.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.).masked_fill(attn_mask == 0, 0.)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.res
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        xw = _window_partition(x, self.ws).view(-1, self.ws * self.ws, C)
        aw = self.attn(xw, mask=self.attn_mask)
        x = _window_reverse(aw.view(-1, self.ws, self.ws, C), self.ws, H, W).view(B, H * W, C)
        if self.shift > 0:
            x = torch.roll(x.view(B, H, W, C), shifts=(self.shift, self.shift), dims=(1, 2)).view(B, H * W, C)
        x = shortcut + self.dp(x)
        x = x + self.dp(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, res, dim):
        super().__init__()
        self.res = res
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        H, W = self.res
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        return self.reduction(self.norm(torch.cat([x0, x1, x2, x3], -1))).view(B, -1, 2 * C)


class SwinBasicLayer(nn.Module):
    def __init__(self, dim, res, depth, num_heads, ws, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., dp=0., downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim, res, num_heads, ws, shift=0 if i % 2 == 0 else ws // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                dp=dp[i] if isinstance(dp, list) else dp)
            for i in range(depth)])
        self.downsample = downsample(res, dim) if downsample else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample:
            x = self.downsample(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self, res, dim, dim_scale=2):
        super().__init__()
        self.res = res
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = nn.LayerNorm(dim // dim_scale)

    def forward(self, x):
        H, W = self.res
        x = self.expand(x)
        x = x.view(-1, H, W, 4 * (self.dim // 2))
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=self.dim // 2)
        return self.norm(x.view(-1, H * 2 * W * 2, self.dim // 2))


class FinalPatchExpand(nn.Module):
    def __init__(self, res, dim, patch_size=4):
        super().__init__()
        self.res = res
        self.dim = dim
        self.ps = patch_size
        self.expand = nn.Linear(dim, patch_size ** 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        H, W = self.res
        x = self.expand(x)
        x = x.view(-1, H, W, self.ps ** 2 * self.dim)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.ps, p2=self.ps, c=self.dim)
        return self.norm(x)


class SwinUNetDecoder(nn.Module):
    def __init__(self, embed_dim, depths, num_heads, ws, mlp_ratio, qkv_bias, qk_scale,
                 drop, attn_drop, dp_rates, norm_layer, patches_resolution, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.concat_linears = nn.ModuleList()
        num_dec = num_layers // 2

        bottleneck_res = patches_resolution[0] // (2 ** (num_layers - 1))
        cur_dim = int(embed_dim * 2 ** (num_layers - 1))

        for i in range(num_dec):
            skip_dim = int(embed_dim * 2 ** (num_layers - 1 - i))
            out_dim = skip_dim // 2
            res_side = bottleneck_res * (2 ** i)
            layer_res = (res_side, res_side)
            has_expand = i < (num_dec - 1)

            self.concat_linears.append(nn.Linear(cur_dim + skip_dim, out_dim))
            self.layers.append(SwinBasicLayer(
                out_dim, layer_res,
                depths[num_dec - 1 - i],
                num_heads[num_dec - 1 - i],
                ws, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                dp_rates[sum(depths[:num_dec - 1 - i]):sum(depths[:num_dec - i])],
                downsample=PatchExpand if has_expand else None))

            cur_dim = out_dim // 2 if has_expand else out_dim

    def forward(self, x, skips):
        for i, (layer, cl) in enumerate(zip(self.layers, self.concat_linears)):
            x = cl(torch.cat([x, skips[i]], -1))
            x = layer(x, 0, 0)
        return x


class SwinTransformerSys(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=3, num_classes=1,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., dp=0.1, norm_layer=nn.LayerNorm, patch_norm=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_norm_layer = norm_layer(embed_dim) if patch_norm else nn.Identity()

        patches_resolution = (img_size // patch_size, img_size // patch_size)
        self.patches_resolution = patches_resolution

        total_depth = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, dp, total_depth)]

        self.enc_layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer_res = (patches_resolution[0] // (2 ** i), patches_resolution[1] // (2 ** i))
            layer_dim = int(embed_dim * 2 ** i)
            self.enc_layers.append(SwinBasicLayer(
                layer_dim, layer_res, depths[i], num_heads[i], window_size,
                mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                dp_rates[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=PatchMerging if i < self.num_layers - 1 else None))
        self.norm = norm_layer(self.num_features)

        self.decoder = SwinUNetDecoder(
            embed_dim, depths, num_heads, window_size, mlp_ratio,
            qkv_bias, qk_scale, drop, attn_drop, dp_rates, norm_layer,
            patches_resolution, self.num_layers)

        num_dec = self.num_layers // 2
        bottleneck_res = patches_resolution[0] // (2 ** (self.num_layers - 1))
        decoder_out_res = bottleneck_res * (2 ** (num_dec - 1))
        decoder_out_dim = int(embed_dim * 2 ** (num_dec - 1))

        self.final_expand = FinalPatchExpand(
            (decoder_out_res, decoder_out_res), decoder_out_dim, patch_size)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=4)
        self.head = nn.Conv2d(decoder_out_dim, num_classes, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        ps = self.final_expand.ps
        x = self.patch_norm_layer(self.patch_embed(x).flatten(2).transpose(1, 2))
        skips = []
        cur = x
        for layer in self.enc_layers:
            skips.append(cur)
            cur = layer(cur, 0, 0)
        cur = self.norm(cur)
        cur = self.decoder(cur, list(reversed(skips[1:])))
        B, L, C = cur.shape
        side = int(L ** 0.5)
        cur = self.final_expand(cur)
        cur = cur.view(B, side * ps, side * ps, C).permute(0, 3, 1, 2).contiguous()
        cur = self.upsample(cur)
        return self.head(cur)
