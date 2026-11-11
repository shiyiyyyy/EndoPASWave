"""
WaveDPTHead: wavelet-based DPT decoder.

Self-contained copy of the decoder used in depth_anything_v2/dpt.py
(WaveBand / WaveFusionBlock / WaveOutputHead / WaveDPTHead), together with
its dependencies (Haar DWT/IWT, norm helper, scratch layers).
Parameter names are identical to the original, so checkpoints load as-is.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Haar wavelet transform
# ============================================================
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, x_HL, x_LH, x_HH


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width], device=x.device, dtype=x.dtype)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def forward(self, x):
        return iwt_init(x)


# ============================================================
# Helpers
# ============================================================
def _make_norm(channels, use_bn):
    if use_bn:
        return nn.BatchNorm2d(channels)
    for g in [8, 4, 2, 1]:
        if channels % g == 0:
            return nn.GroupNorm(g, channels)


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch


# ============================================================
# Wavelet blocks
# ============================================================
class WaveBand(nn.Module):
    """Single wavelet sub-band processor with direction-aware convolution."""
    def __init__(self, channels, direction='iso', use_bn=False):
        super().__init__()
        if direction == 'HL':
            k, p = (3, 1), (1, 0)
        elif direction == 'LH':
            k, p = (1, 3), (0, 1)
        else:
            k, p = (3, 3), (1, 1)

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, k, padding=p, bias=False),
            _make_norm(channels, use_bn),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(self, x):
        return x + self.conv(x)


class WaveFusionBlock(nn.Module):
    """Wavelet-domain fusion block replacing FeatureFusionBlock (RefineNet)."""
    def __init__(self, features, use_bn=False):
        super().__init__()
        self.dwt = DWT()
        self.iwt = IWT()

        self.ll_proc = WaveBand(features, 'iso', use_bn)
        self.hl_proc = WaveBand(features, 'HL', use_bn)
        self.lh_proc = WaveBand(features, 'LH', use_bn)
        self.hh_scale = nn.Parameter(torch.ones(1) * 0.1)

        self.ll_guide = nn.Conv2d(features, features, 1, bias=False)

        self.skip_proj = nn.Sequential(
            nn.Conv2d(features, features, 1, bias=False),
            _make_norm(features, use_bn),
        )
        self.main_proj = nn.Sequential(
            nn.Conv2d(features, features, 1, bias=False),
            _make_norm(features, use_bn),
        )

        self.fuse_act = nn.GELU()
        self.residual_proj = nn.Conv2d(features, features, 1, bias=False)

    def _wave_refine(self, x):
        B, C, H, W = x.shape
        pad_h, pad_w = H % 2, W % 2
        xp = F.pad(x, (0, pad_w, 0, pad_h), 'reflect') if (pad_h or pad_w) else x

        ll, hl, lh, hh = self.dwt(xp)
        ll = self.ll_proc(ll)

        guide = self.ll_guide(ll)
        hl = self.hl_proc(hl + guide)
        lh = self.lh_proc(lh + guide)
        hh = self.hh_scale * hh

        out = self.iwt(torch.cat([ll, hl, lh, hh], dim=1))

        if pad_h or pad_w:
            out = out[:, :, :H, :W]

        return out

    def forward(self, x, skip=None, size=None):
        if skip is not None:
            x = self.fuse_act(self.main_proj(x) + self.skip_proj(skip))

        out = self._wave_refine(x) + self.residual_proj(x)

        if size is not None:
            out = F.interpolate(out, size, mode='bilinear', align_corners=True)

        return out


class WaveOutputHead(nn.Module):
    """Output head with wavelet HF refinement before final upsample."""
    def __init__(self, features, use_bn=False):
        super().__init__()
        self.dwt = DWT()
        self.iwt = IWT()

        mid = features // 2

        self.pre_conv = nn.Sequential(
            nn.Conv2d(features, mid, 3, padding=1, bias=False),
            _make_norm(mid, use_bn),
            nn.GELU(),
        )

        self.hl_refine = nn.Sequential(
            nn.Conv2d(mid, mid, (3, 1), padding=(1, 0), bias=False),
            _make_norm(mid, use_bn),
            nn.GELU(),
        )

        self.lh_refine = nn.Sequential(
            nn.Conv2d(mid, mid, (1, 3), padding=(0, 1), bias=False),
            _make_norm(mid, use_bn),
            nn.GELU(),
        )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(mid, 32, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, target_h, target_w):
        x = self.pre_conv(x)

        B, C, H, W = x.shape
        pad_h, pad_w = H % 2, W % 2
        xp = F.pad(x, (0, pad_w, 0, pad_h), 'reflect') if (pad_h or pad_w) else x

        ll, hl, lh, hh = self.dwt(xp)

        hl = hl + self.hl_refine(hl)
        lh = lh + self.lh_refine(lh)

        x_rec = self.iwt(torch.cat([ll, hl, lh, hh], dim=1))

        if pad_h or pad_w:
            x_rec = x_rec[:, :, :H, :W]

        x_up = F.interpolate(x_rec, (target_h, target_w), mode='bilinear', align_corners=True)

        return self.depth_conv(x_up)


# ============================================================
# WaveDPTHead
# ============================================================
class WaveDPTHead(nn.Module):
    """Wavelet-based DPT decoder for ViT features.

    Args:
        patch_size: backbone patch size used for the final upsample
                    (16 for DINOv3, 14 for DINOv2). Original code hard-codes 16.
    """
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        patch_size=16,
    ):
        super().__init__()

        self.use_clstoken = use_clstoken
        self.patch_size = patch_size

        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, out_channel, 1)
            for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], 3, stride=2, padding=1),
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList([
                nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                for _ in range(4)
            ])

        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)

        self.wave4 = WaveFusionBlock(features, use_bn)
        self.wave3 = WaveFusionBlock(features, use_bn)
        self.wave2 = WaveFusionBlock(features, use_bn)
        self.wave1 = WaveFusionBlock(features, use_bn)

        self.wave_out = WaveOutputHead(features, use_bn)

    def forward(self, out_features, patch_h, patch_w):
        out = []

        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls = x[0], x[1]
                readout = cls.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        l1, l2, l3, l4 = out

        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        p4 = self.wave4(l4_rn, size=l3_rn.shape[2:])
        p3 = self.wave3(p4, skip=l3_rn, size=l2_rn.shape[2:])
        p2 = self.wave2(p3, skip=l2_rn, size=l1_rn.shape[2:])
        p1 = self.wave1(p2, skip=l1_rn)

        return self.wave_out(p1, patch_h * self.patch_size, patch_w * self.patch_size)
