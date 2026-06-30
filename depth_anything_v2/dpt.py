import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from .dinov2 import DINOv2
from .dinov3 import DINOv3
from dinov3.models.vision_transformer import DynamicStorageAdapter, StorageGuidedLoRAScaleGenerator
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet
from .dinov2_layers.trans import DWT, IWT



def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv_block(x)



class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0] #(16,1369,768)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))  #(16,768,37,37)

            x = self.projects[i](x) #(16,96,37,37) #(16,192,37,37)  (16,384,37,37) (16,768,37,37)

            x = self.resize_layers[i](x) #(16,96,148,148） #(16,192,74,74)  (16,384,37,37) (16,768,19,19)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out



        layer_1_rn = self.scratch.layer1_rn(layer_1) #(16,128,148,148) 16,128,37,37
        layer_2_rn = self.scratch.layer2_rn(layer_2) #(16,128,74,74) 16,128,37,37
        layer_3_rn = self.scratch.layer3_rn(layer_3) #(16,128,37,37) 16,128,37,37
        layer_4_rn = self.scratch.layer4_rn(layer_4) #(16,128,19,19) 16,128,37,37

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])    #(16,128,37,37) 16,128,37,37
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:]) #(16,128,74,74) 16,128,37,37
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:]) #(16,128,148,148) 16,128,37,37
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn) #(16,128,296,296) 16,128,74,74

        out = self.scratch.output_conv1(path_1) #(16,64,296,296) 16 64,74,74
        out = F.interpolate(out, (int(patch_h * 16), int(patch_w * 16)), mode="bilinear", align_corners=True) #(16,64,512,640)
        out = self.scratch.output_conv2(out) #(16,1,518,518) 16,1,518,518

        return out


# ============================================================
# WaveDPT: Wavelet-based DPT Decoder
# ============================================================
#
def _make_norm(channels, use_bn):
    if use_bn:
        return nn.BatchNorm2d(channels)
    for g in [8, 4, 2, 1]:
        if channels % g == 0:
            return nn.GroupNorm(g, channels)


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
    """
    Wavelet-domain fusion block replacing FeatureFusionBlock (RefineNet).
    """
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
    """
    Output head with wavelet HF refinement before final upsample.
    """
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

        x_up = F.interpolate(
            x_rec,
            (target_h, target_w),
            mode='bilinear',
            align_corners=True
        )

        return self.depth_conv(x_up)


class WaveDPTHead(nn.Module):
    """Wavelet-based DPT decoder for DINOv2 features."""
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
    ):
        super().__init__()

        self.use_clstoken = use_clstoken

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
                nn.Sequential(
                    nn.Linear(2 * in_channels, in_channels),
                    nn.GELU()
                )
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

            x = x.permute(0, 2, 1).reshape(
                x.shape[0], x.shape[-1], patch_h, patch_w
            )

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

        return self.wave_out(p1, patch_h * 16, patch_w * 16)








class DepthAnythingV2(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        max_depth=20.0,
        use_lora=False,
        lora_rank=4,
    ):
        super(DepthAnythingV2, self).__init__()

        self.intermediate_layer_idx = {
            'vits':      [2, 5, 8, 11],
            'vitb':      [2, 5, 8, 11],
            'vitl':      [4, 11, 17, 23],
            'vitg':      [9, 19, 29, 39],
            'vitso400m': [6, 13, 20, 26],
            'vithuge':   [7, 15, 23, 31],
            'vit7b':     [9, 19, 29, 39],
        }

        self.max_depth = max_depth

        self.encoder = encoder
        self.pretrained = DINOv3(model_name=encoder)
        # self.encoder = encoder
        # self.pretrained = DINOv2(model_name=encoder)
        #
        if use_lora:
            self.pretrained.enable_dynamic_lora(rank=lora_rank)

        print(f"DPTHead features: {features}")

        self.depth_head = WaveDPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

        #self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)


    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // self.pretrained.patch_size, x.shape[-1] // self.pretrained.patch_size

        features = self.pretrained.get_intermediate_layers(x, n=self.intermediate_layer_idx[self.encoder], return_class_token=True)

        depth = self.depth_head(features, patch_h, patch_w) * self.max_depth
        #depth = F.interpolate(depth, x.shape[-2:], mode='bilinear', align_corners=True)

        return depth.squeeze(1)

    @torch.no_grad()
    def infer_image(self, raw_image, input_height=476, input_width=588):
        image, (h, w) = self.image2tensor(raw_image, input_height, input_width)

        depth = self.forward(image)

        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]

        return depth.cpu().numpy()

    def image2tensor(self, raw_image, input_height=476, input_width=588):
        transform = Compose([
            Resize(
                width=input_width,
                height=input_height,
                resize_target=False,
                keep_aspect_ratio=False,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        h, w = raw_image.shape[:2]

        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)

        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)

        return image, (h, w)
