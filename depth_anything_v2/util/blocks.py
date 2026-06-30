import torch.nn as nn


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


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)

        output = self.out_conv(output)

        return output
# class FeatureFusionBlock(nn.Module):
#     """Feature fusion block with Pixel Shuffle upsampling for sharper edges."""
#
#     def __init__(
#             self,
#             features,
#             activation,
#             deconv=False,
#             bn=False,
#             expand=False,
#             align_corners=True,
#             size=None,
#             use_pixel_shuffle=True,  # 新增：控制是否用pixel shuffle
#     ):
#         super(FeatureFusionBlock, self).__init__()
#
#         self.deconv = deconv
#         self.align_corners = align_corners
#         self.groups = 1
#         self.expand = expand
#         self.use_pixel_shuffle = use_pixel_shuffle
#
#         out_features = features
#         if self.expand == True:
#             out_features = features // 2
#
#         self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
#
#         self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
#         self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
#
#         self.skip_add = nn.quantized.FloatFunctional()
#         self.size = size
#
#         # ========== 新增：Pixel Shuffle 上采样模块 ==========
#         if self.use_pixel_shuffle:
#             # 1x1 conv 把通道扩4倍，然后pixel shuffle拆到空间维度
#             self.ps_conv = nn.Conv2d(features, features * 4, kernel_size=1, stride=1, padding=0, bias=True)
#             self.pixel_shuffle = nn.PixelShuffle(2)
#             # 上采样后再做一次3x3 refine，稳定输出
#             self.ps_refine = nn.Sequential(
#                 nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True),
#                 activation,
#             )
#         # ===================================================
#
#     def forward(self, *xs, size=None):
#         output = xs[0]
#
#         if len(xs) == 2:
#             base_res = self.resConfUnit1(xs[1])
#             output = self.skip_add.add(output, base_res)
#
#         output = self.resConfUnit2(output)
#
#         # ========== 上采样逻辑 ==========
#         if self.use_pixel_shuffle and size is None and self.size is None:
#             # 2倍上采样：走pixel shuffle路径
#             output = self.ps_conv(output)
#             output = self.pixel_shuffle(output)
#             output = self.ps_refine(output)
#         else:
#             # 指定size的情况（比如最后一层要对齐到特定分辨率），仍用bilinear
#             if size is None and self.size is None:
#                 modifier = {"scale_factor": 2}
#             elif size is None:
#                 modifier = {"size": self.size}
#             else:
#                 modifier = {"size": size}
#
#             output = nn.functional.interpolate(
#                 output, **modifier, mode="bilinear", align_corners=self.align_corners
#             )
#         # ==================================
#
#         output = self.out_conv(output)
#         return output
