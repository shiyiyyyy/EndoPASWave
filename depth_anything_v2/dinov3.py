import sys
import os

_dinov3_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dinov3')
if _dinov3_root not in sys.path:
    sys.path.insert(0, _dinov3_root)

from dinov3.models.vision_transformer import vit_small, vit_base, vit_large, vit_so400m, vit_huge2, vit_giant2, vit_7b


def DINOv3(model_name):
    # vits config inferred from dinov3_vits16_pretrain_lvd1689m checkpoint:
    # n_storage_tokens=4, layerscale_init=1e-5, mask_k_bias=True
    model_zoo = {
        'vits':      lambda: vit_small(patch_size=16, n_storage_tokens=4, layerscale_init=1e-5, mask_k_bias=True),
        'vitb':      lambda: vit_base(patch_size=16, n_storage_tokens=4, layerscale_init=1e-5, mask_k_bias=True),
        'vitl':      lambda: vit_large(patch_size=16),
        'vitg':      lambda: vit_giant2(patch_size=16),
        'vitso400m': lambda: vit_so400m(patch_size=16),
        'vithuge':   lambda: vit_huge2(patch_size=16),
        'vit7b':     lambda: vit_7b(patch_size=16),
    }
    return model_zoo[model_name]()
