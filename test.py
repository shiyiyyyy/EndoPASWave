import argparse
import logging
import os
import pprint
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from depth_anything_v2.dpt import DepthAnythingV2
from util.dist_helper import setup_distributed
from util.metric import eval_depth
from util.utils import init_log
from dataset.c3vd import C3VD
from dataset.scared import SCARED

# ------------------ Args ------------------
parser = argparse.ArgumentParser("Depth Anything V2 Val Only")

parser.add_argument('--encoder', default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg', 'vitso400m', 'vithuge', 'vit7b'])
parser.add_argument('--dataset', default='c3vd', choices=['hypersim', 'scared','c3vd'])
parser.add_argument('--img-height', default=512, type=int)
parser.add_argument('--img-width', default=640, type=int)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--max-depth', default=100, type=float)
#parser.add_argument('--pretrained-from',default="./run/seed/lora_free_qkv_fc/40_epotwo_01/latest_model.pth",type=str)
parser.add_argument('--checkpoint', default="results/c3vd/new_size_base/latest_model.pth", type=str)
#parser.add_argument('--checkpoint', default="./results/all/latest_model.pth", type=str)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument("--lora_type",default="None",choices=["lora","None"],type=str,help="whether lora use for the model")
parser.add_argument("--head_type",default="dpt",choices=["dpt","fpn","hwbdpt"],type=str,help="decoder head type")
parser.add_argument("--use-lora", action="store_true", help="enable LoRA (must match training)")

def random_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

# ------------------ Main ------------------
def main():
    args = parser.parse_args()
    warnings.simplefilter('ignore', np.RankWarning)

    logger = init_log('val', logging.INFO)
    logger.propagate = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank, world_size = setup_distributed(port=args.port)
    if rank == 0:
        logger.info(pprint.pformat(vars(args)))
    #random_seeds(314)
    cudnn.enabled = True
    cudnn.benchmark = False

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # ------------------ Dataset ------------------
    size = (args.img_width, args.img_height)

    if args.dataset == 'hypersim':
        valset = Hypersim('dataset/splits/kitti/val.txt', 'val', size=size)
    elif args.dataset == 'vkitti':
        valset = KITTI('dataset/splits/kitti/val.txt', 'val', size=size)
    elif args.dataset == 'scared':
        filelist_path = 'root/scared/scared_test.txt'
        logger.info(f"Test file: {filelist_path}")
        valset = SCARED(filelist_path, 'dpt', size=size)
    elif args.dataset == 'c3vd':
        filelist_path = 'root/c3vd513/test513.txt'
        logger.info(f"Test file: {filelist_path}")
        valset = C3VD(filelist_path,'test', size=size)
    else:
        raise NotImplementedError

    valsampler = torch.utils.data.distributed.DistributedSampler(
        valset, shuffle=False
    )
    valloader = DataLoader(
        valset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        sampler=valsampler
    )

    # ------------------ Model ------------------
    model_configs = {
        'vits':      {'encoder': 'vits',      'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb':      {'encoder': 'vitb',      'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl':      {'encoder': 'vitl',      'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg':      {'encoder': 'vitg',      'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
        'vitso400m': {'encoder': 'vitso400m', 'features': 256, 'out_channels': [256, 512, 1024, 1152]},
        'vithuge':   {'encoder': 'vithuge',   'features': 320, 'out_channels': [320, 640, 1280, 1280]},
        'vit7b':     {'encoder': 'vit7b',     'features': 512, 'out_channels': [1024, 2048, 4096, 4096]},
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth, 'use_lora': args.use_lora})
    #model = SurgicalDINO(backbone_size="no_seed_ori", r=4, lora_layer=None, image_shape=(518,518), decode_type = 'linear4').to(device = device)
    # model = DepthAnythingV2(
    #     **{**model_configs[args.encoder], 'max_depth': args.max_depth},
    # )

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = {}
    for k, v in ckpt.items():
        state_dict[k.replace('module.', '')] = v
    model.load_state_dict(state_dict, strict=False)

    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False
    )

    # ------------------ Validation ------------------
    model.eval()

    results = {
        'd1': torch.tensor([0.0]).cuda(),'d2': torch.tensor([0.0]).cuda(),'d3': torch.tensor([0.0]).cuda(),
        'abs_rel': torch.tensor([0.0]).cuda(),'sq_rel': torch.tensor([0.0]).cuda(),'rmse': torch.tensor([0.0]).cuda(),
        'rmse_log': torch.tensor([0.0]).cuda(),'log10': torch.tensor([0.0]).cuda(),'silog': torch.tensor([0.0]).cuda()
    }
    nsamples = torch.tensor([0.0]).cuda()

    with (torch.no_grad()):
        for sample in valloader:
            img = sample['image'].cuda().float()
            depth = sample['depth'].cuda()[0]
            valid_mask = sample['valid_mask'].cuda()[0]

            # GT depth normalization (和原代码一致)
            # depth /= 256
            # depth *= 100
            # depth *= 100
            # pred = model(img).predicted_depth
            # pred = pred.squeeze(1)
            pred = model(img)

            pred = F.interpolate(
                pred[:, None],
                depth.shape[-2:],
                mode='bilinear',
                align_corners=True
            )[0, 0]

            valid = (valid_mask == 1) & \
                    (depth >= args.min_depth) & \
                    (depth <= args.max_depth)


            if valid.sum() < 10:
                continue
            # scale = depth[valid].median() / pred[valid].median() #计算dev-fine-truned
            # pred[valid] *= scale

            # print(pred[valid].shape)
            # print(valid.sum())

            cur = eval_depth(pred[valid], depth[valid])
            for k in results:
                results[k] += cur[k]
            nsamples += 1

    # ------------------ Reduce ------------------
    torch.distributed.barrier()
    for k in results:
        dist.reduce(results[k], dst=0)
    dist.reduce(nsamples, dst=0)

    # ------------------ Print ------------------
    if rank == 0:
        logger.info("======== Validation Results ========")
        for k in results:
            logger.info(f"{k:>8}: {(results[k] / nsamples).item():.4f}")
        logger.info("===================================")

if __name__ == '__main__':
    main()
