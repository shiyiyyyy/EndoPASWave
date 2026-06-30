import argparse
import logging
import os
import pprint
import random
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from dataset.c3vd import C3VD
from dataset.scared import SCARED
from depth_anything_v2.dpt import DepthAnythingV2
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss, EdgeAwareSmoothnessLoss, EdgeAwareGradientLoss, SurfaceNormalLoss, MultiScaleGradientLoss
from util.metric import eval_depth
from util.utils import init_log
from collections import OrderedDict



parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')
parser.add_argument('--encoder', default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg', 'vitso400m', 'vithuge', 'vit7b'])
parser.add_argument('--dataset', default='c3vd', choices=['scared','c3vd'])
parser.add_argument('--img-height', default=512, type=int)
parser.add_argument('--img-width', default=640, type=int)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--max-depth', default=100, type=float)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--bs', default=16, type=int)
parser.add_argument('--lr', default=0.000005, type=float)
parser.add_argument('--pretrained-from',default="./checkpoints/depth_anything_v2_metric_hypersim_vitb.pth",type=str)
#parser.add_argument('--pretrained-from',type=str) #default="checkpoints/depth_anything_v2_metric_hypersim_vitl.pth",.././checkpoints/depth_anything_v2_metric_hypersim_vitb.pth
parser.add_argument('--save-path', default="./test4/yzp/dwt_base_test/dpt",type=str)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument("--lora_type",default="None",choices=["lora","galora","None"],type=str,help="whether lora use for the model")
parser.add_argument("--use-lora", action="store_true", help="enable LoRA + storage adapter")


def random_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


def main():
    args = parser.parse_args()

    warnings.simplefilter('ignore', np.RankWarning)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    rank, world_size = setup_distributed(port=args.port)
    if rank == 0:
        all_args = {**vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
    random_seeds(314)
    cudnn.enabled = True
    cudnn.benchmark = False

    print("Torch CPU random int:", torch.randint(0, 100, (1,)).item())
    if torch.cuda.is_available():
        print("Torch CUDA random int:", torch.randint(0, 100, (1,), device='cuda').item())
    print("NumPy random int:", np.random.randint(0, 100))
    print("Python random int:", random.randint(0, 100))
    print("cudnn.benchmark =", torch.backends.cudnn.benchmark)
    print("cudnn.deterministic =", torch.backends.cudnn.deterministic)
 #   cudnn.benchmark = True #set false

    size = (args.img_width, args.img_height)
    if args.dataset == 'c3vd':
        trainset = C3VD('root/c3vd513/tra513.txt', 'train', size=size)
    elif args.dataset == 'scared':
        trainset = SCARED('root/scared/scared_tra.txt', 'train', size=size)
    else:
        raise NotImplementedError
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler)

    if args.dataset == 'c3vd':
        filelist_path = 'root/c3vd513/val513.txt'
        logger.info(f"Test file: {filelist_path}")
        valset = C3VD(filelist_path, 'val', size=size)
    elif args.dataset == 'scared':
        valset = SCARED('root/scared/scared_val.txt', 'val', size=size)
    else:
        raise NotImplementedError
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, sampler=valsampler)

    local_rank = int(os.environ["LOCAL_RANK"])

    model_configs = {
        'vits':      {'encoder': 'vits',      'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb':      {'encoder': 'vitb',      'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl':      {'encoder': 'vitl',      'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg':      {'encoder': 'vitg',      'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
        'vitso400m': {'encoder': 'vitso400m', 'features': 256, 'out_channels': [256, 512, 1024, 1152]},
        'vithuge':   {'encoder': 'vithuge',   'features': 320, 'out_channels': [320, 640, 1280, 1280]},
        'vit7b':     {'encoder': 'vit7b',     'features': 512, 'out_channels': [1024, 2048, 4096, 4096]},
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth,
                               'use_lora': args.use_lora})



    if args.pretrained_from:
        print("load_pretrain_model")
        ckpt = torch.load(args.pretrained_from, map_location='cpu')
        ret = model.load_state_dict({'pretrained.' + k: v for k, v in ckpt.items()}, strict=False)
        missing_backbone = [k for k in ret.missing_keys if 'pretrained' in k]
        print(f"Backbone missing keys: {len(missing_backbone)}, Unexpected: {len(ret.unexpected_keys)}")
    # if args.pretrained_from:
    #     print("load_pretrain_model")
    #     model.load_state_dict({k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k}, strict=False)
    # model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    #
    # if args.pretrained_from:
    #     ret = model.load_state_dict(
    #         {k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k},
    #         strict=False)
    #     missing_backbone = [k for k in ret.missing_keys if 'pretrained' in k]
    #     print(f"Backbone missing keys: {len(missing_backbone)}, Unexpected: {len(ret.unexpected_keys)}")

    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model.cuda(local_rank)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
    #                                                   output_device=local_rank, find_unused_parameters=True)

    if args.use_lora:
        # freeze backbone, unfreeze only LoRA + scale generator
        for p in model.pretrained.parameters():
            p.requires_grad = False
        lora_modules = (
            [blk.attn.qkv_lora_As for blk in model.pretrained.blocks] +
            [blk.attn.qkv_lora_Bs for blk in model.pretrained.blocks] +
            [blk.attn.qkv_lora_gate for blk in model.pretrained.blocks] +
            [blk.mlp.fc1_lora_As for blk in model.pretrained.blocks] +
            [blk.mlp.fc1_lora_Bs for blk in model.pretrained.blocks] +
            [blk.mlp.fc1_lora_gate for blk in model.pretrained.blocks] +
            [blk.mlp.fc2_lora_As for blk in model.pretrained.blocks] +
            [blk.mlp.fc2_lora_Bs for blk in model.pretrained.blocks] +
            [blk.mlp.fc2_lora_gate for blk in model.pretrained.blocks] +
            [model.pretrained.lora_scale_gen]
        )
        for m in lora_modules:
            for p in m.parameters():
                p.requires_grad = True

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)

    criterion = SiLogLoss(lambd=0).cuda(local_rank)
    criterion_msgrad   = MultiScaleGradientLoss().cuda(local_rank)
    criterion_edge_grad = EdgeAwareGradientLoss().cuda(local_rank)
    criterion_smooth   = EdgeAwareSmoothnessLoss().cuda(local_rank)



    optimizer = AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr  },
                       {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 20 }],
                      lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01) #
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total / 1e6:.2f} M")
    print(f"Trainable params: {trainable / 1e6:.2f} M")

    total_iters = args.epochs * len(trainloader)

    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}
    for epoch in range(args.epochs):
        if rank == 0:
            logger.info('===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(epoch, args.epochs, previous_best['d1'], previous_best['d2'], previous_best['d3']))
            logger.info('===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, '
                        'log10: {:.3f}, silog: {:.3f}'.format(
                            epoch, args.epochs, previous_best['abs_rel'], previous_best['sq_rel'], previous_best['rmse'],
                            previous_best['rmse_log'], previous_best['log10'], previous_best['silog']))

        trainloader.sampler.set_epoch(epoch + 1)

        model.train()
        total_loss = 0

        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()

            img, depth, valid_mask = sample['image'].cuda(), sample['depth'].cuda(), sample['valid_mask'].cuda()
            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            pred = model(img)
            valid = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
            #loss = criterion(pred, depth, valid)
            loss = criterion(pred, depth, valid)  \
                + 0.2 * criterion_edge_grad(pred, depth, valid, img) \
                + 0.2 * criterion_smooth(pred, img) \
                + 0.2 * criterion_msgrad(pred, depth, valid)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters = epoch * len(trainloader) + i

            lr = args.lr * (1 - iters / total_iters) ** 0.5

            optimizer.param_groups[0]["lr"] = lr      # encoder
            optimizer.param_groups[1]["lr"] = lr  * 20  # decoder

            if rank == 0:
                writer.add_scalar('train/loss', loss.item(), iters)

            if rank == 0 and i % 100 == 0:
                logger.info('Iter: {}/{}, Encoder LR: {:.7f}, Decoder LR: {:.7f}, Loss: {:.3f}'.format(
                    i, len(trainloader),
                    optimizer.param_groups[0]['lr'],
                    optimizer.param_groups[1]['lr'],
                    loss.item()))

        model.eval()

        results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(),
                   'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(),
                   'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}
        nsamples = torch.tensor([0.0]).cuda()

        for i, sample in enumerate(valloader):

            img, depth, valid_mask = sample['image'].cuda().float(), sample['depth'].cuda()[0], sample['valid_mask'].cuda()[0]
            with torch.no_grad():
                pred = model(img)

                pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]

            valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)

            if valid_mask.sum() < 10:
                continue

            cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1

        torch.distributed.barrier()

        for k in results.keys():
            dist.reduce(results[k], dst=0)
        dist.reduce(nsamples, dst=0)

        if rank == 0:
            logger.info('==========================================================================================')
            logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
            logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
            logger.info('==========================================================================================')
            print()

            for name, metric in results.items():
                writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)

        for k in results.keys():
            if k in ['d1', 'd2', 'd3']:
                previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
            else:
                previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())

        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            #torch.save(checkpoint, os.path.join(args.save_path, 'best_res.pth')) do not need
            state_dict = model.state_dict()
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '')  # 去掉 module.
                new_state_dict[name] = v

            torch.save(new_state_dict, os.path.join(args.save_path, 'latest_model.pth'))

            current_rmse = (results['rmse'] / nsamples).item()
            if current_rmse <= previous_best['rmse']:
                torch.save(new_state_dict, os.path.join(args.save_path, 'best_model.pth'))

if __name__ == '__main__':
    time.sleep(5)
    main()