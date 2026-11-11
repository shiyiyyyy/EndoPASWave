"""
Training losses (from util/loss.py), limited to the ones used in train.py:

    loss = SiLog(pred, depth, valid)
         + 0.2 * EdgeAwareGradient(pred, depth, valid, img)
         + 0.2 * EdgeAwareSmoothness(pred, img)
         + 0.2 * MultiScaleGradient(pred, depth, valid)

Shapes: pred / target / valid_mask are (B, H, W); image is (B, 3, H, W).
"""
import torch
from torch import nn
import torch.nn.functional as F


class SiLogLoss(nn.Module):
    """Scale-invariant log loss. train.py uses lambd=0."""
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))
        return loss


class EdgeAwareGradientLoss(nn.Module):
    """L1 on log-depth gradients, weighted by (1 + normalized Sobel edge magnitude)."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target, valid_mask, image):
        log_pred = torch.log(pred.clamp(min=1e-4))
        log_tgt = torch.log(target.clamp(min=1e-4))

        p_dx = log_pred[:, :, 1:] - log_pred[:, :, :-1]
        p_dy = log_pred[:, 1:, :] - log_pred[:, :-1, :]
        t_dx = log_tgt[:, :, 1:] - log_tgt[:, :, :-1]
        t_dy = log_tgt[:, 1:, :] - log_tgt[:, :-1, :]

        mask_x = valid_mask[:, :, 1:] & valid_mask[:, :, :-1]
        mask_y = valid_mask[:, 1:, :] & valid_mask[:, :-1, :]

        gray = image.mean(dim=1, keepdim=True)
        ex = F.conv2d(gray, self.sobel_x, padding=1)
        ey = F.conv2d(gray, self.sobel_y, padding=1)
        edge_mag = (ex.pow(2) + ey.pow(2)).sqrt().squeeze(1)
        edge_mag = edge_mag / (edge_mag.amax(dim=[1, 2], keepdim=True) + 1e-6)

        w_x = 1.0 + edge_mag[:, :, 1:]
        w_y = 1.0 + edge_mag[:, 1:, :]

        loss_x = (w_x * torch.abs(p_dx - t_dx))[mask_x].mean()
        loss_y = (w_y * torch.abs(p_dy - t_dy))[mask_y].mean()
        return (loss_x + loss_y) * 0.5


class EdgeAwareSmoothnessLoss(nn.Module):
    """Mean-normalized depth smoothness, down-weighted at image edges (no GT needed)."""
    def forward(self, pred, image):
        mean_d = pred.mean(dim=[1, 2], keepdim=True).clamp(min=1e-4)
        d = pred / mean_d

        img_gray = image.mean(dim=1)  # (B, H, W)

        d_dx = torch.abs(d[:, :, 1:] - d[:, :, :-1])
        d_dy = torch.abs(d[:, 1:, :] - d[:, :-1, :])

        i_dx = torch.abs(img_gray[:, :, 1:] - img_gray[:, :, :-1])
        i_dy = torch.abs(img_gray[:, 1:, :] - img_gray[:, :-1, :])

        return (torch.exp(-i_dx) * d_dx).mean() + (torch.exp(-i_dy) * d_dy).mean()


class MultiScaleGradientLoss(nn.Module):
    """L1 on log-depth gradients at 3 scales (1, 1/2, 1/4)."""
    def __init__(self, weights=(1.0, 0.5, 0.25)):
        super().__init__()
        self.weights = weights

    def _grad_loss(self, pred, target, valid_mask):
        log_pred = torch.log(pred.clamp(min=1e-4))
        log_target = torch.log(target.clamp(min=1e-4))

        p_dx = log_pred[:, :, 1:] - log_pred[:, :, :-1]
        p_dy = log_pred[:, 1:, :] - log_pred[:, :-1, :]
        t_dx = log_target[:, :, 1:] - log_target[:, :, :-1]
        t_dy = log_target[:, 1:, :] - log_target[:, :-1, :]

        mask_x = valid_mask[:, :, 1:] & valid_mask[:, :, :-1]
        mask_y = valid_mask[:, 1:, :] & valid_mask[:, :-1, :]

        lx = torch.abs(p_dx[mask_x] - t_dx[mask_x]).mean() if mask_x.sum() > 0 else pred.new_tensor(0.0)
        ly = torch.abs(p_dy[mask_y] - t_dy[mask_y]).mean() if mask_y.sum() > 0 else pred.new_tensor(0.0)
        return (lx + ly) * 0.5

    def forward(self, pred, target, valid_mask):
        loss = pred.new_tensor(0.0)
        p, t, vm = pred, target, valid_mask
        for w in self.weights:
            loss = loss + w * self._grad_loss(p, t, vm)
            p = F.interpolate(p.unsqueeze(1), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(1)
            t = F.interpolate(t.unsqueeze(1), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(1)
            vm = F.interpolate(vm.unsqueeze(1).float(), scale_factor=0.5, mode='nearest').squeeze(1).bool()
        return loss


class TotalLoss(nn.Module):
    """Combined loss exactly as in train.py."""
    def __init__(self, silog_lambd=0.0, w_edge_grad=0.2, w_smooth=0.2, w_msgrad=0.2):
        super().__init__()
        self.silog = SiLogLoss(lambd=silog_lambd)
        self.edge_grad = EdgeAwareGradientLoss()
        self.smooth = EdgeAwareSmoothnessLoss()
        self.msgrad = MultiScaleGradientLoss()
        self.w_edge_grad = w_edge_grad
        self.w_smooth = w_smooth
        self.w_msgrad = w_msgrad

    def forward(self, pred, target, valid_mask, image):
        return (self.silog(pred, target, valid_mask)
                + self.w_edge_grad * self.edge_grad(pred, target, valid_mask, image)
                + self.w_smooth * self.smooth(pred, image)
                + self.w_msgrad * self.msgrad(pred, target, valid_mask))
