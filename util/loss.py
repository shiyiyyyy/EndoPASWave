import torch
from torch import nn
import torch.nn.functional as F


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss


class GradientLoss(nn.Module):
    def forward(self, pred, target, valid_mask):
        log_pred = torch.log(pred.clamp(min=1e-4))
        log_target = torch.log(target.clamp(min=1e-4))

        pred_dx = log_pred[:, :, 1:] - log_pred[:, :, :-1]
        pred_dy = log_pred[:, 1:, :] - log_pred[:, :-1, :]
        gt_dx = log_target[:, :, 1:] - log_target[:, :, :-1]
        gt_dy = log_target[:, 1:, :] - log_target[:, :-1, :]

        mask_x = valid_mask[:, :, 1:] & valid_mask[:, :, :-1]
        mask_y = valid_mask[:, 1:, :] & valid_mask[:, :-1, :]

        loss_x = F.l1_loss(pred_dx[mask_x], gt_dx[mask_x])
        loss_y = F.l1_loss(pred_dy[mask_y], gt_dy[mask_y])

        return (loss_x + loss_y) * 0.5


class EdgeAwareSmoothnessLoss(nn.Module):
    def forward(self, pred, image):
        # pred: (B, H, W), image: (B, 3, H, W)
        mean_d = pred.mean(dim=[1, 2], keepdim=True).clamp(min=1e-4)
        d = pred / mean_d

        img_gray = image.mean(dim=1)  # (B, H, W)

        d_dx = torch.abs(d[:, :, 1:] - d[:, :, :-1])
        d_dy = torch.abs(d[:, 1:, :] - d[:, :-1, :])

        i_dx = torch.abs(img_gray[:, :, 1:] - img_gray[:, :, :-1])
        i_dy = torch.abs(img_gray[:, 1:, :] - img_gray[:, :-1, :])

        loss = (torch.exp(-i_dx) * d_dx).mean() + (torch.exp(-i_dy) * d_dy).mean()
        return loss




class SurfaceNormalLoss(nn.Module):
    """Cosine distance between surface normals derived from depth gradients."""
    def forward(self, pred, target, valid_mask):
        # Central differences: (B, H-2, W-2)
        dx_pred = pred[:, 1:-1, 2:] - pred[:, 1:-1, :-2]
        dy_pred = pred[:, 2:, 1:-1] - pred[:, :-2, 1:-1]
        dx_gt   = target[:, 1:-1, 2:] - target[:, 1:-1, :-2]
        dy_gt   = target[:, 2:, 1:-1] - target[:, :-2, 1:-1]

        # Valid only when center + 4 neighbours are valid
        vm = (valid_mask[:, 1:-1, 1:-1] &
              valid_mask[:, 1:-1, :-2]  & valid_mask[:, 1:-1, 2:] &
              valid_mask[:, :-2, 1:-1]  & valid_mask[:, 2:, 1:-1])

        if vm.sum() < 10:
            return torch.tensor(0.0, device=pred.device)

        ones = torch.ones_like(dx_pred) * 2.0
        n_pred = F.normalize(torch.stack([-dx_pred, -dy_pred, ones], dim=-1), dim=-1)
        n_gt   = F.normalize(torch.stack([-dx_gt,   -dy_gt,   ones], dim=-1), dim=-1)

        cos_sim = (n_pred * n_gt).sum(dim=-1)  # (B, H-2, W-2)
        return 1.0 - cos_sim[vm].mean()


class SSIMLoss(nn.Module):
    """1 - SSIM in log-depth space, applied only to valid pixels."""
    def __init__(self, window_size=7, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

        # Fixed Gaussian kernel
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.outer(g).unsqueeze(0).unsqueeze(0)  # (1,1,w,w)
        self.register_buffer('kernel', kernel)

    def _conv(self, x):
        pad = self.window_size // 2
        return F.conv2d(x, self.kernel, padding=pad)

    def forward(self, pred, target, valid_mask):
        log_p = torch.log(pred.clamp(min=1e-4)).unsqueeze(1)    # (B,1,H,W)
        log_t = torch.log(target.clamp(min=1e-4)).unsqueeze(1)
        vm    = valid_mask.unsqueeze(1).float()

        mu_p  = self._conv(log_p)
        mu_t  = self._conv(log_t)

        sig_p  = self._conv(log_p * log_p) - mu_p * mu_p
        sig_t  = self._conv(log_t * log_t) - mu_t * mu_t
        sig_pt = self._conv(log_p * log_t) - mu_p * mu_t

        ssim_map = ((2 * mu_p * mu_t + self.C1) * (2 * sig_pt + self.C2)) / \
                   ((mu_p**2 + mu_t**2 + self.C1) * (sig_p + sig_t + self.C2))

        ssim_map = ssim_map * vm
        n = vm.sum().clamp(min=1)
        return 1.0 - ssim_map.sum() / n


class tr(nn.Module):
    def __init__(self, weights=(1.0, 0.5, 0.25)):
        super().__init__()
        self.weights = weights

    def _grad_loss(self, pred, target, valid_mask):
        log_pred   = torch.log(pred.clamp(min=1e-4))
        log_target = torch.log(target.clamp(min=1e-4))

        p_dx = log_pred[:, :, 1:]   - log_pred[:, :, :-1]
        p_dy = log_pred[:, 1:, :]   - log_pred[:, :-1, :]
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
            p  = F.interpolate(p.unsqueeze(1),  scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(1)
            t  = F.interpolate(t.unsqueeze(1),  scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(1)
            vm = F.interpolate(vm.unsqueeze(1).float(), scale_factor=0.5, mode='nearest').squeeze(1).bool()
        return loss

class MultiScaleGradientLoss(nn.Module):
    def __init__(self, weights=(1.0, 0.5, 0.25)):
        super().__init__()
        self.weights = weights

    def _grad_loss(self, pred, target, valid_mask):
        log_pred   = torch.log(pred.clamp(min=1e-4))
        log_target = torch.log(target.clamp(min=1e-4))

        p_dx = log_pred[:, :, 1:]   - log_pred[:, :, :-1]
        p_dy = log_pred[:, 1:, :]   - log_pred[:, :-1, :]
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
            p  = F.interpolate(p.unsqueeze(1),  scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(1)
            t  = F.interpolate(t.unsqueeze(1),  scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(1)
            vm = F.interpolate(vm.unsqueeze(1).float(), scale_factor=0.5, mode='nearest').squeeze(1).bool()
        return loss

class EdgeAwareGradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target, valid_mask, image):
        log_pred = torch.log(pred.clamp(min=1e-4))
        log_tgt  = torch.log(target.clamp(min=1e-4))

        p_dx = log_pred[:, :, 1:] - log_pred[:, :, :-1]
        p_dy = log_pred[:, 1:, :] - log_pred[:, :-1, :]
        t_dx = log_tgt[:, :, 1:]  - log_tgt[:, :, :-1]
        t_dy = log_tgt[:, 1:, :]  - log_tgt[:, :-1, :]

        mask_x = valid_mask[:, :, 1:] & valid_mask[:, :, :-1]
        mask_y = valid_mask[:, 1:, :] & valid_mask[:, :-1, :]

        gray     = image.mean(dim=1, keepdim=True)
        ex       = F.conv2d(gray, self.sobel_x, padding=1)
        ey       = F.conv2d(gray, self.sobel_y, padding=1)
        edge_mag = (ex.pow(2) + ey.pow(2)).sqrt().squeeze(1)
        edge_mag = edge_mag / (edge_mag.amax(dim=[1, 2], keepdim=True) + 1e-6)

        w_x = 1.0 + edge_mag[:, :, 1:]
        w_y = 1.0 + edge_mag[:, 1:, :]

        loss_x = (w_x * torch.abs(p_dx - t_dx))[mask_x].mean()
        loss_y = (w_y * torch.abs(p_dy - t_dy))[mask_y].mean()
        return (loss_x + loss_y) * 0.5