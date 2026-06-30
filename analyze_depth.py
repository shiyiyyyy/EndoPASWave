"""
Quantitative comparison: GT depth (tiff) vs Wave prediction (grayscale PNG).
Usage:
    python analyze_depth.py
"""
import numpy as np
import cv2
import os

GT_DIR   = 'img_test/depth'
PRED_DIR = 'img_test/wave_depth'
FRAMES   = ['0000', '0020', '0040', '0060']
MAX_DEPTH = 100.0   # mm


def load_gt(frame):
    path = os.path.join(GT_DIR, f'{frame}_depth.tiff')
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return d / 255.0 * MAX_DEPTH   # C3VD normalization


def load_pred_gray(frame):
    path = os.path.join(PRED_DIR, f'{frame}_color_gray.png')
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    # grayscale: white=near(small depth), black=far(large depth)
    # invert and normalize to [0, MAX_DEPTH]
    return (255.0 - g) / 255.0 * MAX_DEPTH


def stats(d, valid=None):
    v = d[valid] if valid is not None else d.flatten()
    v = v[v > 0.001]
    return {
        'mean':  v.mean(),
        'std':   v.std(),
        'min':   v.min(),
        'max':   v.max(),
        'range': v.max() - v.min(),
        'cv':    v.std() / (v.mean() + 1e-6),   # coefficient of variation
    }


def gradient_mean(d):
    dx = np.abs(np.diff(d, axis=1))
    dy = np.abs(np.diff(d, axis=0))
    return (dx.mean() + dy.mean()) / 2


def background_std(d, H, W):
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    bg = (xx - cx)**2 + (yy - cy)**2 > (min(H, W) * 0.38)**2
    vals = d[bg]
    vals = vals[vals > 0.001]
    return vals.std() if len(vals) > 100 else 0.0


print(f"\n{'='*90}")
print(f"{'Frame':<8} {'Metric':<20} {'GT':>10} {'Wave':>10} {'Ratio(W/G)':>12} {'Meaning'}")
print(f"{'='*90}")

for fr in FRAMES:
    gt   = load_gt(fr)
    pred = load_pred_gray(fr)

    valid = gt > 0.001
    H, W  = gt.shape

    sg = stats(gt, valid)
    sp = stats(pred)

    gt_grad   = gradient_mean(gt)
    pred_grad = gradient_mean(pred)
    gt_bgstd  = background_std(gt, H, W)
    pred_bgstd= background_std(pred, H, W)

    rows = [
        ('mean (mm)',    sg['mean'],  sp['mean'],  '↑ overestimate / ↓ underestimate'),
        ('std (mm)',     sg['std'],   sp['std'],   '< 1 → depth range compressed'),
        ('range (mm)',   sg['range'], sp['range'], '< 1 → depth range compressed'),
        ('CV',           sg['cv'],    sp['cv'],    '< 1 → relative variation compressed'),
        ('grad mean',    gt_grad,     pred_grad,   '< 1 → prediction too smooth'),
        ('bg std (mm)',  gt_bgstd,    pred_bgstd,  '< 1 → background flattened'),
    ]

    print(f"\n--- Frame {fr} ---")
    for name, gv, pv, meaning in rows:
        ratio = pv / (gv + 1e-6)
        flag  = ' !!!' if ratio < 0.5 else (' !' if ratio < 0.75 else '')
        print(f"{'':8} {name:<20} {gv:>10.3f} {pv:>10.3f} {ratio:>12.3f}  {meaning}{flag}")

print(f"\n{'='*90}")
print("Note: Ratio = Wave/GT. Target ~1.0. '!!!' = severe issue, '!' = notable issue")
