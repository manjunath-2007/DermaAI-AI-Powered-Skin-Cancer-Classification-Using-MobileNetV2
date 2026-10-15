"""Batched, per-sample GPU augmentation. Input/output float32 (B,3,H,W) in [0,1].

Every random parameter is drawn per sample, so one call behaves like a per-image pipeline
while costing a single set of CUDA kernels (no CPU/GPU round trips, no PIL).

Convention: `_bern(B, ...)` returns a bool tensor of shape (B,); call sites reshape
explicitly for the layout they broadcast against.
"""

import math

import torch
import torch.nn.functional as F


def _rand(B, device, lo=0.0, hi=1.0):
    return torch.rand(B, device=device) * (hi - lo) + lo


def _bern(B, device, p):
    return torch.rand(B, device=device) < p


# --------------------------------------------------------------------------- geometry
def random_affine(x, rot_deg=180.0, scale=(0.70, 1.0), ratio=(0.80, 1.25),
                  shear_deg=8.0, translate=0.06, flip_h=True, flip_v=False):
    """Crop + rotate + scale + shear + translate in a single affine_grid/grid_sample."""
    B, _, H, W = x.shape
    dev = x.device

    area = _rand(B, dev, scale[0], scale[1]).sqrt()          # crop side fraction
    logr = _rand(B, dev, math.log(ratio[0]), math.log(ratio[1]))
    ar = torch.exp(logr)
    sx = (area * ar).clamp(max=1.0)                          # crop width  fraction
    sy = (area / ar).clamp(max=1.0)                          # crop height fraction
    sx = torch.maximum(sx, sy)                               # keep aspect within crop box
    sy = torch.maximum(sx, sy)

    ang = _rand(B, dev, -rot_deg, rot_deg) * math.pi / 180.0
    shear = _rand(B, dev, -shear_deg, shear_deg) * math.pi / 180.0

    cos, sin = torch.cos(ang), torch.sin(ang)
    tan = torch.tan(shear).abs()
    a = cos / sx                                             # inverse affine
    b = (-sin + cos * tan) / sy
    c = sin / sx
    d = (cos + sin * tan) / sy
    tx = _rand(B, dev, -translate, translate) * 2.0
    ty = _rand(B, dev, -translate, translate) * 2.0

    theta = torch.zeros(B, 2, 3, device=dev, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = a, b, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = c, d, ty

    grid = F.affine_grid(theta, (B, 3, H, W), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                      align_corners=False)

    if flip_h:
        m = _bern(B, dev, 0.5).view(B, 1, 1, 1)
        x = torch.where(m, torch.flip(x, dims=[3]), x)
    if flip_v:
        m = _bern(B, dev, 0.5).view(B, 1, 1, 1)
        x = torch.where(m, torch.flip(x, dims=[2]), x)
    return x.clamp_(0.0, 1.0)


# --------------------------------------------------------------------------- colour
def _rgb_to_hsv(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx, mn = rgb.max(1).values, rgb.min(1).values
    d = (mx - mn).clamp_min(1e-8)
    h = torch.where(mx == r, ((g - b) / d) % 6.0,
                    torch.where(mx == g, (b - r) / d + 2.0, (r - g) / d + 4.0))
    h = (h / 6.0) % 1.0
    s = torch.where(mx > 1e-8, d / mx.clamp_min(1e-8), torch.zeros_like(d))
    return h, s, mx


def _hsv_to_rgb(h, s, v):
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    B, H, W = h.shape
    out = torch.stack([
        torch.stack([v, t, p], 1), torch.stack([q, v, p], 1),
        torch.stack([p, v, t], 1), torch.stack([p, q, v], 1),
        torch.stack([t, p, v], 1), torch.stack([v, p, q], 1),
    ], 0)                                                    # (6,B,3,H,W)
    idx = (i % 6).long().reshape(1, B, 1, H, W).expand(1, B, 3, H, W)
    return torch.gather(out, 0, idx).squeeze(0)


def color_jitter(x, brightness=0.25, contrast=0.25, saturation=0.25, hue=0.06,
                 p=0.9, gray_p=0.0):
    B, _, _, _ = x.shape
    dev = x.device
    pm = _bern(B, dev, p)
    on = pm.view(B, 1, 1, 1)
    on_hsv = pm.to(x.dtype).view(B)

    b = _rand(B, dev, 1 - brightness, 1 + brightness).view(B, 1, 1, 1)
    c = _rand(B, dev, 1 - contrast, 1 + contrast).view(B, 1, 1, 1)
    sat = _rand(B, dev, 1 - saturation, 1 + saturation)
    hu = _rand(B, dev, -hue, hue)

    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = torch.where(on, x * b, x)
    x = torch.where(on, (x - mean) * c + mean, x)

    h, sa, v = _rgb_to_hsv(x.clamp(0, 1))
    h = (h + (hu * on_hsv).view(B, 1, 1)) % 1.0
    sa = (sa * (sat * on_hsv + (1 - on_hsv)).view(B, 1, 1)).clamp(0, 1)
    x = _hsv_to_rgb(h, sa, v).clamp(0, 1)

    if gray_p > 0:
        g = (x * torch.tensor([0.299, 0.587, 0.114], device=dev,
                              dtype=x.dtype).view(1, 3, 1, 1)).sum(1, keepdim=True)
        x = torch.where(_bern(B, dev, gray_p).view(B, 1, 1, 1), g.expand_as(x), x)
    return x.clamp_(0.0, 1.0)


def gaussian_blur(x, sigma_range=(0.2, 1.6), p=0.25):
    """Separable 5-tap blur, per-sample random sigma (small sigma ~ identity)."""
    B, C, H, W = x.shape
    dev = x.device
    sig = _rand(B, dev, *sigma_range)
    k = torch.arange(5, device=dev, dtype=x.dtype) - 2.0
    ker = torch.exp(-(k.view(1, -1) ** 2) / (2 * sig.view(B, 1) ** 2))       # (B,5)
    ker = ker / ker.sum(1, keepdim=True)
    out = x.reshape(1, B * C, H, W)
    w = ker.view(B, 1, 1, 5).expand(B, C, 1, 5).reshape(B * C, 1, 1, 5)
    out = F.conv2d(F.pad(out, (2, 2, 0, 0), mode="reflect"), w, groups=B * C)
    w2 = ker.view(B, 1, 5, 1).expand(B, C, 5, 1).reshape(B * C, 1, 5, 1)
    out = F.conv2d(F.pad(out, (0, 0, 2, 2), mode="reflect"), w2, groups=B * C)
    out = out.reshape(B, C, H, W)
    return torch.where(_bern(B, dev, p).view(B, 1, 1, 1), out, x)


def _box3(x):
    k = torch.ones(1, 1, 3, 3, device=x.device, dtype=x.dtype) / 9.0
    C = x.shape[1]
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k.expand(C, 1, 3, 3), groups=C)


def sharpen(x, amount=0.3, p=0.2):
    out = x + amount * (x - _box3(x))
    return torch.where(_bern(x.shape[0], x.device, p).view(-1, 1, 1, 1),
                       out, x).clamp_(0, 1)


def random_erase(x, p=0.35, area=(0.02, 0.25), ratio=(0.3, 3.3)):
    """Cutout: forces the model off single-scale dark blobs."""
    B, _, H, W = x.shape
    dev = x.device
    a = _rand(B, dev, area[0], area[1])
    r = torch.exp(_rand(B, dev, math.log(ratio[0]), math.log(ratio[1])))
    h = (torch.sqrt(a * H * W * r)).clamp(1, H - 1).long()
    w = (torch.sqrt(a * H * W / r)).clamp(1, W - 1).long()
    y0 = (torch.rand(B, device=dev) * (H - h).float()).long()
    x0 = (torch.rand(B, device=dev) * (W - w).float()).long()
    rows = torch.arange(H, device=dev).view(1, H, 1)
    cols = torch.arange(W, device=dev).view(1, 1, W)
    m = ((rows >= y0.view(B, 1, 1)) & (rows < (y0 + h).view(B, 1, 1))
         & (cols >= x0.view(B, 1, 1)) & (cols < (x0 + w).view(B, 1, 1))
         & _bern(B, dev, p).view(B, 1, 1)).unsqueeze(1)
    return torch.where(m, torch.rand_like(x), x)


def dark_blob(x, p=0.15, k=5):
    """Synthetic hairs / ruler marks / bubbles: thin dark strokes, a classic HAM10000 artefact."""
    B, _, H, W = x.shape
    dev = x.device
    yy = torch.arange(H, device=dev, dtype=x.dtype).view(1, H, 1)
    xx = torch.arange(W, device=dev, dtype=x.dtype).view(1, 1, W)
    img = x
    for _ in range(k):
        th = torch.rand(B, device=dev) * math.pi
        cx = _rand(B, dev, 0.1, 0.9) * W
        cy = _rand(B, dev, 0.1, 0.9) * H
        L = _rand(B, dev, 0.15, 0.6) * W
        wid = _rand(B, dev, 0.004, 0.012) * W
        dx = xx - cx.view(B, 1, 1)
        dy = yy - cy.view(B, 1, 1)
        dist = torch.abs(dx * torch.sin(th).view(B, 1, 1)
                         - dy * torch.cos(th).view(B, 1, 1))
        line = ((dist < wid.view(B, 1, 1)) &
                ((dx * torch.cos(th).view(B, 1, 1)
                  + dy * torch.sin(th).view(B, 1, 1)).abs() < L.view(B, 1, 1))
                & _bern(B, dev, p).view(B, 1, 1)).unsqueeze(1).to(x.dtype)
        img = img * (1 - line) + line * 0.08
    return img.clamp_(0, 1)


# --------------------------------------------------------------------------- mixing
def _np():
    import numpy as np
    return np.random


def mixup(x, y, alpha=0.2, prob=0.5):
    B = x.shape[0]
    if _np().random_sample() > prob:
        return x, y
    lam = max(float(_np().beta(alpha, alpha)), 1e-6)
    lam = max(lam, 1 - lam)
    perm = torch.randperm(B, device=x.device)
    return x * lam + x[perm] * (1 - lam), y * lam + y[perm] * (1 - lam)


def cutmix(x, y, alpha=1.0, prob=0.5):
    B, _, H, W = x.shape
    if _np().random_sample() > prob:
        return x, y
    lam = float(_np().beta(alpha, alpha))
    rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
    if rh < 1 or rw < 1:
        return x, y
    perm = torch.randperm(B, device=x.device)
    cy, cx = int(torch.randint(H, (1,)).item()), int(torch.randint(W, (1,)).item())
    y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
    x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
    lam = 1 - ((y2 - y1) * (x2 - x1) / (H * W))
    return x, y * lam + y[perm] * (1 - lam)


# --------------------------------------------------------------------------- pipeline
class TrainAug:
    """Configurable train-time augmentation applied to a float batch in [0,1]."""

    STRENGTH = {"none": 0.0, "light": 0.6, "medium": 1.0, "heavy": 1.35}

    def __init__(self, size=224, strength="medium"):
        self.size = size
        self.s = self.STRENGTH[strength]

    def __call__(self, x):
        s = self.s
        if s == 0:
            return x
        x = random_affine(x,
                          rot_deg=180.0 * s,
                          scale=(1.0 - 0.30 * s, 1.0),
                          ratio=(1.0 - 0.20 * s, 1.0 + 0.20 * s),
                          shear_deg=8.0 * s,
                          translate=0.06 * s,
                          flip_h=True, flip_v=False)
        x = color_jitter(x, brightness=0.25 * s, contrast=0.25 * s,
                         saturation=0.25 * s, hue=0.06 * s, p=0.85,
                         gray_p=0.05 * s)
        x = sharpen(x, amount=0.25 * s, p=0.15)
        x = dark_blob(x, p=0.12 * s)
        x = random_erase(x, p=0.30 * s, area=(0.02, 0.10 + 0.15 * s))
        return x.clamp_(0.0, 1.0).contiguous(memory_format=torch.channels_last)


class EvalTTA:
    """Deterministic test-time augmentation views.

    The 8 dihedral transforms (flips + 90-degree rotation) are label preserving for
    dermoscopy, so they are free extra views.
    """

    NAMES = ("id", "h", "v", "hv", "r90", "r90h", "r90v", "r90hv")

    def __init__(self, n_views=1):
        self.n_views = int(max(1, min(int(n_views), 8)))

    @staticmethod
    def apply(x, k):
        f = EvalTTA.NAMES[k % 8]
        if "h" in f[1:]:
            x = torch.flip(x, dims=[3])
        if "v" in f[1:]:
            x = torch.flip(x, dims=[2])
        if f.startswith("r90"):
            x = torch.rot90(x, 1, dims=[2, 3])
        return x