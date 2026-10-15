"""Where does the epoch time actually go? Run: python -m dermai_exp.profile_step"""
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, r"C:\Users\Manjunath\OneDrive\Desktop\MINI3\skin-cancer-app-simple\skin-cancer-app")

from dermai_exp.aug import TrainAug
from dermai_exp.common import get_logger, set_seed
from dermai_exp.data import CachedHAM
from dermai_exp.engine import Resizer
from dermai_exp.models import build_model, freeze_policy

dev = torch.device("cuda")
log = get_logger("prof")
set_seed(42)

cache224 = np.load(r"D:\dermai_cache\ham10000_224x224_u8.npy", mmap_mode="r")
cache384 = np.load(r"D:\dermai_cache\ham10000_master_384x384_u8.npy", mmap_mode="r")
log("cache224 %s  cache384 %s" % (cache224.shape, cache384.shape))

import pandas as pd
rows = np.arange(7010)
labels = np.zeros(7010, dtype=np.float32)


class Frame(torch.utils.data.Dataset):
    def __init__(self, arr, rows):
        self.arr, self.rows = arr, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = np.array(self.arr[self.rows[i]], dtype=np.uint8)
        return torch.from_numpy(img).permute(2, 0, 1), torch.tensor(0.0), i


def bench(tag, ds, bs, nbatch=20, workers=0, pin=True):
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=pin)
    it = iter(dl)
    next(it)
    torch.cuda.synchronize()
    t0 = time.time()
    n = 0
    for _ in range(nbatch):
        xb, yb, ib = next(it)
        n += len(yb)
    torch.cuda.synchronize()
    dt = time.time() - t0
    log("  %-34s bs=%2d w=%d pin=%-5s %6.0f img/s  (%.1f ms/batch)"
        % (tag, bs, workers, pin, n / dt, dt / nbatch * 1000))
    return n / dt


log("DATA LOADER (train rows only)")
bench("mmap384", Frame(cache384, rows), 24)
bench("mmap224", Frame(cache224, rows), 24)
bench("mmap384 nopin", Frame(cache384, rows), 24, pin=False)

ram224 = np.ascontiguousarray(cache224)
log("  loaded 224 cache into RAM: %.2f GB" % (ram224.nbytes / 1024 ** 3))
bench("ram224", Frame(ram224, rows), 24)
bench("ram224 nopin", Frame(ram224, rows), 24, pin=False)

log("")
log("GPU STAGES (bs=24, 224)")
m = build_model("convnext_tiny", True, 0.3, True).to(dev)
opt = torch.optim.AdamW(freeze_policy(m, -1, 3e-4, 3e-5, 0.05, 0.7), lr=3e-4)
scaler = torch.amp.GradScaler("cuda")
res = Resizer(224, dev)
aug = TrainAug(224, "medium")
xb = torch.randint(0, 256, (24, 3, 224, 224), dtype=torch.uint8)
yb = torch.randint(0, 2, (24,)).float().to(dev)


def tm(tag, fn, n=20):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    log("  %-34s %6.1f ms" % (tag, (time.time() - t0) / n * 1000))


tm("resize only", lambda: res(xb))
tm("resize+aug", lambda: aug(res(xb)))
xa = aug(res(xb)).detach()


def fwdbwd():
    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m(xa).float(), yb)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)


tm("fwd+bwd+step (aug tensor)", fwdbwd)


def whole():
    x = aug(res(xb))
    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m(x).float(), yb)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)


tm("whole step resize+aug+fwd+bwd", whole)