"""Micro-benchmark ConvNeXt-Tiny fwd+bwd under different layouts / batch sizes."""
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, r"C:\Users\Manjunath\OneDrive\Desktop\MINI3\skin-cancer-app-simple\skin-cancer-app")
from dermai_exp.models import build_model, freeze_policy

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = torch.device("cuda")


def bench(tag, model, bs, res, cl, dtype=torch.float16, n=12):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randint(0, 256, (bs, 3, res, res), dtype=torch.uint8, device=dev)
    if cl:
        model = model.to(memory_format=torch.channels_last)

    def step():
        xf = x.float().div(255.0)
        if cl:
            xf = xf.contiguous(memory_format=torch.channels_last)
        yb = torch.randint(0, 2, (bs,), device=dev).float()
        with torch.amp.autocast("cuda", dtype=dtype):
            loss = nn.functional.binary_cross_entropy_with_logits(model(xf).float(), yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    try:
        step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / n
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print("  %-38s bs=%2d res=%3d cl=%-5s  %6.0f ms  %6.1f img/s  peak %.2f GB"
              % (tag, bs, res, cl, dt * 1000, bs / dt, peak), flush=True)
        return bs / dt
    except torch.cuda.OutOfMemoryError:
        print("  %-38s bs=%2d res=%3d cl=%-5s  OOM" % (tag, bs, res, cl), flush=True)
        return 0
    finally:
        opt = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


print("ConvNeXt-Tiny fwd+bwd (fp16 AMP)", flush=True)
for res in (224, 288, 320):
    for cl in (False, True):
        m = build_model("convnext_tiny", True, 0.3, True)
        m.to(dev)
        for bs in (24, 48, 64):
            bench("convnext_tiny", m, bs, res, cl)
        del m
        torch.cuda.empty_cache()