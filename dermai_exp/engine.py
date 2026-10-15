"""Training / evaluation engine: AMP, EMA, layer-wise LR, resolution schedule, TTA."""

import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .aug import EvalTTA, TrainAug, cutmix, mixup
from .common import CLASS_NAMES, SEED
from .data import CachedHAM, CachedHAMAux
from .models import build_model, freeze_policy


# --------------------------------------------------------------------------- EMA
class ModelEMA:
    """Exponential moving average of weights; a cheap, reliable +0.3-1.5% at eval time."""

    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model, decay=None):
        d = self.decay if decay is None else decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            mv = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(d).add_(mv.detach(), alpha=1 - d)
            else:
                v.copy_(mv)

    def state_dict(self):
        return self.ema.state_dict()


# --------------------------------------------------------------------------- metrics
def metrics_at(y, p, thr):
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr), "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def sweep(y, p, grid):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    rows = []
    for t in grid:
        pred = (p >= t).astype(int)
        rows.append((float(t), accuracy_score(y, pred),
                     precision_score(y, pred, zero_division=0),
                     recall_score(y, pred, zero_division=0),
                     f1_score(y, pred, zero_division=0)))
    return rows


def best_threshold(y, p, grid, key="accuracy"):
    """Grid search on the given objective. Returns (threshold, metrics_dict)."""
    rows = sweep(y, p, grid)
    i = int(np.argmax([r[{"accuracy": 1, "f1": 4, "balanced_accuracy": 0}[key]] for r in rows]))
    return rows[i][0], metrics_at(y, p, rows[i][0])


# --------------------------------------------------------------------------- inference
class Resizer:
    """Downsample the 384 master cache to the model resolution on GPU, channels-last.

    channels_last matters: fp16 convnets on sm_86 reach tensor cores only with the NHWC
    layout, and the input must match the weights' layout or it is silently converted.
    """

    def __init__(self, size, device, channels_last=True, sizes=None):
        self.device = device
        self.channels_last = channels_last
        self.sizes = [int(s) for s in (sizes or [size])]
        self.eval_size = int(size)
        self.random_size = len(self.sizes) > 1
        self.cur = self.eval_size

    def __call__(self, xb_u8):
        self.cur = (int(self.sizes[int(torch.randint(len(self.sizes), (1,)))])
                    if self.random_size else self.eval_size)
        x = xb_u8.to(self.device, non_blocking=True).float().div_(255.0)
        if x.shape[-1] != self.cur:
            x = F.interpolate(x, size=(self.cur, self.cur), mode="bilinear",
                              align_corners=False, antialias=True)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x


@torch.no_grad()
def predict(model, loader, resizer, tta=1, amp_dtype=torch.float16, return_logits=False,
            aux_mix=0.0, sizes=None):
    """Returns (probs, labels) aligned positionally with the loader's dataset frame.

    `tta`  : number of dihedral views (1..8) per evaluation resolution.
    `sizes`: list of evaluation resolutions. Averaging logits across scales and flips is
             free extra diversity; validation showed 256/288 beat the 224 training size.
    With an auxiliary 7-way dx head, the binary probability is blended with the summed
    softmax mass of the three malignant dx classes (`aux_mix`).
    """
    was = model.training
    model.eval()
    acc_p, acc_l, acc_i = [], [], []
    acc_logit = []
    views = list(range(int(max(1, min(int(tta), 8)))))
    ev_sizes = [int(s) for s in (sizes or [resizer.eval_size])]
    was_random = getattr(resizer, "random_size", False)
    resizer.random_size = False                  # evaluation always uses eval_size
    aux_mix = float(aux_mix) if aux_mix > 0 else 0.0
    cancer_idx = (0, 1, 4)                       # akiec, bcc, mel in DX_CLASSES order
    try:
        for batch in loader:
            xb, yb, ib = batch[0], batch[1], batch[2]
            logits, aux_p = None, None
            n = 0
            for s in ev_sizes:
                resizer.eval_size = s
                x = resizer(xb)
                for v in views:
                    xv = EvalTTA.apply(x, v)
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        out = model(xv)
                    lg, aux = out if isinstance(out, tuple) else (out, None)
                    lg = lg.float()
                    logits = lg if logits is None else logits + lg
                    if aux is not None:
                        ap = torch.softmax(aux.float(), 1)[:, list(cancer_idx)].sum(1)
                        aux_p = ap if aux_p is None else aux_p + ap
                    n += 1
            logits = logits / n
            p = torch.sigmoid(logits)
            if aux_p is not None and aux_mix > 0:
                p = (1 - aux_mix) * p + aux_mix * (aux_p / n)
            acc_p.append(p.cpu().numpy())
            acc_logit.append(logits.cpu().numpy())
            acc_l.append(yb.numpy())
            acc_i.append(ib.numpy())
    finally:
        resizer.random_size = was_random
        resizer.eval_size = ev_sizes[0]
    if was:
        model.train()
    p = np.concatenate(acc_p).astype(np.float64)
    lg = np.concatenate(acc_logit).astype(np.float32)
    y = np.concatenate(acc_l).astype(np.int32)
    order = np.concatenate(acc_i)
    out_p = np.empty_like(p)
    out_y = np.empty_like(y)
    out_l = np.empty_like(lg)
    out_p[order], out_y[order], out_l[order] = p, y, lg
    return (out_p, out_y, out_l) if return_logits else (out_p, out_y)


# --------------------------------------------------------------------------- runner
class Trainer:
    def __init__(self, cfg, log, train_df=None, val_df=None, eval_frames=None):
        from .data import attach_rows  # noqa: F401  (kept for symmetry)
        self.cfg = cfg
        self.log = log
        self.device = torch.device("cuda")
        self.train_df = train_df
        self.val_df = val_df
        self.eval_frames = eval_frames or {}

        self.model = build_model(cfg["arch"], pretrained=cfg["pretrained"],
                                 dropout=cfg["dropout"], bn_norm=cfg["bn_norm"],
                                 aux_classes=cfg.get("aux_classes", 0),
                                 log=log).to(self.device)
        if cfg.get("channels_last", True):
            self.model = self.model.to(memory_format=torch.channels_last)
        self.ema = ModelEMA(self.model, cfg["ema_decay"]) if cfg["ema_decay"] > 0 else None
        ms = cfg.get("multiscale")
        self.resizer = Resizer(cfg["input_size"], self.device,
                               channels_last=cfg.get("channels_last", True),
                               sizes=([int(s) for s in ms.split(",")] if ms else None))
        self.eval_size = int(cfg.get("eval_size") or cfg["input_size"])
        self.scaler = torch.amp.GradScaler("cuda")
        self.epoch = 0
        self.best = {"val_auc": -1.0, "val_acc": -1.0, "epoch": -1, "kind": "raw"}
        self.history = []
        self._build_loaders()

    # -------------------------------------------------------------- data
    def _build_loaders(self):
        c = self.cfg
        cache = c["cache"]
        micro = c["micro_batch"]
        ds_cls = CachedHAMAux if c.get("aux_classes", 0) else CachedHAM
        if self.train_df is not None:
            ds = ds_cls(cache, self.train_df, train=True)
            self.train_loader = DataLoader(
                ds, batch_size=micro, shuffle=True, num_workers=0, pin_memory=True,
                drop_last=False, generator=torch.Generator().manual_seed(SEED))
        self.val_loader = None
        if self.val_df is not None:
            self.val_loader = DataLoader(ds_cls(cache, self.val_df),
                                         batch_size=micro * 2, shuffle=False,
                                         num_workers=0, pin_memory=True)
        self.eval_loaders = {k: DataLoader(ds_cls(cache, df), batch_size=micro * 2,
                                           shuffle=False, num_workers=0, pin_memory=True)
                             for k, df in self.eval_frames.items()}

    # -------------------------------------------------------------- schedule
    def _lr_scale(self, epoch, total, warmup, final_frac):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        t = (epoch - warmup) / max(total - warmup, 1)
        return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    def _make_optimizer(self, epoch_total, warmup, final_frac):
        c = self.cfg
        groups = freeze_policy(self.model, c["unfreeze"], c["lr_head"], c["lr_backbone"],
                               c["weight_decay"], c["stage_decay"], log=self.log)
        self._base_lrs = [g["lr"] for g in groups]
        opt = torch.optim.AdamW(groups, lr=c["lr_head"], weight_decay=c["weight_decay"],
                                betas=(0.9, 0.999), eps=1e-8)
        return opt

    def _apply_res(self, res):
        """Input-resolution schedule (fine -> full)."""
        self.resizer.eval_size = int(res)
        self.resizer.sizes[-1] = int(res)
        self.cfg["input_size"] = int(res)

    # -------------------------------------------------------------- loss
    def _loss(self, out, yb, pos_weight, dxb=None):
        logits, aux = out if isinstance(out, tuple) else (out, None)
        if self.cfg["label_smoothing"] > 0:
            yb = yb * (1 - self.cfg["label_smoothing"]) + 0.5 * self.cfg["label_smoothing"]
        loss = F.binary_cross_entropy_with_logits(
            logits, yb, pos_weight=None if pos_weight == 1.0 else
            torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype))
        if aux is not None and dxb is not None:
            ls = self.cfg["aux_label_smoothing"]
            loss = loss + self.cfg["aux_weight"] * F.cross_entropy(
                aux.float(), dxb, label_smoothing=ls)
        return loss

    # -------------------------------------------------------------- train
    def train(self, total_epochs, warmup=2, final_frac=0.02, log_every=1,
              score_frames=None, save_best_path=None, on_epoch=None):
        c = self.cfg
        opt = self._make_optimizer(total_epochs, warmup, final_frac)
        aug = TrainAug(c["input_size"], c["aug_strength"])
        pos_weight = float(c["pos_weight"])
        steps = len(self.train_loader)
        accum = max(1, int(round(c["effective_batch"] / c["micro_batch"])))
        best_auc, bad = -1.0, 0
        t_start = time.time()

        for epoch in range(total_epochs):
            if c["res_schedule"] and epoch == int(c["res_switch_epoch"]):
                new = int(c["input_size"])
                if self.resizer.eval_size != new:
                    self._apply_res(new)
                    self.log("  resolution -> %d at epoch %d" % (new, epoch))
            scale = self._lr_scale(epoch, total_epochs, warmup, final_frac)
            for g, bl in zip(opt.param_groups, self._base_lrs):
                g["lr"] = bl * scale

            self.model.train()
            if self.ema:
                self.ema.decay = min(c["ema_decay"], (1 + epoch) / (10 + epoch))
            t0 = time.time()
            run_loss, seen = 0.0, 0
            oom, skipped = 0, 0
            opt.zero_grad(set_to_none=True)
            for step, batch in enumerate(self.train_loader):
                if len(batch) == 4:
                    xb, yb, _ib, dxb = batch
                    dxb = dxb.to(self.device, non_blocking=True)
                else:
                    xb, yb, _ib = batch
                    dxb = None
                try:
                    x = self.resizer(xb)
                    x = aug(x)
                    yb = yb.to(self.device, non_blocking=True)
                    if c["mixup"] > 0:
                        x, yb = mixup(x, yb, alpha=c["mixup"], prob=c["mix_p"])
                        dxb = None                       # mixing invalidates the dx label
                    elif c["cutmix"] > 0:
                        x, yb = cutmix(x, yb, alpha=c["cutmix"], prob=c["mix_p"])
                        dxb = None
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        loss = self._loss(self.model(x), yb, pos_weight, dxb)
                    self.scaler.scale(loss / accum).backward()
                    if (step + 1) % accum == 0 or (step + 1) == steps:
                        if c["clip_grad"] > 0:
                            self.scaler.unscale_(opt)
                            torch.nn.utils.clip_grad_norm_(
                                [p for g in opt.param_groups for p in g["params"]],
                                c["clip_grad"])
                        self.scaler.step(opt)
                        self.scaler.update()
                        opt.zero_grad(set_to_none=True)
                    run_loss += float(loss.detach()) * len(yb)
                    seen += len(yb)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    # A transient cudnn/driver failure (this 4 GB WDDM card does hit them)
                    # must not lose a whole multi-hour run: drop the batch and carry on.
                    oom += 1
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    if oom in (1, 5, 20):
                        self.log("  ! step %d dropped after %s: %s"
                                 % (step, type(exc).__name__, str(exc)[:90]))
                    if oom >= 200:
                        raise RuntimeError("too many CUDA failures: %s" % exc)
            if skipped:
                self.log("  epoch %d: %d/%d steps skipped after CUDA errors"
                         % (epoch, skipped, steps))
            tr_loss = run_loss / max(seen, 1)
            if self.ema:
                self.ema.update(self.model)

            rec = {"epoch": epoch, "train_loss": tr_loss,
                   "lr": opt.param_groups[-1]["lr"], "res": self.resizer.eval_size,
                   "seconds": time.time() - t0}

            cands = [("raw", self.model)]
            if self.ema:
                cands.append(("ema", self.ema.ema))
            if self.val_loader is not None:
                y = self.val_df["label"].to_numpy().astype(int)
                for kind, m in cands:
                    try:
                        p, _ = predict(m, self.val_loader, self.resizer, tta=1,
                                       aux_mix=c.get("aux_mix", 0.0))
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                        self.log("  ! val scoring failed (%s): %s"
                                 % (type(exc).__name__, str(exc)[:90]))
                        torch.cuda.empty_cache()
                        continue
                    rec["val_auc_%s" % kind] = _auc(y, p)
                    rec["val_acc_%s" % kind] = _acc(y, p)
            self.history.append(rec)
            msg = ("  ep %2d/%d loss %.4f lr %.2e res %d %.0fs"
                   % (epoch + 1, total_epochs, tr_loss, rec["lr"], rec["res"],
                      rec["seconds"]))
            for k in ("val_auc_raw", "val_auc_ema", "val_acc_raw", "val_acc_ema"):
                if k in rec:
                    msg += "  %s %.4f" % (k.replace("val_", ""), rec[k])
            self.log(msg)

            pick = max(cands, key=lambda kv: rec.get("val_auc_%s" % kv[0], -1))
            improved = rec.get("val_auc_%s" % pick[0], -1) > best_auc
            if improved:
                best_auc = rec["val_auc_%s" % pick[0]]
                self.best = {"val_auc": best_auc, "epoch": epoch, "kind": pick[0],
                             "val_acc": rec.get("val_acc_%s" % pick[0])}
                if save_best_path:
                    torch.save({"state_dict": pick[1].state_dict(),
                                "cfg": {k: v for k, v in c.items() if k != "cache"},
                                "best": dict(self.best),
                                "total_epochs": total_epochs}, save_best_path)
            bad = 0 if improved else bad + 1
            self.epoch = epoch + 1
            if on_epoch:
                on_epoch(epoch, rec, dict(self.best))
            if bad >= int(c["patience"]):
                self.log("  early stop: %d epochs without val AUC gain" % bad)
                break

        self.log("  train wall time %.1f min | best val AUC %.4f (%s ep %d)"
                 % ((time.time() - t_start) / 60, self.best["val_auc"],
                    self.best["kind"], self.best["epoch"]))
        return self.best

    def eval_model(self, kind="ema"):
        m = self.ema.ema if (kind == "ema" and self.ema) else self.model
        return m


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def _acc(y, p):
    from sklearn.metrics import accuracy_score
    return float(accuracy_score(y, (p >= 0.5).astype(int)))