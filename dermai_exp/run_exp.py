"""Single-experiment runner: train on TRAIN, select on VALIDATION, never touch TEST.

    python -m dermai_exp.run_exp --name e01_baseline --set arch=convnext_tiny input_size=224
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .common import (EXP_ROOT, MASTER_SIZE, PRODUCTION_MODEL, SEED, THRESHOLD_GRID,
                     ensure_dirs, get_logger, require_cuda, set_seed)
from .data import (CachedHAM, attach_rows, build_master_cache, find_dataset_root,
                   load_metadata, make_split)
from .engine import Trainer, best_threshold, metrics_at, predict

DEFAULTS = {
    "arch": "convnext_tiny",
    "pretrained": True,
    "dropout": 0.30,
    "bn_norm": True,

    "input_size": 224,
    "multiscale": "",               # e.g. "224,256,288" -> random size per batch
    "res_schedule": False,          # fine at input_size then switch to final_res
    "final_res": 224,
    "res_switch_epoch": 3,

    "unfreeze": -1,                 # -1 = all backbone stage groups
    "lr_head": 3e-4,
    "lr_backbone": 3e-5,
    "stage_decay": 0.7,
    "weight_decay": 0.05,
    "label_smoothing": 0.05,
    "pos_weight": 1.0,
    "clip_grad": 1.0,

    "mixup": 0.0,
    "cutmix": 0.0,
    "mix_p": 0.5,

    "ema_decay": 0.999,
    "aug_strength": "medium",

    "aux_classes": 0,              # 7 = auxiliary fine-grained dx head
    "aux_weight": 0.3,
    "aux_label_smoothing": 0.1,
    "aux_mix": 0.0,                # blend aux cancer mass into the binary prob at inference

    "effective_batch": 32,
    "micro_batch": 0,               # 0 = auto-probe
    "max_micro_batch": 32,
    "min_micro_batch": 2,
    "channels_last": True,

    "epochs": 18,
    "warmup": 2,
    "final_frac": 0.02,
    "patience": 6,
    "seed": SEED,
    "tta": 8,
    "tta_sizes": "",                # e.g. "224,256" -> average logits over scales
}

_STATE = {}


# --------------------------------------------------------------------------- probe
def probe_micro_batch(arch, size, log, ceiling, floor, effective, train_state,
                      channels_last=True):
    """Largest micro batch that survives a real fwd+bwd+step at this arch/resolution."""
    from .aug import TrainAug
    from .engine import Resizer
    from .models import build_model, freeze_policy

    dev = torch.device("cuda")
    chosen, accum = floor, effective // floor
    for bs in range(ceiling, floor - 1, -2):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = None
        opt = None
        try:
            model = build_model(arch, pretrained=train_state, dropout=0.0)
            if channels_last:
                model = model.to(memory_format=torch.channels_last)
            model = model.to(dev)
            groups = freeze_policy(model, -1, 1e-4, 1e-5, 0.05)
            opt = torch.optim.AdamW(groups, lr=1e-4)
            scaler = torch.amp.GradScaler("cuda")
            aug = TrainAug(size, "medium")
            res = Resizer(size, dev, channels_last=channels_last)
            x = aug(res(torch.randint(0, 256, (bs, 3, MASTER_SIZE, MASTER_SIZE),
                                      dtype=torch.uint8, device="cpu")))
            y = torch.randint(0, 2, (bs,), device=dev).float()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            log("  micro batch %2d : OK  peak %.2f / %.2f GB" % (bs, peak, total))
            chosen = bs
            break
        except torch.cuda.OutOfMemoryError:
            log("  micro batch %2d : OOM" % bs)
            opt = None
        finally:
            del model, opt
            torch.cuda.empty_cache()
    accum = max(1, effective // chosen)
    log("  -> micro %d x accum %d = effective %d" % (chosen, accum, chosen * accum))
    return chosen, accum


# --------------------------------------------------------------------------- context
def context(log):
    if "ctx" in _STATE:
        return _STATE["ctx"]
    set_seed(SEED)
    dev = require_cuda(log)
    root, csv = find_dataset_root(log)
    meta = load_metadata(root, csv, log)
    splits, exact = make_split(meta, log)
    attach_rows(meta, splits)
    cache = build_master_cache(meta, log)
    ctx = {"device": dev, "meta": meta, "splits": splits, "cache": cache,
           "exact": exact, "root": root}
    _STATE["ctx"] = ctx
    return ctx


def coerce(v):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def parse_cfg(args):
    cfg = dict(DEFAULTS)
    for kv in args.set or []:
        k, _, v = kv.partition("=")
        if k not in cfg:
            raise SystemExit("FATAL: unknown config key %r" % k)
        cfg[k] = coerce(v)
    for k, v in (("arch", args.arch), ("input_size", args.input_size),
                 ("epochs", args.epochs), ("seed", args.seed),
                 ("aug_strength", args.aug_strength), ("dropout", args.dropout)):
        if v is not None:
            cfg[k] = v
    if not cfg["res_schedule"]:
        cfg["final_res"] = cfg["input_size"]
    else:
        cfg["input_size"] = min(cfg["input_size"], cfg["final_res"])
    return cfg


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--arch", default=None)
    ap.add_argument("--input-size", dest="input_size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--aug", dest="aug_strength", default=None)
    ap.add_argument("--dropout", type=float, default=None)
    args = ap.parse_args()

    ensure_dirs()
    cfg = parse_cfg(args)
    run_dir = os.path.join(EXP_ROOT, args.name)
    os.makedirs(run_dir, exist_ok=True)
    log = get_logger("dermai.%s" % args.name, logfile=os.path.join(run_dir, "log.txt"))

    log("=" * 78)
    log("EXPERIMENT %s" % args.name)
    log("=" * 78)
    log("  production model (never opened): %s" % PRODUCTION_MODEL)
    log("  config: %s" % json.dumps({k: v for k, v in cfg.items() if k != "cache"},
                                    sort_keys=True))
    ctx = context(log)

    if not cfg["micro_batch"]:
        cfg["micro_batch"], _ = probe_micro_batch(
            cfg["arch"], cfg["final_res"], log, cfg["max_micro_batch"],
            cfg["min_micro_batch"], cfg["effective_batch"], cfg["pretrained"],
            cfg["channels_last"])
    cfg["cache"] = ctx["cache"]

    tr = Trainer(cfg, log, train_df=ctx["splits"]["train"], val_df=ctx["splits"]["val"])
    best_path = os.path.join(run_dir, "best.pt")
    t0 = time.time()
    best = tr.train(cfg["epochs"], warmup=cfg["warmup"], final_frac=cfg["final_frac"],
                    save_best_path=best_path)
    train_min = (time.time() - t0) / 60

    # -------------------------------------------------- validation-only selection
    log("")
    log("VALIDATION SELECTION (threshold never uses the test split)")
    vy = ctx["splits"]["val"]["label"].to_numpy().astype(int)
    res = {"best_epoch": best}
    for kind in ("raw", "ema"):
        m = tr.eval_model(kind)
        if m is None:
            continue
        for tta in (1, cfg["tta"]):
            p, y = predict(m, tr.val_loader, tr.resizer, tta=tta,
                                 aux_mix=cfg.get("aux_mix", 0.0))
            assert np.array_equal(y, vy)
            thr_acc, m_acc = best_threshold(y, p, THRESHOLD_GRID, "accuracy")
            thr_f1, m_f1 = best_threshold(y, p, THRESHOLD_GRID, "f1")
            res["%s_tta%d" % (kind, tta)] = {
                "val_auc": m_acc["roc_auc"], "thr_acc": m_acc, "thr_f1": m_f1,
                "val_acc_at_050": float(((p >= 0.5).astype(int) == y).mean())}
            log("  %-4s tta=%d  AUC %.4f | acc@0.5 %.4f | best-acc thr %.3f -> "
                "acc %.4f sens %.4f spec %.4f F1 %.4f"
                % (kind, tta, m_acc["roc_auc"], res["%s_tta%d" % (kind, tta)]["val_acc_at_050"],
                   thr_acc, m_acc["accuracy"], m_acc["sensitivity"], m_acc["specificity"],
                   m_acc["f1"]))
            np.save(os.path.join(run_dir, "val_proba_%s_tta%d.npy" % (kind, tta)), p)

    cand = [k for k in res if isinstance(res[k], dict) and "thr_acc" in res[k]]
    best_key = max(cand, key=lambda k: res[k]["thr_acc"]["accuracy"])
    res["selection"] = best_key
    summary = {"name": args.name, "cfg": {k: v for k, v in cfg.items() if k != "cache"},
               "result": res,
               "split_matches_production": ctx["exact"],
               "train_minutes": round(train_min, 2),
               "history": tr.history}
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    log("")
    log("  selection  : %s" % best_key)
    log("  summary    : %s" % os.path.join(run_dir, "summary.json"))
    log("  wall time  : %.1f min" % train_min)


if __name__ == "__main__":
    main()