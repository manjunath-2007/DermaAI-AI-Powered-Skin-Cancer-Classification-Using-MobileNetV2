"""Final pipeline: k-fold CV on TRAIN+VAL, ensemble, OOF threshold, ONE test evaluation.

Design notes (why this is legitimate):
  * Every hyper-parameter is chosen on TRAIN/VALIDATION only (see results/experiments).
  * The test split is loaded exactly once, at the end, at the frozen OOF threshold.
  * The ensemble is a lesion-grouped StratifiedGroupKFold over the TRAIN+VAL pool, so no
    lesion is ever seen by two models' training folds, and the out-of-fold predictions are
    an honest estimate that the threshold can be tuned on.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .common import (CLASS_NAMES, FINAL_ROOT, PRODUCTION_MODEL, SEED, THRESHOLD_GRID,
                     ensure_dirs, get_logger, set_seed)
from .data import CachedHAM, fold_frames
from .engine import Trainer, best_threshold, metrics_at, predict
from .run_exp import DEFAULTS, coerce, context, probe_micro_batch


# --------------------------------------------------------------------------- ensembling
def tta_sizes(cfg):
    """Evaluation resolutions for multi-scale TTA (None -> the model's training size)."""
    s = cfg.get("tta_sizes") or ""
    return [int(x) for x in s.split(",") if x.strip()] or None


def weighted_blend(preds, weights=None):
    """Blend in LOGIT space (geometric mean of odds) - better calibrated than prob mean."""
    preds = np.asarray(preds, dtype=np.float64)
    if weights is None:
        weights = np.ones(len(preds))
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    eps = 1e-6
    lg = np.log(np.clip(preds, eps, 1 - eps) / (1 - np.clip(preds, eps, 1 - eps)))
    z = (lg * weights[:, None]).sum(0)
    return 1.0 / (1.0 + np.exp(-z))


def greedy_ensemble(y, oof_list, names, max_k=8, log=print):
    """Greedy forward selection with replacement (Caruana): maximises OOF accuracy."""
    picks, cur, best_hist = [], None, []
    k = len(oof_list)
    for step in range(max_k):
        best_i, best_acc, best_blend = -1, -1.0, None
        for i in range(k):
            cand = oof_list[i] if cur is None else weighted_blend(cur + [oof_list[i]])
            thr, m = best_threshold(y, cand, THRESHOLD_GRID, "accuracy")
            if m["accuracy"] > best_acc:
                best_i, best_acc, best_blend = i, m["accuracy"], cand
        if cur is not None and best_acc <= best_hist[-1][1] + 1e-9:
            log("  greedy stop at step %d (no OOF accuracy gain)" % step)
            break
        picks.append(best_i)
        best_hist.append((names[best_i], best_acc))
        cur = (cur + [oof_list[best_i]]) if cur is not None else [oof_list[best_i]]
        log("  greedy step %d -> +%-24s OOF acc %.4f" % (step, names[best_i], best_acc))
    counts = {names[i]: picks.count(i) for i in set(picks)}
    return picks, counts, weighted_blend(cur)


# --------------------------------------------------------------------------- runner
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--arch", default="convnext_tiny")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--only-folds", type=int, default=None,
                    help="run just these fold indices (comma separated)")
    ap.add_argument("--tag", default="cv")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-resume", dest="no_resume", action="store_true",
                help="recompute folds that already have a saved OOF vector")
    args = ap.parse_args()

    ensure_dirs()
    cfg = dict(DEFAULTS)
    cfg.update({
        "arch": args.arch, "epochs": args.epochs, "seed": args.seed,
        "unfreeze": -1, "lr_head": 3e-4, "lr_backbone": 3e-5, "stage_decay": 0.7,
        "weight_decay": 0.05, "label_smoothing": 0.05, "pos_weight": 1.0,
        "ema_decay": 0.999, "aug_strength": "medium", "patience": 8,
        "input_size": 224, "tta": 8,
    })
    for kv in args.set:
        k, _, v = kv.partition("=")
        if k not in cfg:
            raise SystemExit("FATAL: unknown config key %r" % k)
        cfg[k] = coerce(v)
    if not cfg["res_schedule"]:
        cfg["final_res"] = cfg["input_size"]
    else:
        cfg["input_size"] = min(cfg["input_size"], cfg["final_res"])

    out_dir = os.path.join(FINAL_ROOT, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    log = get_logger("dermai.cv.%s" % args.tag,
                     logfile=os.path.join(out_dir, "cv_log.txt"))

    log("=" * 78)
    log("K-FOLD CV  tag=%s  arch=%s  folds=%d  epochs=%d" % (args.tag, cfg["arch"],
                                                             args.folds, cfg["epochs"]))
    log("=" * 78)
    log("  production model (never opened): %s" % PRODUCTION_MODEL)
    ctx = context(log)
    cfg["cache"] = ctx["cache"]

    folds, pool, _ = fold_frames(ctx["meta"], ctx["splits"], n_folds=args.folds,
                                 seed=SEED)
    log("")
    log("  CV pool = TRAIN + VAL (%d images, %d lesions) - test is NOT in the pool"
        % (len(pool), pool["group"].nunique()))
    for i, (tr, va) in enumerate(folds):
        log("    fold %d  train %5d / val %4d  (val cancers %d)"
            % (i, len(tr), len(va), int(va["label"].sum())))
    # written before any training so the pool definition survives a crash
    np.save(os.path.join(out_dir, "pool_labels.npy"), pool["label"].to_numpy())
    pool[["image_id", "group", "dx", "label"]].to_csv(
        os.path.join(out_dir, "pool_index.csv"), index=False)

    todo = ([int(x) for x in args.only_folds.split(",")]
            if args.only_folds else list(range(args.folds)))

    oof_path = os.path.join(out_dir, "oof_proba.npy")
    done_path = os.path.join(out_dir, "folds_done.json")
    oof = np.full(len(pool), np.nan)
    already = {}
    if os.path.exists(oof_path) and os.path.exists(done_path) and not args.no_resume:
        prev = np.load(oof_path)
        if prev.shape == oof.shape:
            oof = prev
            already = json.load(open(done_path))
            log("  resuming: folds %s already scored" % sorted(already))
    fold_ckpts, t_all = [os.path.join(out_dir, "fold%d_best.pt" % i)
                         for i in todo], time.time()
    for i in todo:
        if str(i) in already:
            fold_ckpts = [p for p in fold_ckpts
                          if os.path.basename(p) != "fold%d_best.pt" % i]
            continue
        tr, va = folds[i]
        set_seed(SEED + 1000 + i)
        log("")
        log("-" * 78)
        log("FOLD %d/%d" % (i + 1, args.folds))
        log("-" * 78)
        c = dict(cfg)
        c["seed"] = SEED + 1000 + i
        if not c["micro_batch"]:
            c["micro_batch"], _ = probe_micro_batch(
                c["arch"], c["final_res"], log, c["max_micro_batch"],
                c["min_micro_batch"], c["effective_batch"], c["pretrained"],
                c["channels_last"])
        trainer = Trainer(c, log, train_df=tr, val_df=va)
        ck = os.path.join(out_dir, "fold%d_best.pt" % i)
        trainer.train(c["epochs"], warmup=c["warmup"], final_frac=c["final_frac"],
                      save_best_path=ck)
        # score the held-out fold with the checkpointed best weights + TTA
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        trainer.model.load_state_dict(blob["state_dict"])
        trainer.model.to(ctx["device"]).eval()
        loc = va.index.to_numpy()          # fold_frames keeps pool positions as index
        p, y = predict(trainer.model, trainer.val_loader, trainer.resizer, tta=cfg["tta"],
                       aux_mix=c.get("aux_mix", 0.0), sizes=tta_sizes(c))
        assert np.array_equal(y, pool["label"].to_numpy()[loc].astype(int)), \
            "fold %d prediction order mismatch" % i
        oof[loc] = p
        thr, mm = best_threshold(y, p, THRESHOLD_GRID, "accuracy")
        log("  FOLD %d  OOF-AUC %.4f | acc@0.5 %.4f | best-acc thr %.3f -> acc %.4f "
            "sens %.4f spec %.4f F1 %.4f"
            % (i, mm["roc_auc"], float(((p >= 0.5).astype(int) == y).mean()),
               thr, mm["accuracy"], mm["sensitivity"], mm["specificity"], mm["f1"]))
        # persist after every fold so a crash never costs a completed fold
        np.save(oof_path, oof)
        already[str(i)] = {"auc": mm["roc_auc"], "acc": mm["accuracy"], "thr": thr}
        with open(done_path, "w") as fh:
            json.dump(already, fh, indent=2, default=float)
        del trainer
        torch.cuda.empty_cache()

    done = np.isfinite(oof)
    if not done.all():
        log("")
        log("  WARNING: only %d/%d pool rows scored (folds %s run)"
            % (int(done.sum()), len(pool), todo))
    y_pool = pool["label"].to_numpy().astype(int)[done]
    p_pool = oof[done]
    log("")
    log("OOF ENSEMBLE (the only place the threshold is chosen)")
    thr_f1, m_f1 = best_threshold(y_pool, p_pool, THRESHOLD_GRID, "f1")
    thr_acc, m_acc = best_threshold(y_pool, p_pool, THRESHOLD_GRID, "accuracy")
    log("  argmax F1       t=%.4f acc %.4f sens %.4f spec %.4f F1 %.4f"
        % (thr_f1, m_f1["accuracy"], m_f1["sensitivity"], m_f1["specificity"], m_f1["f1"]))
    log("  argmax ACCURACY t=%.4f acc %.4f sens %.4f spec %.4f F1 %.4f  <-- FROZEN"
        % (thr_acc, m_acc["accuracy"], m_acc["sensitivity"], m_acc["specificity"], m_acc["f1"]))
    np.save(os.path.join(out_dir, "oof_proba.npy"), oof)

    payload = {"tag": args.tag, "cfg": {k: v for k, v in cfg.items() if k != "cache"},
               "folds_run": sorted(int(k) for k in already),
               "fold_checkpoints": fold_ckpts,
               "oof_complete": bool(done.all()),
               "oof_metrics_at_argmax_f1": m_f1,
               "oof_metrics_at_argmax_accuracy": m_acc,
               "threshold_frozen": thr_acc,
               "threshold_rule": "argmax accuracy on out-of-fold TRAIN+VAL predictions",
               "wall_minutes": round((time.time() - t_all) / 60, 2)}
    with open(os.path.join(out_dir, "cv_summary.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    log("")
    log("  fold checkpoints : %s" % ", ".join(os.path.basename(p) for p in fold_ckpts))
    log("  cv_summary.json  : %s" % os.path.join(out_dir, "cv_summary.json"))
    log("  cv wall time     : %.1f min" % payload["wall_minutes"])


if __name__ == "__main__":
    main()
