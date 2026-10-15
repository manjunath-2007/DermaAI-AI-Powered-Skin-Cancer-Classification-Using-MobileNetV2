"""Screening driver: run several configs back to back in ONE process so the dataset is
parsed and the image cache mmapped once.

    python -m dermai_exp.sweep --plan plan_a.json
    python -m dermai_exp.sweep --plan plan_a.json --only b03,b05
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from .common import (EXP_ROOT, FINAL_ROOT, PRODUCTION_MODEL, SEED, THRESHOLD_GRID,
                     ensure_dirs, get_logger, set_seed)
from .engine import Trainer, best_threshold, metrics_at, predict
from .run_exp import DEFAULTS, coerce, context, probe_micro_batch


def run_one(name, cfg, ctx, log, tta_list=(1, 4, 8)):
    run_dir = os.path.join(EXP_ROOT, name)
    os.makedirs(run_dir, exist_ok=True)
    log("")
    log("=" * 78)
    log("EXPERIMENT %s" % name)
    log("=" * 78)
    log("  config: %s" % json.dumps(cfg, sort_keys=True))
    set_seed(cfg["seed"])
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

    # the checkpointed best-by-val-AUC weights are what gets validated / reported
    blob = torch.load(best_path, map_location="cpu", weights_only=False)
    if cfg["ema_decay"] > 0 and best.get("kind") == "ema" and tr.ema is not None:
        tr.ema.ema.load_state_dict(blob["state_dict"])
        tr.ema.ema.to(ctx["device"]).eval()
    else:
        tr.model.load_state_dict(blob["state_dict"])
        tr.model.to(ctx["device"]).eval()
    log("  loaded best checkpoint: %s epoch %d val_auc %.4f"
        % (best.get("kind"), best.get("epoch"), best.get("val_auc")))

    vy = ctx["splits"]["val"]["label"].to_numpy().astype(int)
    res = {"best": best}
    base = cfg["input_size"]
    # (label, tta views, list of evaluation resolutions)
    plans = [("r%d" % base, 8, [base]),
             ("r256", 8, [256]),
             ("ms", 8, sorted({base, 256})),
             ("ms3", 8, sorted({base, 256, 288}))]
    for kind in ("raw", "ema"):
        m = tr.eval_model(kind)
        if m is None:
            continue
        for tag, tta, sizes in plans:
            p, y = predict(m, tr.val_loader, tr.resizer, tta=tta, sizes=sizes,
                           aux_mix=cfg.get("aux_mix", 0.0))
            assert np.array_equal(y, vy)
            thr_acc, m_acc = best_threshold(y, p, THRESHOLD_GRID, "accuracy")
            thr_f1, m_f1 = best_threshold(y, p, THRESHOLD_GRID, "f1")
            key = "%s_%s" % (kind, tag)
            res[key] = {"val_auc": m_acc["roc_auc"], "thr_acc": thr_acc,
                        "eval_sizes": sizes, "tta": tta,
                        "thr_acc_metrics": m_acc, "thr_f1": thr_f1,
                        "val_acc_at_050": float(((p >= 0.5).astype(int) == y).mean())}
            np.save(os.path.join(run_dir, "val_proba_%s.npy" % key), p)
            log("  %-10s %-14s AUC %.4f | acc@0.5 %.4f | t=%.3f acc %.4f "
                "sens %.4f spec %.4f F1 %.4f"
                % (key, "+".join(str(s) for s in sizes), m_acc["roc_auc"],
                   res[key]["val_acc_at_050"], thr_acc, m_acc["accuracy"],
                   m_acc["sensitivity"], m_acc["specificity"], m_acc["f1"]))
        tr.resizer.eval_size = tr.eval_size
    cand = [k for k in res if isinstance(res[k], dict) and "thr_acc_metrics" in res[k]]
    res["selection"] = max(cand, key=lambda k: res[k]["thr_acc_metrics"]["accuracy"])
    res["selection_by_auc"] = max(cand, key=lambda k: res[k]["val_auc"])
    summary = {"name": name, "cfg": {k: v for k, v in cfg.items() if k != "cache"},
               "result": res, "split_matches_production": ctx["exact"],
               "train_minutes": round(train_min, 2), "history": tr.history}
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    log("  -> selection %s | train %.1f min | %s"
        % (res["selection"], train_min, os.path.join(run_dir, "summary.json")))
    del tr
    torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    ensure_dirs()
    plan = json.load(open(args.plan))
    log = get_logger("dermai.sweep", logfile=os.path.join(EXP_ROOT, "sweep_log.txt"))
    log("")
    log("#" * 78)
    log("SWEEP %s | %d experiments" % (os.path.basename(args.plan), len(plan)))
    log("#" * 78)
    log("  production model (never opened): %s" % PRODUCTION_MODEL)
    ctx = context(log)
    table = []
    for spec in plan:
        name = spec["name"]
        if args.only and name not in args.only:
            continue
        cfg = dict(DEFAULTS)
        for k, v in spec.get("cfg", {}).items():
            if k not in cfg:
                raise SystemExit("FATAL: unknown config key %r" % k)
            cfg[k] = coerce(str(v)) if not isinstance(v, (int, float, bool)) else v
        if not cfg["res_schedule"]:
            cfg["final_res"] = cfg["input_size"]
        else:
            cfg["input_size"] = min(cfg["input_size"], cfg["final_res"])
        try:
            s = run_one(name, cfg, ctx, log)
        except SystemExit as exc:
            log("  SKIPPED %s: %s" % (name, exc))
            continue
        except Exception as exc:
            import traceback
            log("  FAILED %s: %s" % (name, exc))
            log(traceback.format_exc())
            torch.cuda.empty_cache()
            continue
        sel = s["result"]["selection"]
        table.append((name, sel, s["result"][sel]["val_auc"],
                      s["result"][sel]["thr_acc"],
                      s["result"][sel]["thr_acc_metrics"]["accuracy"],
                      s["train_minutes"]))

    log("")
    log("=" * 78)
    log("SWEEP SUMMARY  (sorted by validation accuracy at the val-selected threshold)")
    log("=" * 78)
    log("  %-26s %-10s %8s %8s %8s %7s" % ("experiment", "sel", "valAUC", "thr",
                                           "valAcc", "min"))
    for row in sorted(table, key=lambda r: -r[4]):
        log("  %-26s %-10s %8.4f %8.3f %8.4f %7.1f" % row)
    with open(os.path.join(EXP_ROOT, "sweep_summary_%s.json"
                           % os.path.splitext(os.path.basename(args.plan))[0]), "w") as fh:
        json.dump(table, fh, indent=2, default=float)


if __name__ == "__main__":
    main()