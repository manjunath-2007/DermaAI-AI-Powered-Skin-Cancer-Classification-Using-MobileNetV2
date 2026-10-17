"""Complete validation for c03_heavy_mixup using existing checkpoint."""
import json
import os
import time

import numpy as np
import torch

from dermai_exp.common import EXP_ROOT, SEED, THRESHOLD_GRID, get_logger, set_seed
from dermai_exp.engine import Trainer, best_threshold, predict, metrics_at
from dermai_exp.run_exp import DEFAULTS, coerce, context, probe_micro_batch

log = get_logger("dermai.complete_c03", logfile=os.path.join(EXP_ROOT, "sweep_log.txt"))

def complete_validation(name, cfg, ctx):
    run_dir = os.path.join(EXP_ROOT, name)
    os.makedirs(run_dir, exist_ok=True)
    log("")
    log("=" * 78)
    log("COMPLETE VALIDATION %s" % name)
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
    
    # Load the existing best checkpoint
    blob = torch.load(best_path, map_location="cpu", weights_only=False)
    best_info = blob.get("best", {})
    if cfg["ema_decay"] > 0 and best_info.get("kind") == "ema" and tr.ema is not None:
        tr.ema.ema.load_state_dict(blob["state_dict"])
        tr.ema.ema.to(ctx["device"]).eval()
        loaded_kind = "ema"
    else:
        tr.model.load_state_dict(blob["state_dict"])
        tr.model.to(ctx["device"]).eval()
        loaded_kind = "raw"
    log("  loaded best checkpoint: %s epoch %d val_auc %.4f"
        % (best_info.get("kind"), best_info.get("epoch"), best_info.get("val_auc")))

    vy = ctx["splits"]["val"]["label"].to_numpy().astype(int)
    res = {"best": blob}
    base = cfg["input_size"]
    plans = [("r%d" % base, 8, [base]),
             ("r256", 8, [256]),
             ("ms", 8, sorted({base, 256})),
             ("ms3", 8, sorted({base, 256, 288}))]
    
    for kind in ("raw", "ema"):
        m = tr.eval_model(kind)
        if m is None:
            continue
        for tag, tta, sizes in plans:
            key = "%s_%s" % (kind, tag)
            npy_path = os.path.join(run_dir, "val_proba_%s.npy" % key)
            if os.path.exists(npy_path):
                log("  %-10s %-14s SKIP (already exists)" % (key, "+".join(str(s) for s in sizes)))
                # Load existing probs to compute metrics
                p = np.load(npy_path)
                assert np.array_equal(p.shape[0], vy.shape[0])
                thr_acc, m_acc = best_threshold(vy, p, THRESHOLD_GRID, "accuracy")
                thr_f1, m_f1 = best_threshold(vy, p, THRESHOLD_GRID, "f1")
                res[key] = {"val_auc": m_acc["roc_auc"], "thr_acc": thr_acc,
                            "eval_sizes": sizes, "tta": tta,
                            "thr_acc_metrics": m_acc, "thr_f1": thr_f1,
                            "val_acc_at_050": float(((p >= 0.5).astype(int) == vy).mean())}
                log("  %-10s %-14s AUC %.4f | acc@0.5 %.4f | t=%.3f acc %.4f "
                    "sens %.4f spec %.4f F1 %.4f"
                    % (key, "+".join(str(s) for s in sizes), m_acc["roc_auc"],
                       res[key]["val_acc_at_050"], thr_acc, m_acc["accuracy"],
                       m_acc["sensitivity"], m_acc["specificity"], m_acc["f1"]))
                continue
            
            p, y = predict(m, tr.val_loader, tr.resizer, tta=tta, sizes=sizes,
                           aux_mix=cfg.get("aux_mix", 0.0))
            assert np.array_equal(y, vy)
            thr_acc, m_acc = best_threshold(y, p, THRESHOLD_GRID, "accuracy")
            thr_f1, m_f1 = best_threshold(y, p, THRESHOLD_GRID, "f1")
            res[key] = {"val_auc": m_acc["roc_auc"], "thr_acc": thr_acc,
                        "eval_sizes": sizes, "tta": tta,
                        "thr_acc_metrics": m_acc, "thr_f1": thr_f1,
                        "val_acc_at_050": float(((p >= 0.5).astype(int) == y).mean())}
            np.save(npy_path, p)
            log("  %-10s %-14s AUC %.4f | acc@0.5 %.4f | t=%.3f acc %.4f "
                "sens %.4f spec %.4f F1 %.4f"
                % (key, "+".join(str(s) for s in sizes), m_acc["roc_auc"],
                   res[key]["val_acc_at_050"], thr_acc, m_acc["accuracy"],
                   m_acc["sensitivity"], m_acc["specificity"], m_acc["f1"]))
        tr.resizer.eval_size = tr.eval_size
    
    cand = [k for k in res if isinstance(res[k], dict) and "thr_acc_metrics" in res[k]]
    res["selection"] = max(cand, key=lambda k: res[k]["thr_acc_metrics"]["accuracy"])
    res["selection_by_auc"] = max(cand, key=lambda k: res[k]["val_auc"])
    
    # We don't have training history since we didn't train, but we can load from checkpoint
    summary = {"name": name, "cfg": {k: v for k, v in cfg.items() if k != "cache"},
               "result": res, "split_matches_production": ctx["exact"],
               "train_minutes": 0, "history": blob.get("history", [])}
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    log("  -> selection %s | %s"
        % (res["selection"], os.path.join(run_dir, "summary.json")))
    del tr
    torch.cuda.empty_cache()
    return summary


def main():
    plan = json.load(open(os.path.join(EXP_ROOT, "plan_c.json")))
    ctx = context(log)
    
    for spec in plan:
        name = spec["name"]
        if name != "c03_heavy_mixup":
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
        
        complete_validation(name, cfg, ctx)


if __name__ == "__main__":
    main()