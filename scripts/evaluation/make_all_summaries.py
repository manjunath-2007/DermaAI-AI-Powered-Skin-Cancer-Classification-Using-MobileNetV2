"""Create summary for c01_aux7 from existing npy files."""
import json
import os

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from dermai_exp.common import EXP_ROOT, THRESHOLD_GRID

def metrics_at(y, p, thr):
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
    rows = []
    for t in grid:
        pred = (p >= t).astype(int)
        rows.append((float(t), accuracy_score(y, pred),
                     precision_score(y, pred, zero_division=0),
                     recall_score(y, pred, zero_division=0),
                     f1_score(y, pred, zero_division=0)))
    return rows

def best_threshold(y, p, grid, key="accuracy"):
    rows = sweep(y, p, grid)
    idx_map = {"accuracy": 1, "f1": 4, "balanced_accuracy": 0}
    i = int(np.argmax([r[idx_map[key]] for r in rows]))
    return rows[i][0], metrics_at(y, p, rows[i][0])

def make_summary(name, train_minutes):
    run_dir = os.path.join(EXP_ROOT, name)
    
    # Load validation labels
    from dermai_exp.run_exp import context
    from dermai_exp.common import get_logger
    log = get_logger("dermai.make_summary")
    ctx = context(log)
    vy = ctx["splits"]["val"]["label"].to_numpy().astype(int)
    
    # Check existing npy files
    existing_keys = []
    for kind in ("raw", "ema"):
        for tag in ("r224", "r256", "ms", "ms3"):
            npy_path = os.path.join(run_dir, f"val_proba_{kind}_{tag}.npy")
            if os.path.exists(npy_path):
                p = np.load(npy_path)
                if p.shape == (1500,) and p.min() < 0.4 and p.max() > 0.6:
                    existing_keys.append((kind, tag, npy_path))
    
    print(f"Found valid npy files for {name}:")
    for kind, tag, path in existing_keys:
        p = np.load(path)
        print(f"  {kind}_{tag}: mean={p.mean():.4f}, min={p.min():.4f}, max={p.max():.4f}")
    
    # Compute metrics for each
    res = {}
    for kind, tag, npy_path in existing_keys:
        key = f"{kind}_{tag}"
        p = np.load(npy_path)
        thr_acc, m_acc = best_threshold(vy, p, THRESHOLD_GRID, "accuracy")
        thr_f1, m_f1 = best_threshold(vy, p, THRESHOLD_GRID, "f1")
        
        sizes_map = {"r224": [224], "r256": [256], "ms": [224, 256], "ms3": [224, 256, 288]}
        res[key] = {
            "val_auc": m_acc["roc_auc"], 
            "thr_acc": thr_acc,
            "eval_sizes": sizes_map[tag], 
            "tta": 8,
            "thr_acc_metrics": m_acc, 
            "thr_f1": thr_f1,
            "val_acc_at_050": float(((p >= 0.5).astype(int) == vy).mean())
        }
        print(f"  {key}: AUC={m_acc['roc_auc']:.4f}, acc@0.5={res[key]['val_acc_at_050']:.4f}, "
              f"thr={thr_acc:.3f}, acc={m_acc['accuracy']:.4f}, sens={m_acc['sensitivity']:.4f}, "
              f"spec={m_acc['specificity']:.4f}, F1={m_acc['f1']:.4f}")
    
    # Determine selection
    cand = [k for k in res if "thr_acc_metrics" in res[k]]
    if cand:
        selection = max(cand, key=lambda k: res[k]["thr_acc_metrics"]["accuracy"])
        selection_by_auc = max(cand, key=lambda k: res[k]["val_auc"])
        print(f"\nSelection by accuracy: {selection} (acc={res[selection]['thr_acc_metrics']['accuracy']:.4f})")
        print(f"Selection by AUC: {selection_by_auc} (AUC={res[selection_by_auc]['val_auc']:.4f})")
        
        # Load config from plan_c.json
        with open(os.path.join(EXP_ROOT, "plan_c.json")) as f:
            plan = json.load(f)
        cfg = None
        for spec in plan:
            if spec["name"] == name:
                cfg = spec["cfg"]
                break
        
        # Create summary without state_dict
        summary = {
            "name": name, 
            "cfg": cfg,
            "result": {
                **res,
                "selection": selection,
                "selection_by_auc": selection_by_auc
            },
            "split_matches_production": ctx["exact"],
            "train_minutes": train_minutes,
            "history": []
        }
        
        with open(os.path.join(run_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=float)
        print(f"Saved summary.json to {run_dir}")
        
        return selection, res[selection]["thr_acc_metrics"]["accuracy"], res[selection]["val_auc"], train_minutes
    return None

def main():
    make_summary("c01_aux7", 22.2)
    make_summary("c02_pw4", 24.2)
    make_summary("c03_heavy_mixup", 28.7)

if __name__ == "__main__":
    main()