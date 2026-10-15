"""Post-hoc selection on VALIDATION for one finished run: sweep the auxiliary-head blend
weight and the TTA scale set using the saved best checkpoint (no retraining, no test).

    python -m dermai_exp.analyze_run --run c01_aux7
"""

import argparse
import json
import os

import numpy as np
import torch

from .common import EXP_ROOT, THRESHOLD_GRID, ensure_dirs, get_logger, set_seed
from .engine import Resizer, best_threshold, predict
from .data import CachedHAM, CachedHAMAux
from .models import build_model
from .run_exp import context
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    ensure_dirs()
    log = get_logger("dermai.an", logfile=os.path.join(EXP_ROOT, "analyze_log.txt"))
    run_dir = os.path.join(EXP_ROOT, args.run)
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    cfg = summary["cfg"]
    blob = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu",
                      weights_only=False)
    kind = blob["best"].get("kind", "raw")
    log("run %s | best %s ep %s val_auc %.4f" % (args.run, kind,
                                                 blob["best"].get("epoch"),
                                                 blob["best"]["val_auc"]))
    set_seed(cfg["seed"])
    ctx = context(log)
    model = build_model(cfg["arch"], pretrained=False, dropout=cfg["dropout"],
                        bn_norm=cfg["bn_norm"],
                        aux_classes=cfg.get("aux_classes", 0))
    if cfg.get("channels_last", True):
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(blob["state_dict"])
    model.to(ctx["device"]).eval()

    ds_cls = CachedHAMAux if cfg.get("aux_classes", 0) else CachedHAM
    val = ctx["splits"]["val"]
    loader = DataLoader(ds_cls(ctx["cache"], val), batch_size=32, shuffle=False,
                        pin_memory=True)
    res = Resizer(cfg["input_size"], ctx["device"],
                  channels_last=cfg.get("channels_last", True))
    y = val["label"].to_numpy().astype(int)

    aux_mixes = [0.0, 0.15, 0.3, 0.45, 0.6] if cfg.get("aux_classes", 0) else [0.0]
    scale_sets = [[224], [256], [288], [224, 256], [256, 288], [224, 256, 288]]
    rows = []
    log("")
    log("  %-6s %-14s %8s %8s %8s %8s %8s %8s"
        % ("auxmix", "scales", "AUC", "thr", "acc", "sens", "spec", "F1"))
    for am in aux_mixes:
        for ss in scale_sets:
            p, yy = predict(model, loader, res, tta=8, sizes=ss, aux_mix=am)
            assert np.array_equal(yy, y)
            thr, m = best_threshold(y, p, THRESHOLD_GRID, "accuracy")
            rows.append({"aux_mix": am, "scales": ss, "metrics": m})
            log("  %-6.2f %-14s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f"
                % (am, "+".join(map(str, ss)), m["roc_auc"], thr, m["accuracy"],
                   m["sensitivity"], m["specificity"], m["f1"]))
    best = max(rows, key=lambda r: r["metrics"]["roc_auc"])
    besta = max(rows, key=lambda r: r["metrics"]["accuracy"])
    log("")
    log("  best AUC      : aux_mix=%.2f scales=%s AUC %.4f"
        % (best["aux_mix"], best["scales"], best["metrics"]["roc_auc"]))
    log("  best ACCURACY : aux_mix=%.2f scales=%s acc %.4f (t=%.4f)"
        % (besta["aux_mix"], besta["scales"], besta["metrics"]["accuracy"],
           besta["metrics"]["threshold"]))
    with open(os.path.join(run_dir, "posthoc_tta.json"), "w") as fh:
        json.dump({"best_auc": best, "best_accuracy": besta, "grid": rows},
                  fh, indent=2, default=float)


if __name__ == "__main__":
    main()