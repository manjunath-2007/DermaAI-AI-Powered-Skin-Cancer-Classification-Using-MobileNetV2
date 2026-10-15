"""Two-step final pipeline, deliberately separated:

  1. `--freeze`  reads only TRAIN/VAL out-of-fold predictions, picks the ensemble members,
                 their weights and the decision threshold, and writes frozen_decision.json.
                 The TEST split is not touched.
  2. `--evaluate` loads frozen_decision.json, runs the frozen ensemble on the TEST split
                 exactly once, and writes the final metrics.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from .common import (CLASS_NAMES, FINAL_ROOT, PRODUCTION_MODEL, SEED, THRESHOLD_GRID,
                     ensure_dirs, get_logger, set_seed)
from .cv import tta_sizes, weighted_blend
from .data import CachedHAM
from .engine import Resizer, Trainer, best_threshold, metrics_at, predict
from .models import build_model
from .run_exp import coerce, context


# --------------------------------------------------------------------------- members
def collect_tags(filters):
    tags = sorted(d for d in os.listdir(FINAL_ROOT)
                  if os.path.isdir(os.path.join(FINAL_ROOT, d))
                  and os.path.exists(os.path.join(FINAL_ROOT, d, "cv_summary.json")))
    return [t for t in tags if any(f in t for f in filters)] if filters else tags


def load_member(tag):
    d = os.path.join(FINAL_ROOT, tag)
    s = json.load(open(os.path.join(d, "cv_summary.json")))
    oof = np.load(os.path.join(d, "oof_proba.npy"))
    return {"tag": tag, "dir": d, "summary": s, "oof": oof,
            "folds": sorted(glob.glob(os.path.join(d, "fold*_best.pt"))),
            "cfg": s["cfg"]}


# --------------------------------------------------------------------------- freeze
def cmd_freeze(args, log):
    tags = collect_tags(args.tags)
    if not tags:
        raise SystemExit("FATAL: no CV runs found under %s" % FINAL_ROOT)
    members = [load_member(t) for t in tags]
    pool = np.load(os.path.join(members[0]["dir"], "pool_labels.npy")).astype(int)
    for m in members:
        assert np.array_equal(pool, np.load(os.path.join(m["dir"], "pool_labels.npy")).astype(int)), \
            "CV runs used different pools - cannot ensemble"
        assert m["oof"].shape[0] == len(pool)

    oof_list = [m["oof"] for m in members]
    names = [m["tag"] for m in members]
    log("OUT-OF-FOLD ENSEMBLE SELECTION (TRAIN+VAL only; test not loaded)")
    for m, p in zip(members, oof_list):
        ok = np.isfinite(p)
        if not ok.all():
            log("  WARNING %-28s has %d unscored pool rows - excluded"
                % (m["tag"], int((~ok).sum())))
    keep = [i for i, p in enumerate(oof_list) if np.isfinite(p).all()]
    if not keep:
        raise SystemExit("FATAL: no member has a complete OOF vector.")
    names = [names[i] for i in keep]
    oof_list = [oof_list[i] for i in keep]

    for nm, p in zip(names, oof_list):
        thr, m = best_threshold(pool, p, THRESHOLD_GRID, "accuracy")
        log("  %-34s OOF AUC %.4f | acc@0.5 %.4f | acc@t*.4f %.4f (t=%.3f)"
            % (nm, m["roc_auc"], float(((p >= 0.5).astype(int) == pool).mean()),
               m["accuracy"], thr))

    uniform = weighted_blend(oof_list)
    _, mu = best_threshold(pool, uniform, THRESHOLD_GRID, "accuracy")
    picks, counts, blend = greedy(pool, oof_list, names, log=log)
    _, mb = best_threshold(pool, blend, THRESHOLD_GRID, "accuracy")
    log("")
    log("  uniform blend OOF AUC %.4f acc %.4f" % (mu["roc_auc"], mu["accuracy"]))
    log("  greedy   blend OOF AUC %.4f acc %.4f" % (mb["roc_auc"], mb["accuracy"]))
    if mb["roc_auc"] < mu["roc_auc"]:
        log("  -> greedy lost AUC to the uniform blend; using the uniform blend")
        counts = {n: 1.0 / len(names) for n in names}
        blend = uniform
    thr, m = best_threshold(pool, blend, THRESHOLD_GRID, "accuracy")
    thr_f1, mf1 = best_threshold(pool, blend, THRESHOLD_GRID, "f1")
    log("")
    log("  ENSEMBLE (logit-space) OOF:")
    log("    weights      : %s" % json.dumps(counts))
    log("    argmax ACC   : t=%.4f acc %.4f sens %.4f spec %.4f prec %.4f F1 %.4f AUC %.4f"
        % (thr, m["accuracy"], m["sensitivity"], m["specificity"], m["precision"],
           m["f1"], m["roc_auc"]))
    log("    argmax F1    : t=%.4f acc %.4f (reported, not used)"
        % (thr_f1, mf1["accuracy"]))

    decision = {
        "members": [{"tag": nm, "folds": [os.path.basename(f)
                                          for f in glob.glob(os.path.join(FINAL_ROOT, nm,
                                                                        "fold*_best.pt"))]}
                    for nm in names],
        "weights": counts,
        "threshold": thr,
        "threshold_rule": ("argmax accuracy on out-of-fold predictions of the frozen "
                           "TRAIN+VAL ensemble"),
        "oof_metrics": m,
        "oof_metrics_at_f1_threshold": mf1,
        "oof_alternative_threshold": thr_f1,
        "tta_views": 8,
        "per_member_tta_sizes": {m["tag"]: tta_sizes(
            json.load(open(os.path.join(FINAL_ROOT, m["tag"], "cv_summary.json")))["cfg"])
            for m in decision["members"]},
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(FINAL_ROOT, "frozen_decision.json")
    with open(path, "w") as fh:
        json.dump(decision, fh, indent=2, default=float)
    log("")
    log("  FROZEN -> %s" % path)
    log("  (test split has NOT been read)")
    return decision


def greedy(y, oof_list, names, max_k=10, log=print, objective="roc_auc"):
    """Caruana greedy forward selection with replacement.

    The objective is OOF ROC-AUC, NOT accuracy-at-the-best-threshold: with ~8.5k OOF
    rows the accuracy grid argmax is noisy enough that greedy on it latches onto
    whichever member got the luckiest threshold. AUC is smooth and threshold free; the
    single accuracy-optimal threshold is chosen once, at the end, on the final blend.
    """
    cur, best_hist, picks = None, [], []
    for step in range(max_k):
        best_i, best_score, best_blend = -1, -np.inf, None
        for i in range(len(oof_list)):
            cand = oof_list[i] if cur is None else weighted_blend(cur + [oof_list[i]])
            thr, mm = best_threshold(y, cand, THRESHOLD_GRID, "accuracy")
            if mm[objective] > best_score:
                best_i, best_score, best_blend = i, mm[objective], cand
        if cur is not None and best_score <= best_hist[-1] + 1e-9:
            log("  greedy stop after %d steps (no further OOF %s gain)" % (step, objective))
            break
        picks.append(best_i)
        best_hist.append(best_score)
        cur = [oof_list[best_i]] if cur is None else cur + [oof_list[best_i]]
        log("  greedy step %d: +%-34s OOF %s %.4f" % (step, names[best_i], objective,
                                                      best_score))
    counts = {}
    for i in picks:
        counts[names[i]] = counts.get(names[i], 0) + 1
    total = sum(counts.values())
    counts = {k: v / total for k, v in counts.items()}
    return picks, counts, weighted_blend(cur)


# --------------------------------------------------------------------------- evaluate
def cmd_evaluate(args, log):
    path = os.path.join(FINAL_ROOT, "frozen_decision.json")
    if not os.path.exists(path):
        raise SystemExit("FATAL: run --freeze first (no frozen_decision.json).")
    decision = json.load(open(path))
    log("=" * 78)
    log("FINAL TEST EVALUATION - single pass with the frozen ensemble")
    log("=" * 78)
    log("  frozen weights : %s" % json.dumps(decision["weights"]))
    log("  frozen threshold: %.6f (%s)" % (decision["threshold"],
                                           decision["threshold_rule"]))

    ctx = context(log)
    test = ctx["splits"]["test"]
    y = test["label"].to_numpy().astype(int)

    per_member, t0 = {}, time.time()
    for mem in decision["members"]:
        tag = mem["tag"]
        s = json.load(open(os.path.join(FINAL_ROOT, tag, "cv_summary.json")))
        c = dict(s["cfg"])
        c["cache"] = ctx["cache"]
        preds = []
        for fp in sorted(glob.glob(os.path.join(FINAL_ROOT, tag, "fold*_best.pt"))):
            cfg_run = dict(c)
            model = build_model(cfg_run["arch"], pretrained=False,
                                dropout=cfg_run["dropout"],
                                bn_norm=cfg_run["bn_norm"],
                                aux_classes=cfg_run.get("aux_classes", 0))
            if cfg_run.get("channels_last", True):
                model = model.to(memory_format=torch.channels_last)
            model = model.to(ctx["device"])
            blob = torch.load(fp, map_location="cpu", weights_only=False)
            model.load_state_dict(blob["state_dict"])
            model.eval()
            res = Resizer(cfg_run["input_size"], ctx["device"], channels_last=cfg_run.get("channels_last", True))
            loader = torch.utils.data.DataLoader(CachedHAM(ctx["cache"], test),
                                                 batch_size=int(cfg_run["micro_batch"]) * 2,
                                                 shuffle=False, pin_memory=True)
            p, yt = predict(model, loader, res, tta=decision["tta_views"],
                              aux_mix=cfg_run.get("aux_mix", 0.0),
                              sizes=tta_sizes(cfg_run))
            assert np.array_equal(yt, y), "test prediction order mismatch"
            preds.append(p)
            del model
            torch.cuda.empty_cache()
        per_member[tag] = weighted_blend(preds)
        log("  member %-30s %d folds scored" % (tag, len(preds)))

    tags = list(decision["weights"].keys())
    w = [decision["weights"][t] for t in tags]
    p_test = weighted_blend([per_member[t] for t in tags], w)
    thr = float(decision["threshold"])
    m = metrics_at(y, p_test, thr)

    log("")
    log("TEST @ frozen threshold %.4f" % thr)
    log("  accuracy    %.4f" % m["accuracy"])
    log("  ROC-AUC     %.4f" % m["roc_auc"])
    log("  precision   %.4f" % m["precision"])
    log("  recall / sensitivity %.4f" % m["recall"])
    log("  specificity %.4f" % m["specificity"])
    log("  F1          %.4f" % m["f1"])
    log("  tn=%d fp=%d fn=%d tp=%d  (n=%d)" % (m["tn"], m["fp"], m["fn"], m["tp"], m["n"]))

    out = os.path.join(FINAL_ROOT, "test")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "test_proba.npy"), p_test)
    np.save(os.path.join(out, "test_true.npy"), y)
    for t in tags:
        np.save(os.path.join(out, "test_proba_%s.npy" % t), per_member[t])

    payload = {"frozen_decision": decision,
               "test_metrics": m,
               "per_member_test_metrics": {
                   t: metrics_at(y, per_member[t], thr) for t in tags},
               "test_images": int(len(y)),
               "test_lesions": int(test["group"].nunique()),
               "test_class_counts": test["label_name"].value_counts().to_dict(),
               "eval_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out, "test_metrics.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=float)

    from sklearn.metrics import classification_report, confusion_matrix
    pred = (p_test >= thr).astype(int)
    rep = classification_report(y, pred, target_names=CLASS_NAMES, digits=4,
                               zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    log("")
    log(rep)
    log("Confusion matrix (rows = true, cols = predicted)")
    log("%-14s %12s %12s" % ("", *CLASS_NAMES))
    for i, row in enumerate(cm):
        log("%-14s %12d %12d" % (CLASS_NAMES[i], row[0], row[1]))
    with open(os.path.join(out, "classification_report.txt"), "w") as fh:
        fh.write("DermaAI - final ensemble - TEST classification report\n")
        fh.write("=" * 62 + "\n")
        fh.write("threshold      : %.6f (%s)\n" % (thr, decision["threshold_rule"]))
        fh.write("ensemble       : %s\n" % json.dumps(decision["weights"]))
        fh.write("TTA views      : %d (dihedral)\n" % decision["tta_views"])
        fh.write("test split     : evaluated exactly once\n\n")
        fh.write(rep)
        fh.write("\nConfusion matrix (rows = true, columns = predicted)\n")
        fh.write("%-22s %12s %12s\n" % ("", *CLASS_NAMES))
        for i, row in enumerate(cm):
            fh.write("%-22s %12d %12d\n" % (CLASS_NAMES[i], row[0], row[1]))
    make_figures(y, p_test, thr, m)
    log("")
    log("  artifacts -> %s" % out)


def make_figures(y, p, thr, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, roc_curve

    out = os.path.join(FINAL_ROOT, "test")
    fpr, tpr, _ = roc_curve(y, p)
    np.savetxt(os.path.join(out, "roc_curve_test.csv"),
               np.column_stack([fpr, tpr]), delimiter=",", header="fpr,tpr", comments="")
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.plot(fpr, tpr, lw=2, label="test AUC = %.4f" % m["roc_auc"])
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=.5, label="chance")
    ax.plot([1 - m["specificity"]], [m["sensitivity"]], "ro", ms=8,
            label="operating point t=%.4f" % thr)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("DermaAI final ensemble - TEST ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "roc_curve_test.png"), dpi=130)
    plt.close(fig)

    cm = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1])
    np.savetxt(os.path.join(out, "confusion_matrix_test.csv"), cm, delimiter=",",
               header=",".join(CLASS_NAMES), comments="rows = true")
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("TEST confusion matrix @ t=%.4f\nacc %.4f | sens %.4f | spec %.4f"
                 % (thr, m["accuracy"], m["sensitivity"], m["specificity"]))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "confusion_matrix_test.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--tags", nargs="*", default=[])
    args = ap.parse_args()
    ensure_dirs()
    log = get_logger("dermai.final", logfile=os.path.join(FINAL_ROOT, "final_log.txt"))
    if not args.freeze and not args.evaluate:
        raise SystemExit("FATAL: pass --freeze or --evaluate")
    if args.freeze:
        cmd_freeze(args, log)
    if args.evaluate:
        cmd_evaluate(args, log)


if __name__ == "__main__":
    main()