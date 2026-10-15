"""DermaAI model / label-mapping verification harness.

Rebuilds the EXACT skin_cancer1.ipynb split from the local HAM10000 copy
(seed 42, lesion-level stratified 70/15/15), then evaluates every candidate
model on validation + test with the preprocessing that model was trained with.

Run:  python verify_models.py
"""
import csv
import json
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

Image.MAX_IMAGE_PIXELS = None
SEED = 42
HAM = r"D:\User\Downloads\archive"
APP = r"C:\Users\Manjunath\OneDrive\Desktop\MINI3\skin-cancer-app-simple\skin-cancer-app"
DL = r"D:\User\Downloads"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

CANCER_CLASSES = ["mel", "bcc", "akiec"]


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- split
def build_split():
    meta = pd.read_csv(os.path.join(HAM, "HAM10000_metadata.csv"))
    paths = {}
    for part in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
        d = os.path.join(HAM, part)
        for f in os.listdir(d):
            if f.lower().endswith(".jpg"):
                paths[os.path.splitext(f)[0]] = os.path.join(d, f)
    meta["path"] = meta["image_id"].map(paths)
    meta = meta[meta["path"].notna()].reset_index(drop=True)
    meta["label"] = meta["dx"].apply(lambda x: 1 if x in CANCER_CLASSES else 0)
    meta["label_name"] = meta["label"].map({0: "Non-Cancer", 1: "Cancer"})

    # EXACT replication of skin_cancer1.ipynb cell 012/013
    lesion_df = meta.groupby("lesion_id")["label"].max().reset_index()
    train_lesions, temp_lesions = train_test_split(
        lesion_df, test_size=0.30, stratify=lesion_df["label"], random_state=SEED)
    val_lesions, test_lesions = train_test_split(
        temp_lesions, test_size=0.50, stratify=temp_lesions["label"], random_state=SEED)
    tr = meta[meta["lesion_id"].isin(train_lesions["lesion_id"])].copy().reset_index(drop=True)
    va = meta[meta["lesion_id"].isin(val_lesions["lesion_id"])].copy().reset_index(drop=True)
    te = meta[meta["lesion_id"].isin(test_lesions["lesion_id"])].copy().reset_index(drop=True)
    return meta, tr, va, te


# ------------------------------------------------------------ preprocess
def prep_rescale_255(x):
    return x / 255.0


def prep_mobilenet(x):
    return x / 127.5 - 1.0


def prep_identity(x):
    return x


PREPROCS = {"255": prep_rescale_255, "mobilenet": prep_mobilenet,
            "raw255": prep_identity}


def predict(model, df, size, prep, batch=32):
    fn = PREPROCS[prep]
    probs = []
    paths = df["path"].values
    for i in range(0, len(paths), batch):
        chunk = []
        for p in paths[i:i + batch]:
            im = Image.open(p).convert("RGB").resize((size, size))
            chunk.append(np.asarray(im, dtype=np.float32))
        x = np.stack(chunk)
        x = fn(x)[..., None] if False else fn(x)
        probs.append(model.predict(x, verbose=0).ravel())
    return np.concatenate(probs)


def eval_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y)),
    }


def best_threshold(y, p, metric="f1"):
    best, best_s = 0.50, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        s = (f1_score(y, (p >= t).astype(int), zero_division=0) if metric == "f1"
             else accuracy_score(y, (p >= t).astype(int)))
        if s > best_s:
            best, best_s = float(t), s
    return best, best_s


# --------------------------------------------------------------- models
def describe(path):
    info = {"path": path, "exists": os.path.exists(path)}
    if not info["exists"]:
        return info
    info["size_mb"] = round(os.path.getsize(path) / 1e6, 2)
    custom = {}
    try:
        import zipfile
        with zipfile.ZipFile(path) as z:
            meta = z.read("metadata.json").decode("utf-8", "replace") if "metadata.json" in z.namelist() else ""
        info["keras_meta_loss"] = [w.strip() for w in meta.split("\n") if "loss" in w.lower()][:4]
    except Exception:
        pass
    try:
        m = tf.keras.models.load_model(path, compile=False)
        info["input_shape"] = list(m.input_shape)
        info["output_shape"] = list(m.output_shape)
        info["params"] = int(m.count_params())
        last = m.layers[-1]
        info["last_layer"] = [type(last).__name__, last.name,
                              last.get_config().get("activation", None),
                              last.get_config().get("units", None)]
        info["ok"] = True
        del m
    except Exception as exc:
        info["ok"] = False
        info["error"] = "%s: %s" % (type(exc).__name__, exc)
    return info


CANDIDATES = [
    ("PRODUCTION skin_cancer_mobilenetv2.h5",
     os.path.join(APP, "model", "skin_cancer_mobilenetv2.h5"), 224, ["255", "mobilenet", "raw255"]),
    ("skin_cancer1 MobileNetV2 (focal-free, BCE)",
     os.path.join(DL, "skin_cancer_mobilenetv2_final.keras"), 224, ["mobilenet", "255"]),
    ("skin_cancer1 EfficientNetB2 focal",
     os.path.join(DL, "skin_cancer_efficientnetb2_final.keras"), 260, ["raw255", "255"]),
    ("skin_cancer1 EfficientNetB2 accuracy",
     os.path.join(DL, "skin_cancer_efficientnetb2_accuracy.keras"), 260, ["raw255", "255"]),
]


def main():
    log("=" * 90)
    log("STEP 1  Rebuild split (skin_cancer1.ipynb logic, seed=42)")
    log("=" * 90)
    meta, tr, va, te = build_split()
    log("total images            : %d" % len(meta))
    log("unique lesions          : %d" % meta.lesion_id.nunique())
    log("train  images=%d lesions=%d  %s" % (len(tr), tr.lesion_id.nunique(),
                                             tr.label_name.value_counts().to_dict()))
    log("val    images=%d lesions=%d  %s" % (len(va), va.lesion_id.nunique(),
                                             va.label_name.value_counts().to_dict()))
    log("test   images=%d lesions=%d  %s" % (len(te), te.lesion_id.nunique(),
                                             te.label_name.value_counts().to_dict()))
    expect = {"train": (7010, 5229, {"Non-Cancer": 5641, "Cancer": 1369}),
              "val": (1500, 1120, {"Non-Cancer": 1203, "Cancer": 297}),
              "test": (1505, 1121, {"Non-Cancer": 1217, "Cancer": 288})}
    split_ok = True
    for nm, d in (("train", tr), ("val", va), ("test", te)):
        e = expect[nm]
        ok = len(d) == e[0] and d.lesion_id.nunique() == e[1] and d.label_name.value_counts().to_dict() == e[2]
        split_ok &= ok
        log("  notebook-match %-6s : %s" % (nm, ok))
    log("SPLIT REPRODUCES NOTEBOOK EXACTLY: %s" % split_ok)
    log("")

    log("=" * 90)
    log("STEP 2  Candidate model inventory")
    log("=" * 90)
    inventory = []
    for name, path, size, preps in CANDIDATES:
        d = describe(path)
        d["name"] = name
        inventory.append(d)
        log("%-42s exists=%s size=%sMB ok=%s input=%s out=%s last=%s"
            % (name, d.get("exists"), d.get("size_mb"), d.get("ok"),
               d.get("input_shape"), d.get("output_shape"), d.get("last_layer")))
        if d.get("error"):
            log("      error: %s" % d["error"])
    log("")

    log("=" * 90)
    log("STEP 3  Evaluate every candidate (threshold chosen on VALIDATION only)")
    log("=" * 90)
    results = {}
    for name, path, size, preps in CANDIDATES:
        if not os.path.exists(path):
            continue
        for prep in preps:
            try:
                m = tf.keras.models.load_model(path, compile=False)
            except Exception as exc:
                log("[skip] %s (%s): %s" % (name, prep, exc))
                continue
            if m.input_shape[1] != size:
                log("[skip] %s (%s): declared size %s != %s" % (name, prep, m.input_shape[1], size))
                del m
                continue
            key = "%s | %s" % (name, prep)
            vp = predict(m, va, size, prep)
            del m
            thr, vscore = best_threshold(va.label.values, vp, "f1")
            tp = predict(tf.keras.models.load_model(path, compile=False), te, size, prep)
            v = eval_at(va.label.values, vp, thr)
            t = eval_at(te.label.values, tp, thr)
            t050 = eval_at(te.label.values, tp, 0.50)
            results[key] = {"val": v, "test": t, "test_at_0.50": t050, "prep": prep, "size": size}
            log("\n--- %s" % key)
            log("    VAL  thr=%.2f acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f  (valF1@thr=%.4f)"
                % (thr, v["accuracy"], v["precision"], v["recall"], v["f1"], v["roc_auc"], vscore))
            log("    TEST thr=%.2f acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f  [tn=%d fp=%d fn=%d tp=%d]"
                % (thr, t["accuracy"], t["precision"], t["recall"], t["f1"], t["roc_auc"],
                   t["tn"], t["fp"], t["fn"], t["tp"]))
            log("    TEST@0.50 (app.py current) acc=%.4f prec=%.4f rec=%.4f f1=%.4f"
                % (t050["accuracy"], t050["precision"], t050["recall"], t050["f1"]))
            log("    TEST classification report:")
            log(classification_report(te.label.values, (tp >= thr).astype(int),
                                      target_names=["Non-Cancer", "Cancer"], digits=4))
            fpr, tpr, _ = roc_curve(te.label.values, tp)
            log("    ROC: %d points, auc=%.4f" % (len(fpr), t["roc_auc"]))

    with open(os.path.join(OUT, "verification.json"), "w", encoding="utf-8") as fh:
        json.dump({"split_ok": bool(split_ok),
                   "split": {"train": len(tr), "val": len(va), "test": len(te)},
                   "inventory": inventory, "results": results}, fh, indent=2)
    log("\nJSON written to %s" % os.path.join(OUT, "verification.json"))


if __name__ == "__main__":
    main()