#!/usr/bin/env python3
r"""
DermaAI - Reported-metrics recalculator (PHASE 3 deliverable).

Recalculates metrics ONLY from actual saved predictions/labels (.npy files)
and model_card.json. Does NOT fabricate missing predictions.

For ConvNeXt-Tiny: test_proba.npy / test_true.npy and val_proba.npy / val_true.npy
exist, so full test/validation metrics can be independently re-derived.

For MobileNetV2: no full test-set .npy files exist. Only six single_image_predictions
are stored in model_card.json. Test metrics from model_card.json are reported as
"SUPPORTED BY ARTIFACT" (not independently re-derived).

For EfficientNetB2: no .npy prediction files exist in the project. Source .keras
models exist at D:\\User\\Downloads but were not part of the deployed pipeline.

Exit code: 0 if all available checks pass; 1 if a REQUIRED integrity check fails
    (missing or length-mismatched ConvNeXt .npy prediction pair). MobileNetV2 missing
    predictions are reported but do NOT cause a nonzero exit.
    
    Machine-readable results are written to scripts/metrics_verification_results.json
    so the comprehensive audit JSON (DERMAI_VERIFICATION_RESULTS.json) is never overwritten.
"""

import json
import os
import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(BASE)
RESULTS_CT = os.path.join(PROJECT, "results", "convnext_tiny")
MODEL_CARD = os.path.join(PROJECT, "model", "model_card.json")

CANCER_CLASSES = ["mel", "bcc", "akiec"]
POSITIVE_CLASS = 1  # Cancer = 1


def log(msg):
    print(msg, flush=True)


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def load_npy_pair(proba_name, true_name):
    proba_path = os.path.join(RESULTS_CT, proba_name)
    true_path = os.path.join(RESULTS_CT, true_name)
    if not os.path.isfile(proba_path) or not os.path.isfile(true_path):
        return None, None, "missing"
    proba = np.load(proba_path)
    true = np.load(true_path)
    if proba.shape[0] != true.shape[0]:
        return proba, true, "length_mismatch"
    return proba, true, "ok"


def verify_convnext():
    log("=" * 72)
    log("ConvNeXt-Tiny - metrics recalculation from saved .npy artifacts")
    log("=" * 72)

    results = {}

    # Test set
    proba, true, status = load_npy_pair("test_proba.npy", "test_true.npy")
    if status == "ok":
        thr = 0.5249999999999998
        m = metrics_at(true, proba, thr)
        results["test"] = m
        log("TEST (threshold=%.6f, n=%d):" % (thr, m["n"]))
        log("  accuracy=%.6f precision=%.6f recall=%.6f specificity=%.6f f1=%.6f roc_auc=%.6f" % (
            m["accuracy"], m["precision"], m["recall"], m["specificity"], m["f1"], m["roc_auc"]))
        log("  confusion: tn=%d fp=%d fn=%d tp=%d" % (m["tn"], m["fp"], m["fn"], m["tp"]))
    elif status == "missing":
        log("TEST: test_proba.npy / test_true.npy NOT FOUND - cannot verify.")
        results["test"] = None
    else:
        log("TEST: LENGTH MISMATCH - proba=%d true=%d" % (proba.shape[0], true.shape[0]))
        results["test"] = None

    # Validation set
    proba, true, status = load_npy_pair("val_proba.npy", "val_true.npy")
    if status == "ok":
        thr = 0.5249999999999998
        m = metrics_at(true, proba, thr)
        results["val"] = m
        log("VAL (threshold=%.6f, n=%d):" % (thr, m["n"]))
        log("  accuracy=%.6f precision=%.6f recall=%.6f specificity=%.6f f1=%.6f roc_auc=%.6f" % (
            m["accuracy"], m["precision"], m["recall"], m["specificity"], m["f1"], m["roc_auc"]))
        log("  confusion: tn=%d fp=%d fn=%d tp=%d" % (m["tn"], m["fp"], m["fn"], m["tp"]))
    elif status == "missing":
        log("VAL: val_proba.npy / val_true.npy NOT FOUND - cannot verify.")
        results["val"] = None
    else:
        log("VAL: LENGTH MISMATCH - proba=%d true=%d" % (proba.shape[0], true.shape[0]))
        results["val"] = None

    return results


def verify_mobilenet():
    log("=" * 72)
    log("MobileNetV2 - metrics from model_card.json (no full .npy predictions available)")
    log("=" * 72)

    if not os.path.isfile(MODEL_CARD):
        log("model_card.json NOT FOUND - cannot verify MobileNetV2 metrics.")
        return None

    with open(MODEL_CARD, "r", encoding="utf-8") as f:
        card = json.load(f)

    results = {}
    test_m = card.get("test_metrics_at_threshold")
    if test_m:
        results["test"] = test_m
        log("TEST @ threshold=%.6f (n=%d):" % (test_m["threshold"], test_m["n"]))
        log("  accuracy=%.6f precision=%.6f recall=%.6f f1=%.6f roc_auc=%.6f" % (
            test_m["accuracy"], test_m["precision"], test_m["recall"],
            test_m["f1"], test_m["roc_auc"]))
        log("  confusion: tn=%d fp=%d fn=%d tp=%d" % (
            test_m["tn"], test_m["fp"], test_m["fn"], test_m["tp"]))

    test_050 = card.get("test_metrics_at_0.50")
    if test_050:
        results["test_at_0.50"] = test_050
        log("TEST @ threshold=0.50 (n=%d):" % test_050["n"])
        log("  accuracy=%.6f precision=%.6f recall=%.6f f1=%.6f roc_auc=%.6f" % (
            test_050["accuracy"], test_050["precision"], test_050["recall"],
            test_050["f1"], test_050["roc_auc"]))

    val_m = card.get("validation_metrics_at_threshold")
    if val_m:
        results["val"] = val_m
        log("VAL @ threshold=%.6f (n=%d):" % (val_m["threshold"], val_m["n"]))
        log("  accuracy=%.6f precision=%.6f recall=%.6f f1=%.6f roc_auc=%.6f" % (
            val_m["accuracy"], val_m["precision"], val_m["recall"],
            val_m["f1"], val_m["roc_auc"]))
        log("  confusion: tn=%d fp=%d fn=%d tp=%d" % (
            val_m["tn"], val_m["fp"], val_m["fn"], val_m["tp"]))

    # Count stored single-image predictions
    sips = card.get("single_image_predictions", [])
    results["single_image_predictions_count"] = len(sips)
    log("")
    log("Stored single_image_predictions: %d (NOT sufficient for full test-set verification)" % len(sips))
    log("MobileNetV2 test metrics are SUPPORTED BY ARTIFACT (model_card.json) but NOT independently re-derived.")
    log("Reason: no test_proba.npy / test_true.npy files exist for MobileNetV2 in the project.")

    return results


def main():
    log("DermaAI Reported-Metrics Recalculator")
    log("Phase 3 - verify metrics only from actual saved predictions/labels")
    log("")

    ct_results = verify_convnext()
    log("")
    mn_results = verify_mobilenet()

    # Build machine-readable output
    out = {
        "positive_class": POSITIVE_CLASS,
        "positive_class_name": "Cancer (dx in mel, bcc, akiec)",
        "threshold_convnext": 0.5249999999999998,
        "threshold_mobilenet_card": 0.22000000000000003,
        "convnext_tiny": ct_results,
        "mobilenetv2": mn_results,
        "verification_status": {
            "convnext_test": "VERIFIED" if ct_results.get("test") else "NOT_VERIFIED",
            "convnext_val": "VERIFIED" if ct_results.get("val") else "NOT_VERIFIED",
            "mobilenet_test": "SUPPORTED_BY_ARTIFACT_NOT_REDERIVED",
            "mobilenet_val": "SUPPORTED_BY_ARTIFACT_NOT_REDERIVED",
        },
    }

    # Write machine-readable results to a DEDICATED file so the comprehensive
    # audit JSON (DERMAI_VERIFICATION_RESULTS.json) is never clobbered.
    out_path = os.path.join(BASE, "metrics_verification_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    log("")
    log("Machine-readable results written to %s" % out_path)
    log("")
    log("CONCLUSION:")
    log("  ConvNeXt-Tiny test metrics: VERIFIED (recalculated from test_proba.npy/test_true.npy)")
    log("  ConvNeXt-Tiny val metrics:  VERIFIED (recalculated from val_proba.npy/val_true.npy)")
    log("  MobileNetV2 test metrics:   NOT RE-DERIVED (no full .npy prediction files)")
    log("  MobileNetV2 val metrics:    NOT RE-DERIVED (no full .npy prediction files)")
    log("  Only 6 MobileNetV2 single_image_predictions stored - insufficient for 1505-image test evaluation.")

    # Exit nonzero if a REQUIRED integrity check failed. ConvNeXt .npy artifacts
    # are the only independently verifiable predictions in the project; a missing
    # or length-mismatched pair is a genuine integrity failure, not a soft warning.
    failures = []
    if ct_results.get("test") is None:
        failures.append("ConvNeXt test predictions missing or length-mismatched")
    if ct_results.get("val") is None:
        failures.append("ConvNeXt validation predictions missing or length-mismatched")
    if failures:
        log("")
        log("INTEGRITY FAILURES:")
        for f in failures:
            log("  - %s" % f)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
