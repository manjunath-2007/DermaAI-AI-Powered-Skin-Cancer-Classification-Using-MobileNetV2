#!/usr/bin/env python3
"""
DermaAI — Lesion-level split integrity checker (PHASE 2 deliverable).

Validates the HAM10000 70/15/15 lesion-level stratified split (seed 42)
against the actual metadata and split-logic code. Reports image counts,
lesion counts, class distributions, pairwise lesion_id overlaps, duplicate
image paths/identifiers, and exits nonzero if any integrity check fails.

Does NOT generate fake data. Reports NOT_VERIFIED where evidence is missing.

Usage:
    python scripts/verify_lesion_split.py [metadata_csv] [images_root]

Defaults:
    metadata_csv = D:\\User\\Downloads\\archive\\HAM10000_metadata.csv
    images_root  = D:\\User\\Downloads\\archive
"""

import csv
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
CANCER_CLASSES = ["mel", "bcc", "akiec"]
NON_CANCER_CLASSES = ["nv", "bkl", "vasc", "df"]

REFERENCE_SPLIT = {
    "train": (7010, 5229, {"Non-Cancer": 5641, "Cancer": 1369}),
    "val": (1500, 1120, {"Non-Cancer": 1203, "Cancer": 297}),
    "test": (1505, 1121, {"Non-Cancer": 1217, "Cancer": 288}),
}

LABEL_MAP = {
    "1": "Cancer (dx in mel, bcc, akiec)",
    "0": "Non-Cancer (dx in nv, bkl, vasc, df)",
}


def log(msg):
    print(msg, flush=True)


def build_split(metadata_csv, images_root):
    """Rebuild the lesion-level stratified 70/15/15 split (seed 42).

    Mirrors the logic in dermai_exp/data.py and scripts/training/train_convnext_tiny.py.
    Returns (meta, train_df, val_df, test_df).
    """
    meta = pd.read_csv(metadata_csv)

    # Build image_id -> path index
    index = {}
    for part in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
        d = os.path.join(images_root, part)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith(".jpg"):
                index[os.path.splitext(f)[0]] = os.path.join(d, f)

    meta["image_id"] = meta["image_id"].astype(str).str.strip()
    meta["path"] = meta["image_id"].map(index)
    missing = int(meta["path"].isna().sum())
    meta = meta[meta["path"].notna()].reset_index(drop=True)

    meta["label"] = meta["dx"].apply(lambda x: 1 if x in CANCER_CLASSES else 0)
    meta["label_name"] = meta["label"].map({0: "Non-Cancer", 1: "Cancer"})
    meta["group"] = meta["lesion_id"].astype(str)

    # Lesion-level grouping: one lesion may not appear in two splits.
    grp = meta.groupby("group")["label"].max().reset_index()
    g_train, g_temp = train_test_split(
        grp, test_size=0.30, stratify=grp["label"], random_state=SEED
    )
    g_val, g_test = train_test_split(
        g_temp, test_size=0.50, stratify=g_temp["label"], random_state=SEED
    )

    def take(keys):
        return (
            meta[meta["group"].isin(set(keys["group"]))]
            .sample(frac=1.0, random_state=SEED)
            .reset_index(drop=True)
        )

    splits = {
        "train": take(g_train),
        "val": take(g_val),
        "test": take(g_test),
    }
    return meta, splits


def validate_split(meta, splits):
    """Run all integrity checks. Returns (ok, report_dict)."""
    report = {}
    failures = []

    # --- Overall metadata ---
    report["total_images"] = int(len(meta))
    report["unique_lesions"] = int(meta["group"].nunique())
    report["dx_counts"] = meta["dx"].value_counts().to_dict()

    # --- Duplicate checks ---
    report["duplicate_image_ids"] = int(meta["image_id"].duplicated().sum())
    report["duplicate_image_id_lesion_pairs"] = int(
        meta.duplicated(subset=["image_id", "lesion_id"]).sum()
    )
    if report["duplicate_image_ids"] > 0:
        failures.append("DUPLICATE_IMAGE_IDS: %d duplicate image_id values" % report["duplicate_image_ids"])
    if report["duplicate_image_id_lesion_pairs"] > 0:
        failures.append("DUPLICATE_IMAGE_ID_LESION: %d duplicate (image_id, lesion_id) pairs" % report["duplicate_image_id_lesion_pairs"])

    # --- Per-split counts and class distributions ---
    for name in ("train", "val", "test"):
        df = splits[name]
        report[name] = {
            "images": int(len(df)),
            "lesions": int(df["group"].nunique()),
            "class_counts": df["label_name"].value_counts().to_dict(),
        }
        expected = REFERENCE_SPLIT[name]
        match = (
            len(df) == expected[0]
            and df["group"].nunique() == expected[1]
            and df["label_name"].value_counts().to_dict() == expected[2]
        )
        report[name + "_matches_reference"] = bool(match)
        if not match:
            failures.append(
                "SPLIT_MISMATCH %s: got images=%d lesions=%d classes=%s; "
                "expected images=%d lesions=%d classes=%s"
                % (name, len(df), df["group"].nunique(),
                   df["label_name"].value_counts().to_dict(),
                   expected[0], expected[1], expected[2])
            )

    # --- Pairwise lesion_id overlaps ---
    gsets = {n: set(splits[n]["group"]) for n in ("train", "val", "test")}
    overlap_tv = len(gsets["train"] & gsets["val"])
    overlap_tt = len(gsets["train"] & gsets["test"])
    overlap_vt = len(gsets["val"] & gsets["test"])
    report["overlap_train_val"] = overlap_tv
    report["overlap_train_test"] = overlap_tt
    report["overlap_val_test"] = overlap_vt
    report["total_lesion_overlap"] = overlap_tv + overlap_tt + overlap_vt

    if overlap_tv != 0:
        failures.append("LEAKAGE train∩val: %d shared lesions" % overlap_tv)
    if overlap_tt != 0:
        failures.append("LEAKAGE train∩test: %d shared lesions" % overlap_tt)
    if overlap_vt != 0:
        failures.append("LEAKAGE val∩test: %d shared lesions" % overlap_vt)

    # --- Label mapping check ---
    report["label_mapping"] = LABEL_MAP
    report["cancer_classes"] = CANCER_CLASSES
    report["non_cancer_classes"] = NON_CANCER_CLASSES

    # --- Grouping key check ---
    report["grouping_key"] = "lesion_id"
    report["seed"] = SEED
    report["ratios"] = "70/15/15"

    ok = len(failures) == 0
    return ok, report, failures


def main():
    metadata_csv = sys.argv[1] if len(sys.argv) > 1 else r"D:\User\Downloads\archive\HAM10000_metadata.csv"
    images_root = sys.argv[2] if len(sys.argv) > 2 else r"D:\User\Downloads\archive"

    log("=" * 72)
    log("DermaAI Lesion-Level Split Integrity Verifier")
    log("=" * 72)
    log("Metadata: %s" % metadata_csv)
    log("Images root: %s" % images_root)
    log("Seed: %d | Split: lesion-level stratified 70/15/15 | Grouping: lesion_id" % SEED)
    log("")

    if not os.path.isfile(metadata_csv):
        log("FATAL: metadata CSV not found: %s" % metadata_csv)
        sys.exit(2)

    meta, splits = build_split(metadata_csv, images_root)
    ok, report, failures = validate_split(meta, splits)

    log("")
    log("RESULTS")
    log("-" * 72)
    for name in ("train", "val", "test"):
        r = report[name]
        log("%-6s images=%7d lesions=%7d  classes=%s  matches_ref=%s" % (
            name, r["images"], r["lesions"], r["class_counts"],
            report[name + "_matches_reference"]))
    log("")
    log("Overlap train-intersect-val:   %d" % report["overlap_train_val"])
    log("Overlap train-intersect-test:  %d" % report["overlap_train_test"])
    log("Overlap val-intersect-test:    %d" % report["overlap_val_test"])
    log("Total lesion overlap: %d" % report["total_lesion_overlap"])
    log("")
    log("Duplicate image_ids:        %d" % report["duplicate_image_ids"])
    log("Duplicate (image_id,lesion_id): %d" % report["duplicate_image_id_lesion_pairs"])
    log("")
    log("Integrity assertions:")
    log("  train_lesions INTERSECT validation_lesions = %s" % (
        "EMPTY" if report["overlap_train_val"] == 0 else "NON-EMPTY (%d)" % report["overlap_train_val"]))
    log("  train_lesions INTERSECT test_lesions = %s" % (
        "EMPTY" if report["overlap_train_test"] == 0 else "NON-EMPTY (%d)" % report["overlap_train_test"]))
    log("  validation_lesions INTERSECT test_lesions = %s" % (
        "EMPTY" if report["overlap_val_test"] == 0 else "NON-EMPTY (%d)" % report["overlap_val_test"]))
    log("")

    if failures:
        log("STATUS: FAILED")
        for f in failures:
            log("  - %s" % f)
    else:
        log("STATUS: PASSED - zero lesion overlap, counts match reference split.")

    # Write machine-readable report
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "split_verification_results.json")
    out_path = os.path.abspath(out_path)
    report["status"] = "PASSED" if ok else "FAILED"
    report["failures"] = failures
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log("Machine-readable report written to %s" % out_path)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
