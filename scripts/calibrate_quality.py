#!/usr/bin/env python3
"""
DermaAI image-quality calibration procedure.

Measures the Laplacian-variance blur score (quality.py / config.QUALITY_BLUR_THRESHOLD)
on real skin photographs and reports the classification obtained at different
candidate thresholds.

Usage:
    python scripts/calibrate_quality.py

This script is READ-ONLY: it does not modify any production code or config.
It prints a reproducible report of blur scores and threshold classifications.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: F401  (used transitively by quality)
from PIL import Image, ImageFilter

import config
import quality

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Real images under manual review.
# Each entry: (path, manually assigned blur label, reviewer notes).
#
# Labels:
#   "sharp"        - genuine, in-focus skin photograph (no intentional blur)
#   "genuinely_blurry" - out of focus / motion blurred skin photograph
#
# NOTE: The four regression test-images are external clinical photos whose
#       blur status has NOT been manually validated. They are included for
#       reference only and are NOT used to justify a threshold.
# ---------------------------------------------------------------------------
REAL_IMAGES = [
    # Sharp (manually reviewed) - placeholder: add real sharp skin photos here.
    # Example format:
    # ("test-images/cancer-melanoma.jpg", "sharp", "Reviewed: lesion edges crisp"),
    #
    # Genuinely blurry (manually reviewed) - placeholder.
    # Example format:
    # ("test-images/noncancer-nevus.jpg", "genuinely_blurry", "Reviewed: severe motion blur"),
]

# Reference images included for diagnostic context only (NOT blur-validated).
REFERENCE_IMAGES = [
    ("test-images/cancer-melanoma.jpg", "melanoma (external, blur NOT validated)"),
    ("test-images/noncancer-nevus.jpg", "nevus (external, blur NOT validated)"),
    ("test-images/noncancer-congenital-nevus.jpg", "congenital nevus (external, blur NOT validated)"),
    ("test-images/noncancer-seborrheic-keratosis.jpg", "seborrheic keratosis (external, blur NOT validated)"),
]


def blur_score(img):
    return quality.check_image_quality(img)["metrics"]["blur_score"]


def classify(score, threshold):
    return "BLUR warning" if score < threshold else "OK"


def main():
    print("=" * 78)
    print("DermaAI image-quality blur calibration report")
    print("Analysis size: %s  |  Current QUALITY_BLUR_THRESHOLD: %.1f" % (
        config.QUALITY_ANALYSIS_SIZE, config.QUALITY_BLUR_THRESHOLD))
    print("=" * 78)

    print("\n--- Reference images (blur status NOT validated - diagnostic only) ---")
    ref_rows = []
    for path, label in REFERENCE_IMAGES:
        full = os.path.join(BASE, path)
        if not os.path.exists(full):
            print("  MISSING: %s" % path)
            continue
        img = Image.open(full)
        s = blur_score(img)
        ref_rows.append((label, s))
        print("%-55s blur_score=%8.2f  -> %s (at current threshold %.1f)" % (
            label, s, classify(s, config.QUALITY_BLUR_THRESHOLD), config.QUALITY_BLUR_THRESHOLD))

    print("\n--- Manually reviewed real images ---")
    if not REAL_IMAGES:
        print("  (none - no manually reviewed sharp/blurry examples available yet)")
    for path, label, notes in REAL_IMAGES:
        full = os.path.join(BASE, path)
        if not os.path.exists(full):
            print("  MISSING: %s" % path)
            continue
        img = Image.open(full)
        s = blur_score(img)
        print("%-45s label=%-18s blur_score=%8.2f  -> %s (at %.1f)  [%s]" % (
            path, label, s, classify(s, config.QUALITY_BLUR_THRESHOLD),
            config.QUALITY_BLUR_THRESHOLD, notes))

    print("\n--- Classification at candidate thresholds (reference images only) ---")
    candidate_thresholds = [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0]
    header = "image" + "".join("%12s" % (t if t == 100.0 else "t=%.0f" % t) for t in candidate_thresholds)
    print(header)
    print("-" * len(header))
    for label, s in ref_rows:
        row = "%-45s" % label
        for t in candidate_thresholds:
            row += "%12s" % ("WARN" if s < t else "OK")
        print(row)

    print("\n--- Recommendation ---")
    sharp_scores = []
    blurry_scores = []
    for path, label, notes in REAL_IMAGES:
        full = os.path.join(BASE, path)
        if not os.path.exists(full):
            continue
        s = blur_score(Image.open(full))
        if label == "sharp":
            sharp_scores.append(s)
        elif label == "genuinely_blurry":
            blurry_scores.append(s)

    if sharp_scores and blurry_scores:
        gap = min(sharp_scores) - max(blurry_scores)
        recommended = (min(sharp_scores) + max(blurry_scores)) / 2.0
        print("  Sharp scores:     %s" % ["%.2f" % s for s in sharp_scores])
        print("  Blurry scores:    %s" % ["%.2f" % s for s in blurry_scores])
        print("  Margin (gap):     %.2f" % gap)
        print("  Recommended threshold: %.2f (midpoint of gap)" % recommended)
        print("  NOTE: A single midpoint is only valid if examples span the decision")
        print("        boundary and are representative of real upload conditions.")
    elif REAL_IMAGES:
        print("  All manually reviewed images share one label; no threshold can be")
        print("  justified from these examples alone.")
    else:
        print("  No manually reviewed real images available.")
        print("  QUALITY_BLUR_THRESHOLD = 100.0 remains UNCALIBRATED against real skin")
        print("  photos. It was set heuristically and may flag smooth legitimate")
        print("  photographs as blurry (see diagnosis).")
        print()
        print("  Additional examples needed:")
        print("    - At least 5-10 genuinely sharp skin photographs (varied skin tones,")
        print("      body sites, lighting, camera devices) manually labelled 'sharp'.")
        print("    - At least 5-10 genuinely blurry skin photographs (out-of-focus,")
        print("      motion-blurred, low-shutter-speed) manually labelled 'genuinely_blurry'.")
        print("    - Labels must come from a human reviewer, NOT from cancer diagnosis.")
        print("    - Score each candidate image with this script before choosing a threshold.")
        print("  Do not choose a threshold based on cancer labels: sharpness and cancer")
        print("  classification are independent properties.")

    print("\n" + "=" * 78)
    print("Calibration complete. No production files were modified.")
    print("=" * 78)


if __name__ == "__main__":
    main()
