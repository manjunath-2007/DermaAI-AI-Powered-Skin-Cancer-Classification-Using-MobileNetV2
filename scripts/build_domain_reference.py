#!/usr/bin/env python3
"""Build the input-domain reference matrix for DermaAI.

The gate in domain.py measures how far an uploaded image sits from the HAM10000
training manifold in the deployed ConvNeXt's own 768-d penultimate feature
space. This script produces that reference matrix and the calibration record
that documents where the thresholds came from.

Only the HAM10000 TRAIN split is used for the reference, so the validation and
test splits remain a clean held-out check of the gate's false-reject rate.

Usage
-----
    python scripts/build_domain_reference.py
    python scripts/build_domain_reference.py --calibration-dir <dir> --calibration-label legit
    python scripts/build_domain_reference.py --calibration-dir <dir> --calibration-label ordinary

The --calibration-dir options score an extra folder of images and fold the
result into the calibration record. "legit" folders must contain genuine
lesion photographs, "ordinary" folders ordinary photographs.

Requirements: the local HAM10000 copy and the 224x224 uint8 cache
(D:\\dermai_cache by default, override with the DERMAI_CACHE env var).
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "scripts"))

import numpy as np
from PIL import Image

import config
import domain
import inference
from verify_lesion_split import build_split

DEFAULT_METADATA = r"D:\User\Downloads\archive\HAM10000_metadata.csv"
DEFAULT_IMAGES = r"D:\User\Downloads\archive"
DEFAULT_CACHE = os.path.join(os.environ.get("DERMAI_CACHE", r"D:\dermai_cache"),
                             "ham10000_224x224_u8.npy")


def log(msg):
    print(msg, flush=True)


def embed_rows(cache, rows, batch=32):
    """Penultimate embeddings for the given cache rows (deployed model)."""
    out = []
    for start in range(0, len(rows), batch):
        chunk = np.ascontiguousarray(cache[rows[start:start + batch]])
        probs, metas, embs = inference.predict_batch_with_embeddings(
            [_as_pil(c) for c in chunk])
        out.append(embs)
    return np.concatenate(out)


def _as_pil(arr):
    return Image.fromarray(arr)


def score_array(arr, batch=32):
    """Max-over-flip domain score for a stack of 224x224x3 uint8 images."""
    norm = domain._reference_norm
    scores = []
    for start in range(0, len(arr), batch):
        chunk = np.ascontiguousarray(arr[start:start + batch])
        _, _, embs = inference.predict_batch_with_embeddings([_as_pil(c) for c in chunk])
        flip = np.ascontiguousarray(chunk[:, :, ::-1])
        _, _, embs_flip = inference.predict_batch_with_embeddings([_as_pil(c) for c in flip])
        d = domain._knn_cosine_distance(embs.astype(np.float32), norm, config.DOMAIN_K)
        d_flip = domain._knn_cosine_distance(embs_flip.astype(np.float32), norm, config.DOMAIN_K)
        scores.append(np.maximum(d, d_flip))
    return np.concatenate(scores)


def score_images(paths):
    """Domain score for a list of image paths (max over original + flip)."""
    images = []
    for p in paths:
        im = Image.open(p)
        im.load()
        images.append(im)
    return domain.domain_scores(images)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=DEFAULT_METADATA)
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--calibration-dir", default=None)
    ap.add_argument("--calibration-label", default="ordinary", choices=["legit", "ordinary"])
    args = ap.parse_args()

    log("=" * 74)
    log("DermaAI - input-domain reference builder")
    log("=" * 74)
    log("  metadata : %s" % args.metadata)
    log("  images   : %s" % args.images)
    log("  cache    : %s" % args.cache)
    log("  output   : %s" % config.DOMAIN_REFERENCE_PATH)

    if not os.path.isfile(args.metadata):
        log("FATAL: HAM10000 metadata not found: %s" % args.metadata)
        sys.exit(2)
    if not os.path.isfile(args.cache):
        log("FATAL: 224x224 cache not found: %s" % args.cache)
        log("       Rebuild it with scripts/training/train_convnext_tiny.py or")
        log("       the Colab notebook that produced the cache.")
        sys.exit(2)

    log("")
    log("Rebuilding the lesion-level split (seed 42) ...")
    meta, splits = build_split(args.metadata, args.images)
    id_to_row = {iid: i for i, iid in enumerate(meta["image_id"].to_numpy())}
    rows = {n: splits[n]["image_id"].map(id_to_row).to_numpy() for n in ("train", "val", "test")}
    for n in ("train", "val", "test"):
        log("  %-6s images=%5d lesions=%5d" % (n, len(rows[n]), splits[n]["group"].nunique()))

    cache = np.load(args.cache, mmap_mode="r")
    if cache.shape[0] != len(meta):
        log("FATAL: cache has %d rows but the metadata yields %d images" % (cache.shape[0], len(meta)))
        sys.exit(2)

    log("")
    log("Loading the deployed checkpoint ...")
    inference.load_convnext_model()

    log("Embedding the TRAIN split ...")
    emb_train = embed_rows(cache, rows["train"])
    emb_train = emb_train.astype(np.float16)

    os.makedirs(os.path.dirname(config.DOMAIN_REFERENCE_PATH), exist_ok=True)
    np.save(config.DOMAIN_REFERENCE_PATH, emb_train)
    log("  wrote %s  shape=%s  size=%.1f MB" % (
        config.DOMAIN_REFERENCE_PATH, tuple(emb_train.shape),
        os.path.getsize(config.DOMAIN_REFERENCE_PATH) / 1e6))

    log("")
    log("Scoring the held-out VAL / TEST splits (gate false-reject check) ...")
    domain._reference = None
    domain._reference_norm = None
    domain._load_reference()

    record = {
        "generated_by": "scripts/build_domain_reference.py",
        "dataset": "HAM10000",
        "reference_split": "train",
        "reference_images": int(emb_train.shape[0]),
        "embedding_dim": int(emb_train.shape[1]),
        "dtype": "float16",
        "checkpoint": os.path.basename(config.CONVNEXT_CKPT),
        "thresholds": {
            "k": config.DOMAIN_K,
            "max_distance": config.DOMAIN_MAX_DISTANCE,
            "suspicious_distance": config.DOMAIN_SUSPICIOUS_DISTANCE,
            "second_opinion_threshold": config.MOBILENET_THRESHOLD,
        },
    }

    for split in ("val", "test"):
        score = score_array(np.ascontiguousarray(cache[rows[split]]))
        record["ham10000_%s" % split] = {
            "images": int(len(score)),
            "max_score": float(score.max()),
            "p99_score": float(np.percentile(score, 99)),
            "mean_score": float(score.mean()),
            "false_reject_rate_at_max_distance": float((score > config.DOMAIN_MAX_DISTANCE).mean()),
            "false_reject_rate_at_suspicious_distance": float((score > config.DOMAIN_SUSPICIOUS_DISTANCE).mean()),
        }
        log("  %-5s n=%5d max=%.4f p99=%.4f mean=%.4f  FRR@max=%.3f%%  FRR@susp=%.3f%%" % (
            split, len(score), score.max(), np.percentile(score, 99), score.mean(),
            100 * (score > config.DOMAIN_MAX_DISTANCE).mean(),
            100 * (score > config.DOMAIN_SUSPICIOUS_DISTANCE).mean()))

    if args.calibration_dir:
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            import glob
            paths.extend(glob.glob(os.path.join(args.calibration_dir, ext)))
        paths = sorted(set(paths))
        if not paths:
            log("WARNING: no images found in %s" % args.calibration_dir)
        else:
            scores = score_images(paths)
            key = "calibration_%s" % args.calibration_label
            record[key] = {
                "directory": args.calibration_dir,
                "images": len(scores),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
                "mean_score": float(np.mean(scores)),
                "rejected_rate_at_max_distance": float(
                    np.mean(np.array(scores) > config.DOMAIN_MAX_DISTANCE)),
            }
            log("")
            log("  %s: n=%d min=%.4f max=%.4f mean=%.4f rejected@max=%.1f%%" % (
                key, len(scores), np.min(scores), np.max(scores), np.mean(scores),
                100 * record[key]["rejected_rate_at_max_distance"]))

    meta_path = os.path.join(os.path.dirname(config.DOMAIN_REFERENCE_PATH),
                             "domain_reference_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    log("")
    log("Calibration record written to %s" % meta_path)
    log("Done.")


if __name__ == "__main__":
    main()
