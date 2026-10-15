"""Dataset discovery, the frozen 70/15/15 lesion-level split, and the image cache."""

import glob
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import (CACHE_MASTER, CACHE_ROOT, CANCER_CLASSES, DATASET_CANDIDATES,
                     MASTER_SIZE, REFERENCE_SPLIT, SEED)


# --------------------------------------------------------------------------- metadata
def find_dataset_root(log):
    for root in DATASET_CANDIDATES:
        if not os.path.isdir(root):
            continue
        meta = glob.glob(os.path.join(root, "**", "HAM10000_metadata.csv"), recursive=True)
        jpgs = glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)
        if meta and len(jpgs) > 1000:
            import pandas as pd
            log("  dataset root : %s" % root)
            log("  metadata csv : %s" % meta[0])
            log("  jpgs         : %d" % len(jpgs))
            return root, meta[0]
    raise SystemExit("FATAL: no HAM10000 copy found. Looked in:\n  "
                     + "\n  ".join(DATASET_CANDIDATES))


def load_metadata(dataset_root, metadata_csv, log):
    """dx in the metadata CSV is the ONLY label source; filenames are never labels."""
    import pandas as pd

    index = {}
    for p in glob.glob(os.path.join(dataset_root, "**", "*.jpg"), recursive=True):
        index[os.path.splitext(os.path.basename(p))[0]] = p

    meta = pd.read_csv(metadata_csv)
    meta["image_id"] = meta["image_id"].astype(str).str.strip()
    meta["path"] = meta["image_id"].map(index)
    missing = int(meta["path"].isna().sum())
    meta = meta[meta["path"].notna()].reset_index(drop=True)
    if missing:
        log("  WARNING: %d metadata rows had no jpg and were dropped" % missing)
    if len(meta) < 5000:
        raise SystemExit("FATAL: only %d images joined - refusing to continue." % len(meta))
    meta["label"] = meta["dx"].apply(lambda x: 1 if x in CANCER_CLASSES else 0)
    meta["label_name"] = meta["label"].map({0: "Non-Cancer", 1: "Cancer"})
    meta["group"] = meta["lesion_id"].astype(str)
    conflict = int((meta.groupby("group")["label"].nunique() > 1).sum())
    if conflict:
        raise SystemExit("FATAL: %d lesions carry both labels - label rule is wrong." % conflict)
    meta.attrs["root"] = dataset_root
    log("  images %d | lesions %d | dx %s"
        % (len(meta), meta["group"].nunique(), meta["dx"].value_counts().to_dict()))
    return meta


def make_split(meta, log):
    """Lesion-level stratified 70/15/15 with seed 42 - byte-for-byte the same logic as
    train_convnext_tiny.make_split so the frozen TEST set stays identical."""
    from sklearn.model_selection import train_test_split

    grp = meta.groupby("group")["label"].max().reset_index()
    g_train, g_temp = train_test_split(grp, test_size=0.30, stratify=grp["label"],
                                       random_state=SEED)
    g_val, g_test = train_test_split(g_temp, test_size=0.50, stratify=g_temp["label"],
                                     random_state=SEED)

    def take(keys):
        return (meta[meta["group"].isin(set(keys["group"]))]
                .sample(frac=1.0, random_state=SEED).reset_index(drop=True))

    splits = {n: take(k) for n, k in (("train", g_train), ("val", g_val), ("test", g_test))}

    gsets = {n: set(s["group"]) for n, s in splits.items()}
    overlap = sum(len(gsets[a] & gsets[b])
                  for i, a in enumerate(("train", "val", "test"))
                  for b in ("train", "val", "test")[i + 1:])
    log("")
    log("  %-6s %8s %9s  %s" % ("split", "images", "lesions", "class counts"))
    for n, s in splits.items():
        log("  %-6s %8d %9d  %s" % (n, len(s), s["group"].nunique(),
                                    s["label_name"].value_counts().to_dict()))
    if overlap:
        raise SystemExit("FATAL: lesion leakage between splits (%d) - aborting." % overlap)
    log("  LEAKAGE CHECK : PASS (0 shared lesions)")
    exact = {n: (len(splits[n]) == REFERENCE_SPLIT[n][0]
                 and int(splits[n]["group"].nunique()) == REFERENCE_SPLIT[n][1]
                 and splits[n]["label_name"].value_counts().to_dict()
                 == REFERENCE_SPLIT[n][2]) for n in splits}
    log("  MATCHES PROD SPLIT (model_card.json) : %s"
        % ("YES" if all(exact.values()) else "no -> %s" % exact))
    return splits, exact


def attach_rows(meta, splits):
    id_to_row = {iid: i for i, iid in enumerate(meta["image_id"].to_numpy())}
    for s in splits.values():
        s["_row"] = s["image_id"].map(id_to_row)
        assert s["_row"].notna().all()
    return splits


# --------------------------------------------------------------------------- cache
def build_master_cache(meta, log, size=MASTER_SIZE):
    """Decode each image once into a uint8 (N, size, size, 3) memmap.

    Cached at the largest resolution any experiment uses; smaller inputs are produced on
    GPU with an area/bicubic downsample, so one decode pass serves every resolution.
    Resize is deterministic (BICUBIC, no crop) -> no information about labels is used.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    n = len(meta)
    path = CACHE_MASTER if size == MASTER_SIZE else CACHE_MASTER.replace(
        "master_384x384", "master_%dx%d" % (size, size))
    if os.path.exists(path):
        try:
            arr = np.load(path, mmap_mode="r")
            if arr.shape == (n, size, size, 3) and arr.dtype == np.uint8:
                log("  master cache : reuse %s (%d x %dx%dx%d u8, %.2f GB)"
                    % (path, n, size, size, 3, arr.nbytes / 1024 ** 3))
                return arr
            log("  master cache : %s has shape %s -> rebuild" % (path, arr.shape))
        except Exception as exc:
            log("  master cache : %s unreadable (%s) -> rebuild" % (path, exc))

    os.makedirs(CACHE_ROOT, exist_ok=True)
    log("  master cache : building %s (%d imgs, %.2f GB) ..."
        % (path, n, n * size * size * 3 / 1024 ** 3))
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8,
                                    shape=(n, size, size, 3))
    t0 = time.time()
    paths = meta["path"].to_numpy()
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            out[i] = np.asarray(im.convert("RGB").resize((size, size), Image.BICUBIC),
                                dtype=np.uint8)
        if (i + 1) % 2500 == 0:
            log("      %5d / %d  (%.0f img/s)"
                % (i + 1, n, (i + 1) / max(time.time() - t0, 1e-9)))
    out.flush()
    del out
    log("      built in %.1f s" % (time.time() - t0))
    return np.load(path, mmap_mode="r")


# --------------------------------------------------------------------------- dataset
class CachedHAM(Dataset):
    """Serves the master cache resized on GPU later; returns uint8 CHW + label + row idx."""

    def __init__(self, arr, df, train=False):
        self.arr = arr
        self.rows = np.ascontiguousarray(df["_row"].to_numpy())
        self.labels = df["label"].to_numpy().astype(np.float32)
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = np.array(self.arr[self.rows[i]], dtype=np.uint8)      # (H, W, 3)
        return torch.from_numpy(img).permute(2, 0, 1), \
            torch.tensor(self.labels[i]), i


DX_CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
DX_INDEX = {d: i for i, d in enumerate(DX_CLASSES)}


def add_dx_index(df):
    df = df.copy()
    df["y_dx"] = df["dx"].map(DX_INDEX).astype(np.int64)
    return df


class CachedHAMAux(CachedHAM):
    """Same as CachedHAM but also returns the 7-way dx index for the auxiliary head."""

    def __init__(self, arr, df, train=False):
        super().__init__(arr, df, train)
        if "y_dx" not in df.columns:
            df = add_dx_index(df)
        self.dx = df["y_dx"].to_numpy().astype(np.int64)

    def __getitem__(self, i):
        img = np.array(self.arr[self.rows[i]], dtype=np.uint8)
        return (torch.from_numpy(img).permute(2, 0, 1),
                torch.tensor(self.labels[i]), i, torch.tensor(int(self.dx[i])))


def fold_frames(meta, splits, n_folds=5, seed=SEED):
    """StratifiedGroupKFold over lesion_id over the TRAIN+VAL pool (test never touched)."""
    from sklearn.model_selection import StratifiedGroupKFold

    pool = splits["train"].copy()
    val = splits["val"].copy()
    tag = np.array(["train"] * len(pool) + ["val"] * len(val))
    pool = __import__("pandas").concat([pool, val], ignore_index=True)
    y = pool["label"].to_numpy()
    g = pool["group"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [[] for _ in range(n_folds)]
    for k, (tr, va) in enumerate(cv.split(np.zeros(len(y)), y, g)):
        folds[k] = (pool.iloc[tr].reset_index(drop=True), pool.iloc[va].reset_index(drop=True))
    return folds, pool, tag