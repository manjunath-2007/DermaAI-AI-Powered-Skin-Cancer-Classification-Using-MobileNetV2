"""
DermaAI - ConvNeXt-Tiny training on HAM10000 (local GPU version of
dermai_convnext_tiny_colab.ipynb).

Runs on CUDA (RTX 2050). TensorFlow >= 2.11 has no GPU support on native
Windows, so the Colab notebook's Keras pipeline is ported to PyTorch here.

Contract (must match the notebook exactly):
  architecture   : ConvNeXt-Tiny (torchvision, IMAGENET1K_V1) + GAP + Dropout + Linear(1)
  input          : 224x224x3 RGB
  label 1        : Cancer     (dx in mel, bcc, akiec)
  label 0        : Non-Cancer (dx in nv, bkl, vasc, df)
  split          : lesion-level stratified 70/15/15, random_state=42
  labels         : HAM10000_metadata.csv 'dx' column only - never from filenames
  threshold      : chosen on VALIDATION only (argmax F1)
  test split     : scored exactly once, at the frozen threshold
  production     : model/skin_cancer_mobilenetv2.h5 is never read or written

Stages:
  stage 1  frozen backbone, head only
  stage 2  trailing ConvNeXt stages unfrozen, discriminative LR, label smoothing

Usage:  python train_convnext_tiny.py
"""

import argparse
import glob
import json
import math
import os
import random
import subprocess
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# --------------------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results", "convnext_tiny")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
CACHE_DIR = os.environ.get("DERMAI_CACHE", r"D:\dermai_cache")
CACHE_NPY = os.path.join(CACHE_DIR, "ham10000_224x224_u8.npy")

DATASET_CANDIDATES = [
    r"D:\User\Downloads\archive",
    r"D:\ham10000",
    os.path.join(HERE, "HAM10000"),
    r"C:\Users\Manjunath\Downloads\archive",
]

PRODUCTION_MODEL = os.path.join(HERE, "model", "skin_cancer_mobilenetv2.h5")

# --------------------------------------------------------------------------- config
SEED = 42
INPUT_SIZE = 224
CANCER_CLASSES = ["mel", "bcc", "akiec"]
CLASS_NAMES = ["Non-Cancer", "Cancer"]

STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 24
UNFREEZE_STAGES = 2
STAGE_ORDER = ["stage1", "stage2"]

LR_HEAD = 1e-3
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.02
DROPOUT = 0.30

EFFECTIVE_BATCH = 32       # what the notebook used (BATCH_SIZE = 32)
MAX_MICRO_BATCH = 16       # hard VRAM ceiling for the 4 GB RTX 2050
MIN_MICRO_BATCH = 4
PATIENCE = 6
KEEP_CKPTS = 2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# numbers produced by the verified production split (model/model_card.json)
REFERENCE_SPLIT = {
    "train": (7010, 5229, {"Non-Cancer": 5641, "Cancer": 1369}),
    "val": (1500, 1120, {"Non-Cancer": 1203, "Cancer": 297}),
    "test": (1505, 1121, {"Non-Cancer": 1217, "Cancer": 288}),
}

GRID = np.arange(0.05, 0.955, 0.005)


def log(msg):
    print(msg, flush=True)


def banner(title):
    log("")
    log("=" * 70)
    log(title)
    log("=" * 70)


# --------------------------------------------------------------------------- device
def require_cuda():
    """Hard-fail unless a real CUDA device is usable. CPU training is not allowed."""
    banner("GPU CHECK")
    if not torch.cuda.is_available():
        raise SystemExit(
            "FATAL: torch.cuda.is_available() is False. CPU training is disabled by "
            "design. Reinstall a CUDA build: pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu128"
        )
    props = torch.cuda.get_device_properties(0)
    log("  torch          : %s" % torch.__version__)
    log("  CUDA available : %s" % torch.version.cuda)
    log("  cudnn          : %s" % torch.backends.cudnn.version())
    log("  device 0       : %s" % props.name)
    log("  capability     : sm_%d%d" % (props.major, props.minor))
    log("  total VRAM     : %.2f GB" % (props.total_memory / 1024 ** 3))
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        log("  nvidia-smi     : %s" % (smi.stdout or smi.stderr).strip())
    except Exception as exc:                       # nvidia-smi is optional
        log("  nvidia-smi     : unavailable (%s)" % exc)

    # real compute smoke test - proves kernels actually run on the device
    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    c = (a @ b).float().mean().item()
    torch.cuda.synchronize()
    log("  fp16 matmul    : OK (mean=%.4f on cuda)" % c)
    return torch.device("cuda")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# --------------------------------------------------------------------------- dataset
def find_dataset_root():
    for root in DATASET_CANDIDATES:
        if not os.path.isdir(root):
            continue
        meta = glob.glob(os.path.join(root, "**", "HAM10000_metadata.csv"), recursive=True)
        jpgs = glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)
        if meta and len(jpgs) > 1000:
            log("  dataset root : %s" % root)
            log("  metadata csv : %s" % meta[0])
            log("  jpg files    : %d" % len(jpgs))
            return root, meta[0]
    raise SystemExit(
        "FATAL: no HAM10000 copy found. Looked in:\n  " + "\n  ".join(DATASET_CANDIDATES)
    )


def load_metadata(dataset_root, metadata_csv):
    """dx in the metadata CSV is the ONLY label source. Filenames are never labels."""
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
        log("  WARNING: %d metadata rows had no matching jpg and were dropped" % missing)
    if len(meta) < 5000:
        raise SystemExit("FATAL: metadata joined only %d images - refusing to continue."
                         % len(meta))
    meta["label"] = meta["dx"].apply(lambda x: 1 if x in CANCER_CLASSES else 0)
    meta["label_name"] = meta["label"].map({0: "Non-Cancer", 1: "Cancer"})
    meta["group"] = meta["lesion_id"].astype(str)      # lesion-level grouping key
    meta.attrs["root"] = dataset_root
    log("  images matched: %d | unique lesions: %d" % (len(meta), meta["group"].nunique()))
    log("  dx counts     : %s" % meta["dx"].value_counts().to_dict())
    return meta


def make_split(meta):
    """Lesion-level stratified 70/15/15, seed 42 - identical logic to the notebook."""
    from sklearn.model_selection import train_test_split

    grp = meta.groupby("group")["label"].max().reset_index()
    g_train, g_temp = train_test_split(grp, test_size=0.30, stratify=grp["label"],
                                       random_state=SEED)
    g_val, g_test = train_test_split(g_temp, test_size=0.50, stratify=g_temp["label"],
                                     random_state=SEED)

    def take(keys):
        return (meta[meta["group"].isin(set(keys["group"]))]
                .sample(frac=1.0, random_state=SEED).reset_index(drop=True))

    splits = {"train": take(g_train), "val": take(g_val), "test": take(g_test)}

    overlap = (len(set(splits["train"]["group"]) & set(splits["val"]["group"]))
               + len(set(splits["train"]["group"]) & set(splits["test"]["group"]))
               + len(set(splits["val"]["group"]) & set(splits["test"]["group"])))
    log("")
    log("  %-6s %8s %9s %8s  %s" % ("split", "images", "lesions", "overlap", "class counts"))
    for name, df in splits.items():
        log("  %-6s %8d %9d %8d  %s" % (name, len(df), df["group"].nunique(), overlap,
                                       df["label_name"].value_counts().to_dict()))
        log("  %-6s %8d %9d %8s  %s" % ("  ref", REFERENCE_SPLIT[name][0],
                                        REFERENCE_SPLIT[name][1], "-",
                                        REFERENCE_SPLIT[name][2]))
    if overlap != 0:
        raise SystemExit("FATAL: lesion leakage between splits - aborting.")
    exact = all(len(splits[n]) == REFERENCE_SPLIT[n][0]
                and splits[n]["group"].nunique() == REFERENCE_SPLIT[n][1]
                and splits[n]["label_name"].value_counts().to_dict() == REFERENCE_SPLIT[n][2]
                for n in splits)
    log("")
    log("  LEAKAGE CHECK      : PASS (0 shared lesions)")
    log("  MATCHES PROD SPLIT : %s"
        % ("YES - identical to model_card.json" if exact
           else "no - counts above are the actual truth for this run"))
    return splits


def build_cache(meta, cache_path=CACHE_NPY):
    """Decode + resize every image once into a uint8 cache.

    Removes the JPEG-decode bottleneck so the 4 GB GPU is not starved by the CPU.
    Deterministic transform (BICUBIC resize to 224x224) - no augmentation, no leakage.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    n = len(meta)
    if os.path.exists(cache_path):
        try:
            arr = np.load(cache_path, mmap_mode="r")
            if arr.shape == (n, INPUT_SIZE, INPUT_SIZE, 3) and arr.dtype == np.uint8:
                log("  image cache  : reusing %s (%d x %dx%dx%d uint8)"
                    % (cache_path, n, INPUT_SIZE, INPUT_SIZE, 3))
                return arr
            log("  image cache  : %s has shape %s, rebuilding" % (cache_path, arr.shape))
        except Exception as exc:
            log("  image cache  : %s unreadable (%s), rebuilding" % (cache_path, exc))

    os.makedirs(CACHE_DIR, exist_ok=True)
    log("  image cache  : building %s (%d images, ~%.2f GB) ..."
        % (cache_path, n, n * INPUT_SIZE * INPUT_SIZE * 3 / 1e9))
    out = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.uint8,
                                    shape=(n, INPUT_SIZE, INPUT_SIZE, 3))
    t0 = time.time()
    for i, p in enumerate(meta["path"].to_numpy()):
        with Image.open(p) as im:
            out[i] = np.asarray(im.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE),
                                                         Image.BICUBIC), dtype=np.uint8)
        if (i + 1) % 2000 == 0:
            log("      %5d / %d  (%.0f img/s)" % (i + 1, n, (i + 1) / (time.time() - t0)))
    out.flush()
    del out
    log("      built in %.1f s" % (time.time() - t0))
    return np.load(cache_path, mmap_mode="r")


class CachedHAM(Dataset):
    """Serves cached uint8 CHW tensors + label + the row position inside the split frame."""

    def __init__(self, arr, df):
        self.arr = arr
        self.rows = df["_row"].to_numpy()
        self.labels = df["label"].to_numpy().astype(np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        # np.array (not asarray) so the read-only memmap slice is copied into writable RAM
        img = np.array(self.arr[self.rows[i]], dtype=np.uint8)          # (H, W, 3)
        return torch.from_numpy(img).permute(2, 0, 1).contiguous(), \
            torch.tensor(self.labels[i]), i


# --------------------------------------------------------------------------- model
class DermaConvNeXt(nn.Module):
    """ConvNeXt-Tiny + GAP + Dropout + Linear(1).

    The ImageNet normalisation is a real layer inside the graph, so a saved .pt file is
    self-contained: feed raw RGB in [0, 255] and nothing else.

    torchvision stores a stage's downsample conv as a *separate* module from that
    stage's CNBlocks (features = stem, stage1, stage2.down, stage2, ...), so the
    stage boundaries are probed from the weight shapes instead of being hardcoded.
    """

    STAGE_OF_WIDTH = {96: 1, 192: 2, 384: 3, 768: 4}

    def __init__(self, dropout=DROPOUT, pretrained=True):
        super().__init__()
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        if pretrained:
            try:
                weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
                net = convnext_tiny(weights=weights)
                log("  backbone     : ConvNeXt-Tiny, ImageNet weights loaded (%s)" % weights)
            except Exception as exc:
                raise SystemExit(
                    "FATAL: could not fetch ImageNet weights (%s).\n"
                    "Check the network, or pass --no-pretrained to train from scratch." % exc)
        else:
            net = convnext_tiny(weights=None)
            log("  backbone     : ConvNeXt-Tiny, RANDOM init (no pretrained weights)")

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.features = net.features          # [stem, stage1, stage2, stage3, stage4]
        self.avgpool = net.avgpool
        self.norm = net.classifier[0]         # pretrained LayerNorm2d(768)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(768, 1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_u8):
        """x_u8: (B,3,H,W) float32 in [0,255] raw RGB."""
        x = x_u8.float().div_(255.0)
        x = (x - self.mean) / self.std
        x = self.features(x)
        x = self.avgpool(x)
        x = self.norm(x)
        x = torch.flatten(x, 1)
        return self.head(self.dropout(x)).squeeze(1)

    @staticmethod
    def _child_width(child):
        """Output width of one `features` child, read from its convolution weights.

        A CNBlock's widest conv is the 1x1 expansion to 4*dim, so the widest conv is
        4*dim for a block group and dim for a downsample conv (its only conv).
        """
        widest = 0
        for m in child.modules():
            if isinstance(m, nn.Conv2d):
                widest = max(widest, m.out_channels)
        if widest in DermaConvNeXt.STAGE_OF_WIDTH:
            return widest                      # downsample conv: out = dim
        if widest % 4 == 0 and widest // 4 in DermaConvNeXt.STAGE_OF_WIDTH:
            return widest // 4                 # CNBlock group: widest = 4*dim
        return widest

    def feature_stage_map(self):
        """{features index -> stage label} for stem / stage1..stage4."""
        return {i: ("stem" if i == 0 else
                    "stage%d" % self.STAGE_OF_WIDTH[self._child_width(c)])
                for i, c in enumerate(self.features)}

    def stage_names(self, unfreeze_stages):
        names = sorted(set(self.feature_stage_map().values()) - {"stem"})
        if unfreeze_stages <= 0:
            return []
        return names[-unfreeze_stages:] if unfreeze_stages < len(names) else names

    def param_groups(self, lr_head, lr_backbone, unfreeze_stages):
        """Head always trainable; only the trailing `unfreeze_stages` backbone stages."""
        wanted = set(self.stage_names(unfreeze_stages))
        labels = self.feature_stage_map()
        for p in self.features.parameters():
            p.requires_grad_(False)
        for i, child in enumerate(self.features):
            if labels[i] in wanted:
                for p in child.parameters():
                    p.requires_grad_(True)
        for p in self.norm.parameters():
            p.requires_grad_(True)
        for p in self.head.parameters():
            p.requires_grad_(True)

        groups = []
        backbone = [p for p in self.features.parameters() if p.requires_grad]
        if backbone:
            groups.append({"params": backbone, "lr": lr_backbone, "name": "backbone"})
        groups.append({"params": list(self.norm.parameters()) + list(self.head.parameters()),
                       "lr": lr_head, "name": "head"})
        return groups


# --------------------------------------------------------------------------- GPU aug
def augment(x):
    """Batched GPU augmentation on a float32 (B,3,H,W) tensor in [0,1]."""
    B, dev = x.shape[0], x.device

    def u(*shape):
        return torch.rand(*shape, device=dev)

    m = (u(B, 1, 1, 1) < 0.5)
    x = torch.where(m, torch.flip(x, dims=[3]), x)          # horizontal flip
    m = (u(B, 1, 1, 1) < 0.5)
    x = torch.where(m, torch.flip(x, dims=[2]), x)          # vertical flip

    ang = (u(B) * 2 - 1) * (0.15 * math.pi)                 # +-15 deg
    scale = 1.0 + (u(B) * 2 - 1) * 0.15                     # +-15% zoom
    cos, sin = torch.cos(ang) * scale, torch.sin(ang) * scale
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 2] = cos, torch.zeros(B, device=dev)
    theta[:, 1, 1], theta[:, 1, 2] = sin, torch.zeros(B, device=dev)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                      align_corners=False)

    x = x * (1.0 + (u(B, 1, 1, 1) * 2 - 1) * 0.15)                       # brightness
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * (1.0 + (u(B, 1, 1, 1) * 2 - 1) * 0.15) + mean      # contrast
    return x.clamp_(0.0, 1.0)


# --------------------------------------------------------------------------- memory
def autodetect_micro_batch(model, device, ceiling=MAX_MICRO_BATCH, floor=MIN_MICRO_BATCH):
    """Find the largest micro batch that survives a real fwd+bwd+step on this VRAM.

    The probe runs before any freezing, so every parameter still requires grad: that is
    the worst case stage 2 will see, which makes the chosen size safe for both stages.
    The search stops at the first size that fits, walking down from `ceiling`.
    """
    banner("VRAM-SAFE BATCH SIZE")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    chosen = floor
    for bs in range(ceiling, floor - 1, -2):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            x = torch.rand(bs, 3, INPUT_SIZE, INPUT_SIZE, device=device) * 255.0
            y = torch.randint(0, 2, (bs,), device=device).float()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = F.binary_cross_entropy_with_logits(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
            log("  micro batch %2d : OK   peak %.2f GB / %.2f GB VRAM" % (bs, peak, total))
            chosen = bs
            break                                    # largest that fits -> stop here
        except torch.cuda.OutOfMemoryError as exc:
            log("  micro batch %2d : OOM (%s)" % (bs, str(exc)[:70]))
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    del opt, scaler
    torch.cuda.empty_cache()
    accum = max(1, EFFECTIVE_BATCH // chosen)
    log("")
    log("  micro batch      : %d  (largest that fits this GPU)" % chosen)
    log("  grad accumulation: %d" % accum)
    log("  effective batch  : %d" % (chosen * accum))
    return chosen, accum


# --------------------------------------------------------------------------- metrics
def metrics_at(y, p, thr):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr), "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "cancer_sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def sweep_thresholds(y, p, grid=GRID):
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    rows = []
    for t in grid:
        pred = (p >= t).astype(int)
        rows.append({"threshold": float(t),
                     "accuracy": accuracy_score(y, pred),
                     "precision": precision_score(y, pred, zero_division=0),
                     "recall": recall_score(y, pred, zero_division=0),
                     "f1": f1_score(y, pred, zero_division=0)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- runner
class Runner:
    def __init__(self, args):
        self.args = args
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(CKPT_DIR, exist_ok=True)
        self.device = require_cuda()
        set_seed(SEED)

        banner("DATASET")
        root, meta_csv = find_dataset_root()
        self.meta = load_metadata(root, meta_csv)
        self.splits = make_split(self.meta)
        self.cache = build_cache(self.meta)
        id_to_row = {iid: i for i, iid in enumerate(self.meta["image_id"].to_numpy())}
        for df in self.splits.values():
            df["_row"] = df["image_id"].map(id_to_row)
            assert df["_row"].notna().all(), "cache row mapping failed"
            assert (df["_row"].to_numpy() < len(self.meta)).all(), "cache index out of range"

        banner("MODEL")
        self.model = DermaConvNeXt(pretrained=not args.no_pretrained).to(self.device)
        total = sum(p.numel() for p in self.model.parameters())
        log("  total params    : %d (%.2f M)" % (total, total / 1e6))
        log("  input contract  : float32 (B,3,%d,%d) raw RGB in [0,255]; "
            "normalisation INSIDE" % (INPUT_SIZE, INPUT_SIZE))
        log("  output contract : logit -> sigmoid -> p(Cancer)")

        self.micro_bs, self.accum = autodetect_micro_batch(self.model, self.device)
        self.steps_per_epoch = math.ceil(len(self.splits["train"]) / self.micro_bs)

        # images are already 224x224 uint8 in RAM; augmentation happens on the GPU, so
        # worker processes would only add overhead (and on Windows they would each
        # re-pickle the cache). One process is the fastest and lightest option here.
        self.train_loader = self._loader(self.splits["train"],
                                         sampler=self._balanced_sampler())
        self.val_loader = self._loader(self.splits["val"])
        self.test_loader = self._loader(self.splits["test"])
        self.state = self._load_state()

    def _loader(self, df, sampler=None):
        return DataLoader(CachedHAM(self.cache, df), batch_size=self.micro_bs,
                          shuffle=sampler is None, sampler=sampler, num_workers=0,
                          pin_memory=True, drop_last=False)

    def _balanced_sampler(self):
        """50/50 cancer / non-cancer resampling of the TRAIN split only."""
        y = self.splits["train"]["label"].to_numpy()
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        log("  balancing      : %d cancer oversampled to match %d non-cancer"
            % (len(pos), len(neg)))
        w = np.concatenate([np.full(len(pos), 0.5 / max(len(pos), 1)),
                            np.full(len(neg), 0.5 / max(len(neg), 1))])
        gen = torch.Generator().manual_seed(SEED)
        return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                     num_samples=len(pos) + len(neg),
                                     replacement=True, generator=gen)

    # ---------------------------------------------------------------- resume
    def _load_state(self):
        if self.args.fresh:
            log("  resume         : disabled (--fresh)")
            return None
        files = sorted(glob.glob(os.path.join(CKPT_DIR, "*_epoch*.pt")))
        if not files:
            log("  resume         : no checkpoint found, starting fresh")
            return None
        path = files[-1]
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            log("  resume         : %s (stage=%s, epoch=%d, best_val_auc=%.4f)"
                % (os.path.basename(path), ck["stage"], ck["epoch"], ck["best_auc"]))
            return {"path": path, **ck}
        except Exception as exc:
            log("  resume         : %s unreadable (%s) - starting fresh" % (path, exc))
            return None

    def _save(self, stage, epoch, optimizer, scheduler, best_auc):
        path = os.path.join(CKPT_DIR, "%s_epoch%02d.pt" % (stage, epoch))
        tmp = path + ".tmp"
        torch.save({
            "stage": stage, "epoch": epoch, "best_auc": best_auc,
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "np_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "micro_batch": self.micro_bs, "accum": self.accum,
            "config": vars(self.args), "saved_at": datetime.now().isoformat(),
        }, tmp)
        os.replace(tmp, path)
        for old in sorted(glob.glob(os.path.join(CKPT_DIR, "*_epoch*.pt")))[:-KEEP_CKPTS]:
            try:
                os.remove(old)
            except OSError:
                pass

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, loader, df):
        """Returns (probs, labels) aligned positionally with df."""
        self.model.eval()
        probs, labs, idxs = [], [], []
        for xb, yb, ib in loader:
            xb = xb.to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logit = self.model(xb)
            probs.append(torch.sigmoid(logit.float()).cpu().numpy())
            labs.append(yb.numpy())
            idxs.append(ib.numpy())
        p = np.concatenate(probs).astype(np.float64)
        y = np.concatenate(labs).astype(np.int32)
        order = np.concatenate(idxs)
        out_p, out_y = np.empty_like(p), np.empty_like(y)
        out_p[order], out_y[order] = p, y
        assert np.array_equal(out_y, df["label"].to_numpy().astype(np.int32)), \
            "prediction order does not match the split frame"
        assert np.all((out_p >= 0) & (out_p <= 1)), "probabilities outside [0,1]"
        return out_p, out_y

    @staticmethod
    def _loss(logits, yb):
        target = yb * (1.0 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
        return F.binary_cross_entropy_with_logits(logits, target)

    def _save_best(self, vp, vy, val_auc, stage, epoch):
        path = os.path.join(RESULTS_DIR, "convnext_tiny_best.pt")
        tmp = path + ".tmp"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "architecture": "ConvNeXt-Tiny (torchvision IMAGENET1K_V1) + GAP + "
                            "Dropout(%.2f) + Linear(768,1)" % DROPOUT,
            "input_size": [INPUT_SIZE, INPUT_SIZE, 3],
            "input_contract": "float32 (B,3,224,224) raw RGB in [0,255]; "
                              "ImageNet mean/std normalisation is INSIDE the module",
            "output_contract": "raw logit; apply sigmoid for p(Cancer)",
            "label_mapping": {"1": "Cancer (dx in mel, bcc, akiec)",
                              "0": "Non-Cancer (dx in nv, bkl, vasc, df)"},
            "val_auc": val_auc, "stage": stage, "epoch": epoch, "seed": SEED,
            "unfreeze_stages": UNFREEZE_STAGES, "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "trained_at": datetime.now().isoformat(),
        }, tmp)
        os.replace(tmp, path)
        np.save(os.path.join(RESULTS_DIR, "val_proba_best_epoch.npy"), vp)
        np.save(os.path.join(RESULTS_DIR, "val_true_best_epoch.npy"), vy)
        log("      * best val_auc %.4f -> %s" % (val_auc, os.path.basename(path)))

    # ---------------------------------------------------------------- training
    def train_stage(self, stage, epochs, unfreeze_stages):
        banner("%s - %s" % (stage.upper(), "frozen backbone, head only"
                            if unfreeze_stages == 0 else "fine-tune %s"
                            % ", ".join(self.model.stage_names(unfreeze_stages))))
        groups = self.model.param_groups(LR_HEAD, LR_BACKBONE, unfreeze_stages)
        n_train = sum(p.numel() for g in groups for p in g["params"])
        n_total = sum(p.numel() for p in self.model.parameters())
        for g in groups:
            log("  group %-9s lr=%-9g params=%d"
                % (g["name"], g["lr"], sum(p.numel() for p in g["params"])))
        log("  trainable       : %d / %d (%.2f%%)" % (n_train, n_total,
                                                      100 * n_train / n_total))
        frozen = [name for name, m in self.model.features.named_children()
                  if not any(p.requires_grad for p in m.parameters())]
        log("  frozen backbone : %s" % (", ".join(frozen) if frozen else "none"))

        optimizer = torch.optim.AdamW(groups, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda e: 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(e, epochs)
                                                          / max(epochs, 1))))
        scaler = torch.amp.GradScaler("cuda")

        start_epoch, best_auc = 0, -1.0
        st = self.state
        resumed_from = st.get("stage") if st else None
        if resumed_from in STAGE_ORDER and \
                STAGE_ORDER.index(resumed_from) > STAGE_ORDER.index(stage):
            # the checkpoint is from a LATER stage, so this one is already finished.
            # self.state is left intact: the next stage still needs it.
            log("  already past this stage (checkpoint is from %s) - skipping" % resumed_from)
            return []
        if resumed_from == stage:
            self.model.load_state_dict(st["model"])
            if st.get("optimizer"):
                optimizer.load_state_dict(st["optimizer"])
            if st.get("scheduler"):
                scheduler.load_state_dict(st["scheduler"])
            torch.set_rng_state(st["torch_rng"])
            if st.get("cuda_rng"):
                torch.cuda.set_rng_state_all(st["cuda_rng"])
            np.random.set_state(st["np_rng"])
            random.setstate(st["python_rng"])
            start_epoch = int(st["epoch"]) + 1
            best_auc = float(st.get("best_auc", -1.0))
            log("  RESUMED at epoch %d/%d (best val AUC so far %.4f)"
                % (start_epoch, epochs, best_auc))
        elif resumed_from in STAGE_ORDER:
            self.model.load_state_dict(st["model"])
            log("  carried the %s weights into %s" % (resumed_from, stage))
        self.state = None

        rows, bad, t0 = [], 0, time.time()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            run_loss, seen = 0.0, 0
            te = time.time()
            steps = len(self.train_loader)
            for step, (xb, yb, _ib) in enumerate(self.train_loader):
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss = self._loss(self.model(xb).float(), yb)
                scaler.scale(loss / self.accum).backward()
                if (step + 1) % self.accum == 0 or (step + 1) == steps:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                run_loss += float(loss.detach()) * len(yb)
                seen += len(yb)
                if self.args.progress and step % 60 == 0:
                    log("        step %4d/%d  loss %.4f  gpu %.2f GB"
                        % (step, steps, float(loss.detach()),
                           torch.cuda.max_memory_allocated() / 1024 ** 3))
            tr_loss = run_loss / max(seen, 1)

            vp, vy = self.predict(self.val_loader, self.splits["val"])
            from sklearn.metrics import roc_auc_score, accuracy_score
            val_auc = float(roc_auc_score(vy, vp))
            acc05 = float(accuracy_score(vy, (vp >= 0.5).astype(int)))
            rows.append({"stage": stage, "epoch": epoch, "train_loss": tr_loss,
                         "val_auc": val_auc, "val_acc_at_0.5": acc05,
                         "lr": optimizer.param_groups[0]["lr"],
                         "epoch_seconds": time.time() - te})
            log("  %s epoch %2d/%d  loss %.4f  val_auc %.4f  val_acc@0.5 %.4f  "
                "lr %.2e  (%.0f s)"
                % (stage, epoch + 1, epochs, tr_loss, val_auc, acc05,
                   optimizer.param_groups[0]["lr"], time.time() - te))

            improved = val_auc > best_auc
            if improved:
                best_auc = val_auc
                self._save_best(vp, vy, val_auc, stage, epoch)
            bad = 0 if improved else bad + 1
            self._save(stage, epoch, optimizer, scheduler, best_auc)
            scheduler.step()
            if bad >= PATIENCE:
                log("  EARLY STOPPING - %d epochs without a val_auc improvement" % bad)
                break
        log("  %s finished in %.1f min | best val_auc %.4f" % (stage, (time.time() - t0) / 60,
                                                              best_auc))
        return rows

    def load_best(self):
        path = os.path.join(RESULTS_DIR, "convnext_tiny_best.pt")
        if not os.path.exists(path):
            raise SystemExit("FATAL: no best model was saved - training did not complete.")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.to(self.device).eval()
        log("  best model    : %s" % path)
        log("                 stage=%s epoch=%d val_auc=%.4f saved=%s"
            % (ck["stage"], ck["epoch"], ck["val_auc"], ck["trained_at"]))
        return ck


# --------------------------------------------------------------------------- figures
def make_figures(vy, vp, ty, tp, thr, val_m, test_m, best_auc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, roc_curve

    fpr_v, tpr_v, _ = roc_curve(vy, vp)
    fpr_t, tpr_t, _ = roc_curve(ty, tp)
    np.savetxt(os.path.join(RESULTS_DIR, "roc_curve_val.csv"),
               np.column_stack([fpr_v, tpr_v]), delimiter=",", header="fpr,tpr", comments="")
    np.savetxt(os.path.join(RESULTS_DIR, "roc_curve_test.csv"),
               np.column_stack([fpr_t, tpr_t]), delimiter=",", header="fpr,tpr", comments="")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr_v, tpr_v, lw=2, label="validation  AUC = %.4f" % val_m["roc_auc"])
    ax.plot(fpr_t, tpr_t, lw=2, label="test  AUC = %.4f" % test_m["roc_auc"])
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ConvNeXt-Tiny ROC\nthreshold %.3f chosen on validation only" % thr)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    for k, (y, p, ttl, m) in enumerate((
            (vy, vp, "VALIDATION", val_m),
            (ty, tp, "TEST (scored once)", test_m))):
        cm = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[k])
        axes[k].set_title("%s confusion matrix @ %.3f\nacc %.4f | sens %.4f | spec %.4f | F1 %.4f"
                          % (ttl, thr, m["accuracy"], m["cancer_sensitivity"],
                             m["specificity"], m["f1"]))
        axes[k].set_xlabel("predicted")
        axes[k].set_ylabel("true")
        np.savetxt(os.path.join(RESULTS_DIR, "confusion_matrix_%s.csv"
                                % ("val" if k == 0 else "test")),
                   cm, delimiter=",", header=",".join(CLASS_NAMES),
                   comments="rows = true (%s)" % ",".join(CLASS_NAMES))
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=130)
    plt.close(fig)

    hists = [os.path.join(RESULTS_DIR, "history_%s.csv" % s) for s in ("stage1", "stage2")]
    hists = [h for h in hists if os.path.exists(h)]
    if hists:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        off = 0
        for h in hists:
            d = pd.read_csv(h)
            axes[0].plot(range(off, off + len(d)), d["val_auc"], "o-",
                         label=os.path.basename(h).replace("history_", "").replace(".csv", ""))
            axes[1].plot(range(off, off + len(d)), d["train_loss"], "o-",
                         label=os.path.basename(h).replace("history_", "").replace(".csv", ""))
            off += len(d)
        axes[0].axhline(best_auc, color="k", ls="--", lw=1, label="best val AUC %.4f" % best_auc)
        axes[0].set_title("validation ROC-AUC per epoch")
        axes[0].set_xlabel("epoch (stage1 then stage2)")
        axes[1].set_title("training loss (label-smoothed BCE)")
        axes[1].set_xlabel("epoch")
        for a in axes:
            a.grid(alpha=0.3)
            a.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=130)
        plt.close(fig)

    log("  figures  : roc_curve.png, confusion_matrix.png, training_curves.png")


# --------------------------------------------------------------------------- report
def report(r, best_ck):
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve

    def epochs_actually_run(stage):
        path = os.path.join(RESULTS_DIR, "history_%s.csv" % stage)
        if not os.path.exists(path):
            return None
        return int(len(pd.read_csv(path)))

    def best_epoch_of(stage):
        path = os.path.join(RESULTS_DIR, "history_%s.csv" % stage)
        if not os.path.exists(path):
            return None
        d = pd.read_csv(path)
        return int(d.loc[d["val_auc"].idxmax(), "epoch"]) if len(d) else None

    banner("PREDICTIONS")
    vp, vy = r.predict(r.val_loader, r.splits["val"])
    tp, ty = r.predict(r.test_loader, r.splits["test"])
    log("  VAL  n=%d  mean p(cancer)=%.4f  mean p(non-cancer)=%.4f"
        % (len(vy), vp[vy == 1].mean(), vp[vy == 0].mean()))
    log("  TEST n=%d  mean p(cancer)=%.4f  mean p(non-cancer)=%.4f"
        % (len(ty), tp[ty == 1].mean(), tp[ty == 0].mean()))
    np.save(os.path.join(RESULTS_DIR, "val_proba.npy"), vp)
    np.save(os.path.join(RESULTS_DIR, "val_true.npy"), vy)
    np.save(os.path.join(RESULTS_DIR, "test_proba.npy"), tp)
    np.save(os.path.join(RESULTS_DIR, "test_true.npy"), ty)

    # ------------------------------------------------ threshold: VALIDATION ONLY
    banner("THRESHOLD SELECTION (validation split only)")
    sweep = sweep_thresholds(vy, vp)
    sweep.to_csv(os.path.join(RESULTS_DIR, "val_threshold_sweep.csv"), index=False)
    i_f1 = int(sweep["f1"].idxmax())
    i_acc = int(sweep["accuracy"].idxmax())
    hi = sweep[sweep["recall"] >= 0.95]
    THRESHOLD = float(sweep.loc[i_f1, "threshold"])
    THR_ACC = float(sweep.loc[i_acc, "threshold"])
    log("  argmax F1            : t=%.3f  acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f"
        % (THRESHOLD, sweep.loc[i_f1, "accuracy"], sweep.loc[i_f1, "precision"],
           sweep.loc[i_f1, "recall"], sweep.loc[i_f1, "f1"]))
    log("  argmax accuracy      : t=%.3f  f1=%.4f  (reported, not used)"
        % (THR_ACC, sweep.loc[i_acc, "f1"]))
    if len(hi):
        i_s95 = int(hi["f1"].idxmax())
        THR_S95 = float(sweep.loc[i_s95, "threshold"])
        log("  sensitivity >= 0.95  : t=%.3f  prec=%.4f  f1=%.4f  (reported, not used)"
            % (THR_S95, sweep.loc[i_s95, "precision"], sweep.loc[i_s95, "f1"]))
    else:
        # Do not silently fall back to the F1-optimal row: say the target is unreachable.
        THR_S95 = None
        ceil_recall = float((vp[vy == 1] >= GRID.min()).mean())
        n_below = int((vp[vy == 1] < GRID.min()).sum())
        log("  sensitivity >= 0.95  : NOT ACHIEVABLE on validation")
        log("                         recall peaks at %.4f even at the grid floor %.3f:"
            " %d of %d validation cancers (%.1f%%) score below it"
            % (ceil_recall, GRID.min(), n_below, int((vy == 1).sum()), 100 * (1 - ceil_recall)))
    log("  FROZEN THRESHOLD     : %.4f   (test never used for tuning)" % THRESHOLD)

    val_m = metrics_at(vy, vp, THRESHOLD)
    test_m = metrics_at(ty, tp, THRESHOLD)          # <-- single test evaluation

    def show(tag, m):
        log("")
        log("  %s @ threshold %.4f" % (tag, m["threshold"]))
        log("    accuracy %.4f | precision %.4f | recall %.4f | F1 %.4f | ROC-AUC %.4f"
            % (m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc"]))
        log("    cancer sensitivity %.4f | specificity %.4f"
            % (m["cancer_sensitivity"], m["specificity"]))
        log("    tn=%d fp=%d fn=%d tp=%d  (n=%d)"
            % (m["tn"], m["fp"], m["fn"], m["tp"], m["n"]))

    banner("VALIDATION METRICS (this split chose the threshold)")
    show("VALIDATION", val_m)
    banner("TEST METRICS (one pass, threshold frozen from validation)")
    show("TEST", test_m)

    test_pred = (tp >= THRESHOLD).astype(int)
    test_cm = confusion_matrix(ty, test_pred, labels=[0, 1])
    assert np.asarray(test_cm).shape == (2, 2), \
        "confusion matrix is %s, expected (2, 2)" % (np.shape(test_cm),)
    rep_txt = classification_report(ty, test_pred, target_names=CLASS_NAMES,
                                    digits=4, zero_division=0)
    log("")
    log(rep_txt)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("DermaAI - ConvNeXt-Tiny - TEST classification report\n")
        fh.write("=" * 62 + "\n")
        fh.write("threshold      : %.4f  (argmax F1 on the VALIDATION split only)\n" % THRESHOLD)
        fh.write("test split     : evaluated exactly once\n")
        fh.write("GPU            : %s\n" % torch.cuda.get_device_name(0))
        fh.write("test ROC-AUC   : %.4f\n\n" % test_m["roc_auc"])
        fh.write(rep_txt)
        fh.write("\nConfusion matrix (rows = true, columns = predicted)\n")
        fh.write("%-22s %12s %12s\n" % ("", *CLASS_NAMES))
        for i, row in enumerate(test_cm):
            fh.write("%-22s %12d %12d\n" % (CLASS_NAMES[i], row[0], row[1]))

    def per_dx(p, thr):
        df = r.splits["test"].copy()
        t = df.assign(pred=(p >= thr).astype(int)).groupby("dx")["pred"].agg(["sum", "count"])
        t["detected_rate"] = t["sum"] / t["count"]
        t["binary_class"] = t.index.map(lambda d: "Cancer" if d in CANCER_CLASSES
                                        else "Non-Cancer")
        return t.sort_values("binary_class")

    per = per_dx(tp, THRESHOLD)
    log("")
    log("  per-dx detection rate on TEST at threshold %.3f:" % THRESHOLD)
    log(per.to_string())
    mel_mask = r.splits["test"]["dx"].to_numpy() == "mel"
    mel_sens = float((tp[mel_mask] >= THRESHOLD).mean()) if mel_mask.any() else float("nan")

    banner("FIGURES + ARTIFACTS")
    make_figures(vy, vp, ty, tp, THRESHOLD, val_m, test_m, best_ck["val_auc"])

    payload = {
        "model": "ConvNeXt-Tiny (torchvision IMAGENET1K_V1) + GAP + Dropout(%.2f) + "
                 "Linear(768,1)" % DROPOUT,
        "framework": "PyTorch %s / CUDA %s" % (torch.__version__, torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2),
        "compute_device": "cuda",
        "cpu_training_used": False,
        "mixed_precision": "float16 autocast + GradScaler (torch.amp)",
        "input_shape": [INPUT_SIZE, INPUT_SIZE, 3],
        "input_contract": "float32 (B,3,224,224) raw RGB in [0,255]; ImageNet mean/std "
                          "normalisation is INSIDE the module",
        "output": "logit -> sigmoid -> p(Cancer)",
        "label_source": "HAM10000_metadata.csv 'dx' column (filenames never used as labels)",
        "label_mapping": {"1": "Cancer (dx in mel, bcc, akiec)",
                          "0": "Non-Cancer (dx in nv, bkl, vasc, df)"},
        "dataset_root": r.meta.attrs.get("root"),
        "split": {
            "grouping": "lesion_id from HAM10000_metadata.csv",
            "ratios": "70/15/15", "seed": SEED,
            "train_images": len(r.splits["train"]), "val_images": len(r.splits["val"]),
            "test_images": len(r.splits["test"]),
            "train_lesions": int(r.splits["train"]["group"].nunique()),
            "val_lesions": int(r.splits["val"]["group"].nunique()),
            "test_lesions": int(r.splits["test"]["group"].nunique()),
            "lesion_overlap_between_splits": 0,
            "matches_production_split": {
                n: (len(r.splits[n]) == REFERENCE_SPLIT[n][0]
                    and int(r.splits[n]["group"].nunique()) == REFERENCE_SPLIT[n][1]
                    and r.splits[n]["label_name"].value_counts().to_dict()
                    == REFERENCE_SPLIT[n][2]) for n in ("train", "val", "test")},
        },
        "class_counts": {k: v["label_name"].value_counts().to_dict()
                         for k, v in r.splits.items()},
        "class_balancing": "WeightedRandomSampler 50/50 on the TRAIN split only",
        "batch": {"micro_batch": r.micro_bs, "grad_accum": r.accum,
                  "effective_batch": r.micro_bs * r.accum,
                  "steps_per_epoch": r.steps_per_epoch},
        "stage1": {"epochs_requested": r.args.stage1_epochs,
                   "epochs_completed": epochs_actually_run("stage1"),
                   "lr_head": LR_HEAD, "backbone": "frozen",
                   "weight_decay": WEIGHT_DECAY, "label_smoothing": LABEL_SMOOTH},
        "stage2": {"epochs_requested": r.args.stage2_epochs,
                   "epochs_completed": epochs_actually_run("stage2"),
                   "best_epoch": best_epoch_of("stage2"),
                   "early_stopping": "val ROC-AUC, patience %d" % PATIENCE,
                   "early_stopping_triggered": ((epochs_actually_run("stage2") or 0)
                                                < r.args.stage2_epochs),
                   "lr_head": LR_HEAD, "lr_backbone": LR_BACKBONE,
                   "unfrozen_stages": r.model.stage_names(r.args.unfreeze_stages),
                   "weight_decay": WEIGHT_DECAY, "label_smoothing": LABEL_SMOOTH},
        "best_checkpoint": {"val_auc": best_ck["val_auc"], "stage": best_ck["stage"],
                            "epoch": best_ck["epoch"]},
        "threshold": {
            "selected_on": "VALIDATION ONLY (argmax F1 over 0.05..0.95 step 0.005)",
            "value": THRESHOLD,
            "accuracy_optimal_alternative": THR_ACC,
            "sensitivity_ge_0.95_alternative": THR_S95,
            "sensitivity_ge_0.95_achievable_on_validation": THR_S95 is not None,
            "validation_max_recall_on_grid": float((vp[vy == 1] >= GRID.min()).mean()),
            "app_py_current_threshold": 0.5,
            "app_py_changed": False,
        },
        "validation_metrics": val_m,
        "test_metrics": test_m,
        "test_melanoma_sensitivity": None if math.isnan(mel_sens) else mel_sens,
        "test_classification_report": classification_report(
            ty, test_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "test_per_dx_detection_rate": per.to_dict(orient="index"),
        "production_model_untouched": True,
        "production_model_path": PRODUCTION_MODEL,
        "finished_at": datetime.now().isoformat(),
    }

    def safe(o):
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return str(o)

    for name, obj in (("metrics.json", payload),
                      ("threshold.json", {
                          "threshold": THRESHOLD,
                          "rule": "argmax F1 on the VALIDATION split only",
                          "accuracy_optimal_alternative": THR_ACC,
                          "sensitivity_ge_0.95_alternative": THR_S95,
                          "app_py_remains": 0.5}),
                      ("config.json", {
                          "seed": SEED, "input_size": INPUT_SIZE,
                          "cancer_classes": CANCER_CLASSES, "class_names": CLASS_NAMES,
                          "lr_head": LR_HEAD, "lr_backbone": LR_BACKBONE,
                          "weight_decay": WEIGHT_DECAY, "label_smoothing": LABEL_SMOOTH,
                          "dropout": DROPOUT, "imagenet_mean": IMAGENET_MEAN,
                          "imagenet_std": IMAGENET_STD, "args": vars(r.args)})):
        with open(os.path.join(RESULTS_DIR, name), "w") as fh:
            json.dump(obj, fh, indent=2, default=safe, allow_nan=False)
        json.load(open(os.path.join(RESULTS_DIR, name)))     # strict re-parse
    log("  metrics.json / threshold.json / config.json re-parsed as strict JSON: OK")

    banner("FINAL RESULT (test split, single evaluation at the frozen threshold)")
    log("  GPU                  : %s (%.1f GB VRAM)"
        % (torch.cuda.get_device_name(0),
           torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))
    log("  training status      : stage1 %s ep frozen + stage2 %s ep fine-tune"
        % (epochs_actually_run("stage1") or r.args.stage1_epochs,
           epochs_actually_run("stage2") or r.args.stage2_epochs))
    if (epochs_actually_run("stage2") or 0) < r.args.stage2_epochs:
        log("                         (stage 2 early-stopped on val ROC-AUC, patience %d)"
            % PATIENCE)
    log("  test ROC-AUC         : %.4f" % test_m["roc_auc"])
    log("  accuracy             : %.4f" % test_m["accuracy"])
    log("  precision            : %.4f" % test_m["precision"])
    log("  recall               : %.4f" % test_m["recall"])
    log("  sensitivity (cancer) : %.4f" % test_m["cancer_sensitivity"])
    log("  specificity          : %.4f" % test_m["specificity"])
    log("  F1                   : %.4f" % test_m["f1"])
    log("  threshold            : %.4f" % THRESHOLD)
    log("  model                : %s" % os.path.join(RESULTS_DIR, "convnext_tiny_best.pt"))
    log("")
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "*"))):
        if os.path.isfile(p):
            log("   %-34s %9.2f MB" % (os.path.basename(p), os.path.getsize(p) / 1e6))
        else:
            log("   %-34s <dir>" % os.path.basename(p))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="DermaAI ConvNeXt-Tiny training (GPU only)")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    ap.add_argument("--no-pretrained", action="store_true", help="random init backbone")
    ap.add_argument("--stage1-epochs", type=int, default=STAGE1_EPOCHS)
    ap.add_argument("--stage2-epochs", type=int, default=STAGE2_EPOCHS)
    ap.add_argument("--unfreeze-stages", type=int, default=UNFREEZE_STAGES)
    ap.add_argument("--progress", action="store_true", help="per-step logging")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training, evaluate the saved best model")
    args = ap.parse_args()

    log("DermaAI - ConvNeXt-Tiny on HAM10000")
    log("started %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log("production model (never read or written): %s" % PRODUCTION_MODEL)

    r = Runner(args)

    if args.eval_only:
        banner("EVAL ONLY - loading the saved best model, no training")
        ck = r.load_best()
        report(r, ck)
        return

    banner("TRAINING PLAN")
    log("  stage 1: %d epochs, frozen backbone, head lr %g"
        % (args.stage1_epochs, LR_HEAD))
    log("  stage 2: %d epochs, unfreeze last %d ConvNeXt stage(s), head lr %g / backbone lr %g"
        % (args.stage2_epochs, args.unfreeze_stages, LR_HEAD, LR_BACKBONE))
    log("  mixed precision: float16 autocast + GradScaler")
    log("  loss: BCE with label smoothing %.2f | dropout %.2f | weight decay %g"
        % (LABEL_SMOOTH, DROPOUT, WEIGHT_DECAY))
    log("  early stopping: val AUC, patience %d" % PATIENCE)
    log("  steps per epoch: %d" % r.steps_per_epoch)

    import pandas as pd
    for stage, epochs_done, unfreeze in (("stage1", args.stage1_epochs, 0),
                                         ("stage2", args.stage2_epochs, args.unfreeze_stages)):
        rows = r.train_stage(stage, epochs_done, unfreeze)
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "history_%s.csv" % stage),
                                      index=False)
            log("  history_%s.csv written (%d epochs completed)" % (stage, len(rows)))
        else:
            log("  history_%s.csv left as-is (stage skipped, already on disk)" % stage)

    report(r, r.load_best())
    log("")
    log("finished %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
