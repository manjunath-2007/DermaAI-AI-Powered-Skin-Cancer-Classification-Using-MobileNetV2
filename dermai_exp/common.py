"""Shared paths, constants, logging and seeding.

Contract inherited from train_convnext_tiny.py / dermai_convnext_tiny_colab.ipynb:
  * labels come from HAM10000_metadata.csv 'dx' only
  * label 1 = Cancer   (dx in mel, bcc, akiec)
  * label 0 = Non-Cancer(dx in nv, bkl, vasc, df)
  * grouping key = lesion_id (no lesion may appear in two splits)
  * split = lesion-level stratified 70/15/15, random_state=42
  * model/skin_cancer_mobilenetv2.h5 is never opened
"""

import logging
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)

RESULTS_ROOT = os.path.join(PROJECT, "results")
EXP_ROOT = os.path.join(RESULTS_ROOT, "experiments")
FINAL_ROOT = os.path.join(RESULTS_ROOT, "final")
CKPT_ROOT = os.path.join(RESULTS_ROOT, "_ckpt")

PRODUCTION_MODEL = os.path.join(PROJECT, "model", "skin_cancer_mobilenetv2.h5")

CACHE_ROOT = os.environ.get("DERMAI_CACHE", r"D:\dermai_cache")
CACHE_MASTER = os.path.join(CACHE_ROOT, "ham10000_master_384x384_u8.npy")

DATASET_CANDIDATES = [
    r"D:\User\Downloads\archive",
    r"D:\ham10000",
    os.path.join(PROJECT, "HAM10000"),
    r"C:\Users\Manjunath\Downloads\archive",
]

SEED = 42
CANCER_CLASSES = ("mel", "bcc", "akiec")
BENIGN_CLASSES = ("nv", "bkl", "vasc", "df")
CLASS_NAMES = ("Non-Cancer", "Cancer")

MASTER_SIZE = 384            # highest resolution any experiment needs
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# numbers produced by the verified production split (model/model_card.json)
REFERENCE_SPLIT = {
    "train": (7010, 5229, {"Non-Cancer": 5641, "Cancer": 1369}),
    "val": (1500, 1120, {"Non-Cancer": 1203, "Cancer": 297}),
    "test": (1505, 1121, {"Non-Cancer": 1217, "Cancer": 288}),
}

THRESHOLD_GRID = np.round(np.arange(0.02, 0.981, 0.0025), 6)


# --------------------------------------------------------------------------- logging
class _Fmt(logging.Formatter):
    def format(self, record):
        msg = record.msg
        if record.args:
            msg = msg % record.args
        return msg


class _Logger(logging.Logger):
    """`Logger.__call__` was removed in Python 3.13; the whole codebase logs as `log(...)`."""

    def __call__(self, msg, *args, **kwargs):
        self.info(msg, *args, **kwargs)


logging.setLoggerClass(_Logger)


def get_logger(name="dermai", level=logging.INFO, logfile=None):
    lg = logging.getLogger(name)
    if lg.handlers:
        for h in list(lg.handlers):
            if isinstance(h, logging.FileHandler) and (
                    not logfile or os.path.abspath(h.baseFilename)
                    != os.path.abspath(logfile)):
                lg.removeHandler(h)
                h.close()
            elif not isinstance(h, logging.FileHandler):
                lg.removeHandler(h)
        if lg.handlers:
            return lg
    lg.setLevel(level)
    lg.propagate = False
    out = logging.StreamHandler(sys.stdout)
    out.setFormatter(_Fmt("%(message)s"))
    lg.addHandler(out)
    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        fh = logging.FileHandler(logfile, mode="a", encoding="utf-8")
        fh.setFormatter(_Fmt("%(asctime)s | %(message)s", "%H:%M:%S"))
        lg.addHandler(fh)
    return lg


class Timer:
    def __init__(self, log, label):
        self.log, self.label = log, label

    def __enter__(self):
        self.t0 = time.time()
        self.log("  %s ..." % self.label)
        return self

    def __exit__(self, *exc):
        self.log("  %s done in %.1f s" % (self.label, time.time() - self.t0))
        return False


# --------------------------------------------------------------------------- seeding
def set_seed(seed=SEED, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Unbounded cudnn autotuning spent 13 s on one shape on this 4 GB card and then
    # thrashed; a small limit keeps the search cheap without disabling it.
    if hasattr(torch.backends.cudnn, "benchmark_limit"):
        torch.backends.cudnn.benchmark_limit = 4
    torch.set_float32_matmul_precision("high")


def worker_init(worker_id):
    s = (torch.initial_seed() + worker_id) % (2 ** 31)
    np.random.seed(s)
    random.seed(s)


def require_cuda(log):
    if not torch.cuda.is_available():
        raise SystemExit("FATAL: CUDA is required; CPU training is disabled by design.")
    props = torch.cuda.get_device_properties(0)
    log("  torch %s | CUDA %s | cudnn %s" % (torch.__version__, torch.version.cuda,
                                            torch.backends.cudnn.version()))
    log("  GPU  %s | %.2f GB VRAM | sm_%d%d"
        % (props.name, props.total_memory / 1024 ** 3, props.major, props.minor))
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    (a @ a).float().sum().item()
    torch.cuda.synchronize()
    return torch.device("cuda")


def ensure_dirs():
    for d in (RESULTS_ROOT, EXP_ROOT, FINAL_ROOT, CKPT_ROOT, CACHE_ROOT):
        os.makedirs(d, exist_ok=True)