"""DermaAI configuration - single source of truth for model, paths, and thresholds."""

import os

# ──────────────────────────────────────────────────────────────────────
# Base paths
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "convnext_tiny")

# ──────────────────────────────────────────────────────────────────────
# Model selection
# ──────────────────────────────────────────────────────────────────────
# Options: "convnext_tiny" (PyTorch, recommended) or "mobilenetv2" (TF/Keras)
DEFAULT_MODEL = "convnext_tiny"

# ──────────────────────────────────────────────────────────────────────
# ConvNeXt-Tiny (PyTorch) settings
# ──────────────────────────────────────────────────────────────────────
# Primary: stage-2 epoch-7 "best-val" checkpoint (convnext_tiny_best.pt).
# These are the weights that produced the published ConvNeXt-Tiny results
# (results/convnext_tiny/metrics.json, test_proba.npy / val_proba.npy) and the
# 0.525 decision threshold, so the deployed model and the reported metrics
# describe the same weights.
CONVNEXT_CKPT = os.path.join(RESULTS_DIR, "convnext_tiny_best.pt")
# Fallback: the last stage-2 checkpoint (epoch 13), used only if the primary
# file is absent. Its weights are NOT the ones behind the published metrics.
CONVNEXT_BEST = os.path.join(RESULTS_DIR, "checkpoints", "stage2_epoch13.pt")
CONVNEXT_THRESHOLD = 0.525          # argmax F1 on validation
CONVNEXT_INPUT_SIZE = (224, 224)
CONVNEXT_BATCH_SIZE = 32            # effective batch for inference
CONVNEXT_DEVICE = "cuda"            # "cuda" or "cpu"

# ConvNeXt preprocessing: raw RGB [0, 255] -> ImageNet norm INSIDE model
CONVNEXT_MEAN = (0.485, 0.456, 0.406)
CONVNEXT_STD = (0.229, 0.224, 0.225)

# ──────────────────────────────────────────────────────────────────────
# MobileNetV2 (TensorFlow/Keras) settings - legacy production model
# ──────────────────────────────────────────────────────────────────────
MOBILENET_PATH = os.path.join(MODEL_DIR, "skin_cancer_mobilenetv2.h5")
MOBILENET_THRESHOLD = 0.22          # argmax F1 on validation (see model_card.json)
MOBILENET_INPUT_SIZE = (224, 224)
MOBILENET_TEMPERATURE = 1.0         # no calibration applied

# ──────────────────────────────────────────────────────────────────────
# General inference settings
# ──────────────────────────────────────────────────────────────────────
ALLOWED_EXT = {"png", "jpg", "jpeg"}
MAX_FILE_SIZE = 10 * 1024 * 1024    # 10 MB
MAX_BATCH_SIZE = 64                 # max images per batch request
MAX_BATCH_FILE_SIZE = 50 * 1024 * 1024  # 50 MB total for batch

# Risk level thresholds (on calibrated cancer probability)
RISK_HIGH = 0.6
RISK_MEDIUM = 0.3

# ──────────────────────────────────────────────────────────────────────
# Label mapping (shared by both models)
# ──────────────────────────────────────────────────────────────────────
CANCER_CLASSES = ["mel", "bcc", "akiec"]
NON_CANCER_CLASSES = ["nv", "bkl", "vasc", "df"]

LABEL_MAP = {
    1: "Cancer (mel, bcc, akiec)",
    0: "Non-Cancer (nv, bkl, vasc, df)",
}

# ──────────────────────────────────────────────────────────────────────
# Flask settings
# ──────────────────────────────────────────────────────────────────────
FLASK_HOST = os.environ.get("DERMA_HOST", "0.0.0.0")
FLASK_PORT = int(os.environ.get("DERMA_PORT", "5000"))
FLASK_DEBUG = os.environ.get("DERMA_DEBUG", "false").lower() == "true"

# ──────────────────────────────────────────────────────────────────────
# Image quality checks
# ──────────────────────────────────────────────────────────────────────
# Lightweight pre-prediction quality screening using OpenCV.
# Checks run on the already-decoded PIL Image (no re-decoding).
# Image is downscaled to QUALITY_ANALYSIS_SIZE for fast computation;
# metrics are scale-invariant so thresholds apply regardless of upload size.
#
# blur_score        = variance of the Laplacian on the grayscale image.
#                     Lower values indicate blurrier images.
#                     A score below QUALITY_BLUR_THRESHOLD triggers a blur warning.
# mean_brightness   = mean pixel intensity of grayscale, normalised to [0, 1].
#                     Below QUALITY_BRIGHTNESS_DARK triggers a "too dark" warning.
#                     Above QUALITY_BRIGHTNESS_BRIGHT triggers an "overexposed" warning.
# overexposure_ratio = fraction of pixels with intensity > 240/255 (~0.941).
#                      Above QUALITY_OVEREXPOSURE_RATIO triggers an overexposure warning.
#
# Warnings never reject the image — predictions still run. Only completely
# un-decodable images are rejected (existing error handling in app.py).
QUALITY_ENABLED = True
QUALITY_ANALYSIS_SIZE = (256, 256)  # downscale for fast, scale-invariant metrics
QUALITY_BLUR_THRESHOLD = 100.0      # Laplacian variance (scale-dependent; tuned for 256×256)
QUALITY_BRIGHTNESS_DARK = 0.20      # mean brightness < 0.20 → "too dark"
QUALITY_BRIGHTNESS_BRIGHT = 0.95    # mean brightness > 0.95 → "too bright"
QUALITY_OVEREXPOSURE_RATIO = 0.10   # >10% near-white pixels → "overexposed"

# ──────────────────────────────────────────────────────────────────────
# Input-domain gate (out-of-distribution rejection)
# ──────────────────────────────────────────────────────────────────────
# The ConvNeXt classifier was trained only on HAM10000 dermoscopic images and
# has no notion of "this image is not a skin image at all". Ordinary photos
# that happen to contain a dark blob on a light field (fruit in a bowl, a car,
# a classroom scene) are therefore scored as malignant lesions with high
# confidence.
#
# The gate measures how far an input sits from the HAM10000 training manifold
# in the deployed model's own 768-d penultimate feature space and refuses to
# emit a diagnosis for images outside that domain. The reference matrix holds
# the embeddings of the HAM10000 TRAIN split only - the validation and test
# splits stay a clean held-out check of the gate's false-reject rate.
#
# Score = mean cosine distance to the k nearest reference embeddings, taken as
# the worst case over {original, horizontal flip} (the model is orientation
# sensitive, so an in-domain image must be explainable in at least one view).
#
# Calibration (see results/convnext_tiny/domain_reference_meta.json):
#   HAM10000 val  max = 0.636   HAM10000 test max = 0.630   -> always in domain
#   real lesion photos max = 0.877 -> in domain
#   ordinary photographs     = 0.59 .. 0.93
#
# The two thresholds below define an ambiguous band. Measured on 100 real
# photographs (30 genuine lesion photographs, 70 ordinary photographs across
# the 14 categories), the band is NOT resolved by the MobileNetV2 second
# opinion: that model is a *cancer* classifier, so its probability on genuine
# lesion photographs (median 0.074) is LOWER than on ordinary photographs
# (median 0.123) - it points the wrong way. Using it to refuse images in the
# band caused 6 of the 7 genuine-lesion false rejections measured.
#
# The band is therefore resolved by the distance itself, split at its
# midpoint: genuine lesions occupy the lower half of the band and ordinary
# photographs the upper half. DOMAIN_BAND_SPLIT is DERIVED from the two
# thresholds above - no new tuned number is introduced.
DOMAIN_GATE_ENABLED = True
DOMAIN_REFERENCE_PATH = os.path.join(RESULTS_DIR, "domain_reference.npy")
DOMAIN_K = 5                          # k nearest reference embeddings
DOMAIN_MAX_DISTANCE = 0.82            # upper edge of the ambiguous band
DOMAIN_SUSPICIOUS_DISTANCE = 0.70     # lower edge of the ambiguous band

# Derived: the operating point inside the ambiguous band. Everything above it
# is treated as outside the lesion domain.
DOMAIN_BAND_SPLIT = (DOMAIN_SUSPICIOUS_DISTANCE + DOMAIN_MAX_DISTANCE) / 2.0

# ──────────────────────────────────────────────────────────────────────
# Grad-CAM settings
# ──────────────────────────────────────────────────────────────────────
ENABLE_GRADCAM_SINGLE = True        # generate for single-image predictions
ENABLE_GRADCAM_BATCH = False        # skip for batch (performance)

# ──────────────────────────────────────────────────────────────────────
# Medical disclaimer
# ──────────────────────────────────────────────────────────────────────
DISCLAIMER = (
    "This tool is for decision support only and is not a medical diagnosis. "
    "Always consult a qualified dermatologist for any skin concern. "
    "Images are analysed in memory and are never stored or transmitted."
)