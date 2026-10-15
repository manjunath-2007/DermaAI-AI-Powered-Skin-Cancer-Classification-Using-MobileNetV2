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
CONVNEXT_CKPT = os.path.join(RESULTS_DIR, "checkpoints", "stage2_epoch13.pt")
CONVNEXT_BEST = os.path.join(RESULTS_DIR, "convnext_tiny_best.pt")
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