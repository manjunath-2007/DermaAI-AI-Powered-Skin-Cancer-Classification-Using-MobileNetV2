"""Lightweight image-quality screening for DermaAI.

Checks a decoded PIL Image for:
  - Excessive blur (Laplacian variance)
  - Very low brightness (underexposure)
  - Excessive brightness / overexposure (near-white pixel fraction)

Designed to run on the **already-decoded** PIL Image so that image
decoding is never duplicated.  The image is internally downscaled to
``config.QUALITY_ANALYSIS_SIZE`` for fast, scale-invariant computation;
this does **not** touch the model preprocessing pipeline (which operates
on a separate 224×224 tensor).

Warnings never reject an image — predictions still run and a prediction
is always returned when the image is decodable.  Only completely invalid
(un-decodable) images are rejected, via the existing error handling in
``app.py``.
"""

import logging

import cv2
import numpy as np
from PIL import Image

import config

log = logging.getLogger(__name__)

# Pre-computed messages for each quality issue.  These match the
# user-facing strings requested in the feature specification.
_WARNING_MESSAGES = {
    "blur": "Image appears blurry. Try taking a clearer photo.",
    "dark": "Image is too dark. Try better lighting.",
    "bright": "Image may be overexposed. Reduce the lighting.",
}


def _pil_to_gray_array(img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a uint8 grayscale numpy array at the
    analysis resolution.  Uses a single resize + cvtColor — no separate
    decode step."""
    size = config.QUALITY_ANALYSIS_SIZE
    small = img.convert("L").resize(size, Image.BILINEAR)
    return np.asarray(small, dtype=np.uint8)


def check_image_quality(img: Image.Image) -> dict:
    """Analyse image quality and return a structured result.

    Args:
        img: A decoded PIL Image (RGB, L, or P mode — ``convert`` is
            handled internally).

    Returns:
        A dict with the following keys:

        ``warnings``
            List of ``{"type": str, "message": str}`` dicts.  Empty when
            no issues are detected.

        ``is_valid``
            Always ``True`` for a decodable image.  (Complete rejection
            of un-decodable images happens earlier in ``app.py``.)

        ``metrics``
            ``{"blur_score": float, "mean_brightness": float,
            "overexposure_ratio": float}`` — raw diagnostic values.
    """
    warnings = []
    metrics = {}

    if not config.QUALITY_ENABLED:
        return {"warnings": [], "is_valid": True, "metrics": metrics}

    try:
        gray = _pil_to_gray_array(img)
    except Exception as exc:
        log.warning("Quality check failed to process image: %s", exc)
        return {"warnings": [], "is_valid": True, "metrics": {}}

    # --- Blur: variance of Laplacian (lower = blurrier) ---
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(lap.var())
    metrics["blur_score"] = round(blur_score, 2)

    if blur_score < config.QUALITY_BLUR_THRESHOLD:
        warnings.append({
            "type": "blur",
            "message": _WARNING_MESSAGES["blur"],
        })

    # --- Brightness: mean of grayscale, normalised to [0, 1] ---
    mean_brightness = float(gray.mean()) / 255.0
    metrics["mean_brightness"] = round(mean_brightness, 4)

    if mean_brightness < config.QUALITY_BRIGHTNESS_DARK:
        warnings.append({
            "type": "dark",
            "message": _WARNING_MESSAGES["dark"],
        })
    elif mean_brightness > config.QUALITY_BRIGHTNESS_BRIGHT:
        warnings.append({
            "type": "bright",
            "message": _WARNING_MESSAGES["bright"],
        })

    # --- Overexposure: fraction of near-white pixels ---
    overexposure_ratio = float((gray > 240).sum()) / gray.size
    metrics["overexposure_ratio"] = round(overexposure_ratio, 4)

    if overexposure_ratio > config.QUALITY_OVEREXPOSURE_RATIO:
        # Only add an overexposure warning if not already covered by the
        # brightness check, to avoid duplicate messages.
        already_warned = any(w["type"] == "bright" for w in warnings)
        if not already_warned:
            warnings.append({
                "type": "bright",
                "message": _WARNING_MESSAGES["bright"],
            })

    log.info("Quality check: blur=%.1f brightness=%.3f overexp=%.3f warnings=%d",
             blur_score, mean_brightness, overexposure_ratio, len(warnings))

    return {
        "warnings": warnings,
        "is_valid": True,
        "metrics": metrics,
    }
