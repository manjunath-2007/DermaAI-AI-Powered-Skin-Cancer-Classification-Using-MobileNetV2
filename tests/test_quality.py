#!/usr/bin/env python3
"""Image-quality warning tests.

Covers the lightweight quality screening added to the DermaAI backend:

- ``quality.check_image_quality`` unit behaviour (blurry / dark /
  overexposed / acceptable images, metric structure, non-rejection)
- ``/predict`` and ``/predict_batch`` API response compatibility
  (existing fields preserved, new ``quality_*`` fields present)

Run with either::

    python tests/test_quality.py          # standalone runner
    pytest tests/test_quality.py -v       # pytest runner

The API tests import ``app`` (which loads the active model once); they are
skipped automatically if the application cannot be imported.
"""

import io
import os
import sys

# Add parent directory to sys.path so 'quality'/'app' can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
logging.disable(logging.CRITICAL)

import numpy as np
import pytest
from PIL import Image, ImageFilter

import config
import quality


# ----------------------------------------------------------------------
# Synthetic test images
# ----------------------------------------------------------------------
def _noisy_image(size=512, low=0, high=255, seed=1234):
    """Deterministic uniform-noise image (high Laplacian variance = sharp)."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(low, high + 1, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _blurred_image(size=512, radius=12, seed=1234):
    """Strongly defocused version of a sharp image (low Laplacian variance)."""
    return _noisy_image(size=size, seed=seed).filter(ImageFilter.GaussianBlur(radius=radius))


def _dark_image(size=512):
    """Underexposed image: textured (not flat) but very low mean brightness."""
    img = _noisy_image(size=size, seed=7)
    arr = (np.asarray(img, dtype=np.float32) * 0.10).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _overexposed_image(size=512):
    """Blown-out highlights: almost every pixel is near white."""
    rng = np.random.default_rng(11)
    arr = rng.integers(248, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _warning_types(result):
    """Accept both the unit-test shape (``warnings``) and the API shape
    (``quality_warnings``)."""
    warnings = result.get("warnings", result.get("quality_warnings", []))
    return {w["type"] for w in warnings}


def _assert_metrics(result):
    assert isinstance(result["metrics"], dict)
    assert set(result["metrics"]) == {"blur_score", "mean_brightness", "overexposure_ratio"}
    assert isinstance(result["metrics"]["blur_score"], float)
    assert 0.0 <= result["metrics"]["mean_brightness"] <= 1.0
    assert 0.0 <= result["metrics"]["overexposure_ratio"] <= 1.0


# ----------------------------------------------------------------------
# Unit tests: quality.check_image_quality
# ----------------------------------------------------------------------
def test_acceptable_image_has_no_warnings():
    """A sharp, well-exposed photo must not raise any warning."""
    result = quality.check_image_quality(_noisy_image())
    assert result["warnings"] == []
    assert result["is_valid"] is True
    _assert_metrics(result)


def test_blurry_image_triggers_blur_warning():
    """A defocused image must produce exactly the blur warning."""
    result = quality.check_image_quality(_blurred_image())
    assert "blur" in _warning_types(result)
    messages = [w["message"] for w in result["warnings"] if w["type"] == "blur"]
    assert messages == ["Image appears blurry. Try taking a clearer photo."]
    assert result["is_valid"] is True          # warning must never reject
    _assert_metrics(result)


def test_dark_image_triggers_dark_warning():
    """An underexposed image must produce the dark warning."""
    result = quality.check_image_quality(_dark_image())
    assert "dark" in _warning_types(result)
    assert result["metrics"]["mean_brightness"] < config.QUALITY_BRIGHTNESS_DARK
    assert result["is_valid"] is True
    _assert_metrics(result)


def test_overexposed_image_triggers_bright_warning():
    """A blown-out image must produce the overexposure warning."""
    result = quality.check_image_quality(_overexposed_image())
    assert "bright" in _warning_types(result)
    assert result["metrics"]["overexposure_ratio"] > config.QUALITY_OVEREXPOSURE_RATIO
    assert result["is_valid"] is True
    _assert_metrics(result)


def test_overexposure_warning_not_duplicated():
    """Brightness and near-white checks share one warning, never two."""
    result = quality.check_image_quality(_overexposed_image())
    assert [w["type"] for w in result["warnings"]].count("bright") == 1


def test_warnings_never_reject_decodable_image():
    """Even a bad image returns is_valid=True and full metrics."""
    for img in (_blurred_image(), _dark_image(), _overexposed_image()):
        result = quality.check_image_quality(img)
        assert result["is_valid"] is True
        assert len(result["warnings"]) >= 1
        _assert_metrics(result)


def test_metrics_consistent_across_resolutions():
    """The same photo at a different resolution is classified the same way."""
    # Sharp noise stays warning-free at both resolutions
    for size in (256, 1024):
        assert quality.check_image_quality(_noisy_image(size=size, seed=42))["warnings"] == []

    # Mean brightness is resolution independent
    dark_small = quality.check_image_quality(_dark_image(size=256))
    dark_large = quality.check_image_quality(_dark_image(size=1024))
    assert abs(dark_small["metrics"]["mean_brightness"] -
               dark_large["metrics"]["mean_brightness"]) < 0.02
    # ...and the dark warning fires at both resolutions
    assert "dark" in _warning_types(dark_small)
    assert "dark" in _warning_types(dark_large)


def test_grayscale_and_rgba_modes_supported():
    """Non-RGB PIL modes are converted internally, no exception raised."""
    gray = _noisy_image(seed=5).convert("L")
    rgba = _noisy_image(seed=5).convert("RGBA")
    for img in (gray, rgba):
        result = quality.check_image_quality(img)
        assert result["is_valid"] is True
        _assert_metrics(result)


def test_disabled_quality_check_returns_empty_warnings(monkeypatch):
    """With QUALITY_ENABLED=False the check is a no-op but still valid."""
    monkeypatch.setattr(config, "QUALITY_ENABLED", False)
    result = quality.check_image_quality(_blurred_image())
    assert result == {"warnings": [], "is_valid": True, "metrics": {}}


def test_configurable_thresholds_are_respected(monkeypatch):
    """Thresholds are configurable: lowering them silences the blur warning,
    and raising them makes even a sharp photo warn."""
    blurred = _blurred_image()
    sharp = _noisy_image()

    blurred_score = quality.check_image_quality(blurred)["metrics"]["blur_score"]
    sharp_score = quality.check_image_quality(sharp)["metrics"]["blur_score"]

    # Lower the threshold below the blurred score -> warning disappears
    monkeypatch.setattr(config, "QUALITY_BLUR_THRESHOLD", blurred_score - 0.5)
    assert "blur" not in _warning_types(quality.check_image_quality(blurred))

    # Raise the threshold above the sharp score -> sharp photo now warns
    monkeypatch.setattr(config, "QUALITY_BLUR_THRESHOLD", sharp_score + 1.0)
    assert "blur" in _warning_types(quality.check_image_quality(sharp))


# ----------------------------------------------------------------------
# Real-image blur diagnostics (documents the actual failure mode)
# ----------------------------------------------------------------------
# The synthetic unit tests above pass because uniform noise scores 800-48000.
# Real skin photos score far lower (see diagnosis), so the configured
# QUALITY_BLUR_THRESHOLD (100.0) can flag smooth legitimate photos as blurry.
# These tests record real-image scores as a regression baseline and verify the
# blur detector still responds *relatively* (intentional blur lowers the score)
# and tracks the configured threshold at the boundary.
REAL_IMAGE_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "test-images", name)
    for name in ("cancer-melanoma.jpg", "noncancer-nevus.jpg",
                 "noncancer-congenital-nevus.jpg", "noncancer-seborrheic-keratosis.jpg")
]


def test_real_image_blur_scores_are_finite():
    """Every real clinical photo yields a finite, non-negative blur score.

    This records reality: several real photos score below QUALITY_BLUR_THRESHOLD
    (100.0) and therefore receive a blur warning even though they may not be
    genuinely blurry. The test does not assert a classification because
    ground-truth blur labels are unavailable for these external images.
    """
    for path in REAL_IMAGE_PATHS:
        img = Image.open(path)
        result = quality.check_image_quality(img)
        score = result["metrics"]["blur_score"]
        assert isinstance(score, float)
        assert score >= 0.0, "%s: negative blur score %s" % (os.path.basename(path), score)
        assert not (np.isnan(score) or np.isinf(score)), "%s: non-finite blur score" % os.path.basename(path)


def test_intentional_blur_lowers_score_on_real_image():
    """Blurring a real photo must strictly lower its Laplacian-variance score."""
    for path in REAL_IMAGE_PATHS:
        sharp = Image.open(path)
        blurred = sharp.filter(ImageFilter.GaussianBlur(radius=8))
        sharp_score = quality.check_image_quality(sharp)["metrics"]["blur_score"]
        blurred_score = quality.check_image_quality(blurred)["metrics"]["blur_score"]
        assert blurred_score < sharp_score, (
            "%s: intentional blur did not reduce score (sharp=%.2f blurred=%.2f)"
            % (os.path.basename(path), sharp_score, blurred_score))


def test_blur_warning_tracks_threshold_at_boundary(monkeypatch):
    """The blur warning fires iff the score is below the configured threshold."""
    img = Image.open(REAL_IMAGE_PATHS[0])  # melanoma: score well above 100
    score = quality.check_image_quality(img)["metrics"]["blur_score"]

    monkeypatch.setattr(config, "QUALITY_BLUR_THRESHOLD", score + 1.0)
    assert "blur" in _warning_types(quality.check_image_quality(img)), \
        "score below raised threshold should warn"

    monkeypatch.setattr(config, "QUALITY_BLUR_THRESHOLD", score - 1.0)
    assert "blur" not in _warning_types(quality.check_image_quality(img)), \
        "score above lowered threshold should not warn"


# ----------------------------------------------------------------------
# API tests: /predict and /predict_batch
# ----------------------------------------------------------------------
try:
    from app import app as flask_app
    _API_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    flask_app = None
    _API_AVAILABLE = False
    _API_IMPORT_ERROR = str(exc)

pytestmark = pytest.mark.skipif(
    not _API_AVAILABLE,
    reason="Flask app could not be imported: %s" % (_API_IMPORT_ERROR if not _API_AVAILABLE else ""),
)


def _post_image(endpoint, img, field="file"):
    payload = {field: (io.BytesIO(_png_bytes(img)), "test.png")}
    return flask_app.test_client().post(endpoint, data=payload, content_type="multipart/form-data")


def _png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    return flask_app.test_client()


def test_predict_returns_quality_fields_and_existing_schema(client):
    """Successful response keeps every existing field and adds quality_*."""
    resp = client.post(
        "/predict",
        data={"file": (io.BytesIO(_png_bytes(_noisy_image())), "ok.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True

    # Existing API fields (unchanged contract)
    for field in ("cancer_probability", "verdict", "cancer_detected",
                  "confidence", "risk_level", "heatmap", "prediction"):
        assert field in data, "missing existing field: %s" % field
    assert data["verdict"] in ("affected", "not_affected")
    assert data["risk_level"] in ("high", "medium", "low")

    # New quality fields
    assert data["quality_warnings"] == []
    assert data["quality_is_valid"] is True
    assert isinstance(data["quality_metrics"], dict)


def test_predict_blurry_image_still_predicts_with_warning(client):
    """A blurry image is warned about but still returns a prediction."""
    resp = _post_image("/predict", _blurred_image())
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert "blur" in _warning_types(data)
    assert "cancer_probability" in data
    assert data["quality_is_valid"] is True


def test_predict_dark_and_overexposed_images_still_predict(client):
    """Dark / overexposed images are warned about, never rejected by the quality layer.

    The quality screen never rejects a decodable image. A separate input-domain
    gate may still refuse an image that is not a skin photograph at all (a
    uniform noise texture is not a lesion photo), and then it does so with
    "not_analysable" rather than with a quality error.
    """
    for img in (_dark_image(), _overexposed_image()):
        resp = _post_image("/predict", img)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["quality_warnings"]) >= 1
        assert data["quality_is_valid"] is True
        if data.get("analysable") is False:
            # Refused by the input-domain gate, not by the quality screen.
            assert data["verdict"] == "not_analysable"
            assert data["domain_in_domain"] is False
            assert data["cancer_probability"] is None
        else:
            assert data["verdict"] in ("affected", "not_affected")


def test_predict_invalid_image_still_rejected(client):
    """Un-decodable uploads keep the existing 400 error contract."""
    resp = client.post(
        "/predict",
        data={"file": (io.BytesIO(b"not an image"), "broken.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "error" in data


def test_predict_batch_returns_quality_per_image(client):
    """Batch results carry per-image quality info and keep the schema."""
    good = _noisy_image(seed=1)
    dark = _dark_image()
    resp = client.post(
        "/predict_batch",
        data={
            "files": [
                (io.BytesIO(_png_bytes(good)), "good.png"),
                (io.BytesIO(_png_bytes(dark)), "dark.png"),
            ]
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["count"] == 2
    assert len(data["predictions"]) == 2

    first, second = data["predictions"]
    for pred in (first, second):
        for field in ("cancer_probability", "verdict", "cancer_detected",
                      "confidence", "risk_level"):
            assert field in pred, "missing existing field: %s" % field
        assert "quality_warnings" in pred
        assert "quality_metrics" in pred
        assert pred["quality_is_valid"] is True

    # Order is preserved: the sharp image is clean, the dark one is warned.
    assert first["quality_warnings"] == []
    assert "dark" in _warning_types(second)


# ----------------------------------------------------------------------
# Standalone runner (no pytest required)
# ----------------------------------------------------------------------
def _run_standalone():
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    passed = failed = 0
    for name, fn in tests:
        if "client" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
            print("[SKIP] %s (pytest-only fixture)" % name)
            continue
        try:
            if "monkeypatch" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                class _MP:
                    def __init__(self): self._undo = []
                    def setattr(self, obj, attr, value):
                        self._undo.append((obj, attr, getattr(obj, attr)))
                        setattr(obj, attr, value)
                    def undo(self):
                        for obj, attr, old in reversed(self._undo):
                            setattr(obj, attr, old)
                mp = _MP()
                try:
                    fn(mp)
                finally:
                    mp.undo()
            else:
                fn()
            print("[PASS] %s" % name)
            passed += 1
        except AssertionError as exc:
            print("[FAIL] %s -- %s" % (name, exc))
            failed += 1
        except Exception as exc:
            print("[ERROR] %s -- %s: %s" % (name, type(exc).__name__, exc))
            failed += 1
    print("\n%d passed, %d failed (unit tests only; run pytest for API tests)" % (passed, failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_standalone())
