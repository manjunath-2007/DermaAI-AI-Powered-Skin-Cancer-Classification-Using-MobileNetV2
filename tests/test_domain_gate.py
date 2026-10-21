#!/usr/bin/env python3
"""Input-domain gate tests.

Covers the safety layer added to stop the DermaAI backend from reporting an
ordinary photograph (a classroom scene, a car, fruit in a bowl) as a skin
lesion:

- the reference artifact is present and well formed
- genuine lesion photographs stay inside the domain and keep their verdict
- the decision rules behave as calibrated
- out-of-domain images are refused and carry no probability at all
- the gate never changes the prediction for an in-domain image

Run with either::

    python tests/test_domain_gate.py          # standalone runner
    pytest tests/test_domain_gate.py -v       # pytest runner
"""

import io
import os
import sys

# Add parent directory to sys.path so 'domain'/'app' can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
logging.disable(logging.CRITICAL)

import numpy as np
import pytest
from PIL import Image

import config
import domain

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST_IMAGES_DIR = os.path.join(APP_DIR, "test-images")
LESION_IMAGES = []
if os.path.isdir(_TEST_IMAGES_DIR):
    LESION_IMAGES = sorted(
        os.path.join(_TEST_IMAGES_DIR, name)
        for name in os.listdir(_TEST_IMAGES_DIR)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )


# ----------------------------------------------------------------------
# Synthetic test images
# ----------------------------------------------------------------------
def _scene_image(seed=0, width=640, height=480):
    """Deterministic synthetic scene: saturated colour blocks on a flat field.

    Stands in for an ordinary photograph. It lands in the ambiguous band of the
    gate (its distance to the lesion manifold is around 0.80), so it is only
    refused when the independent second opinion also reports "not a lesion" -
    which is exactly the rule the calibrated thresholds encode.
    """
    rng = np.random.default_rng(seed)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = 40
    arr[:, :, 1] = 160
    arr[:, :, 2] = 60
    for _ in range(30):
        x = int(rng.integers(0, width - 100))
        y = int(rng.integers(0, height - 80))
        w = int(rng.integers(50, 180))
        h = int(rng.integers(40, 110))
        arr[y:y + h, x:x + w] = rng.integers(0, 256, size=3)
    arr = (arr.astype(np.float32) + rng.normal(0, 14, arr.shape)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _open(path):
    img = Image.open(path)
    img.load()
    return img


# ----------------------------------------------------------------------
# Reference artifact
# ----------------------------------------------------------------------
def test_reference_artifact_is_present_and_well_formed():
    assert os.path.exists(config.DOMAIN_REFERENCE_PATH), \
        "domain reference artifact is missing - run scripts/build_domain_reference.py"
    ref = np.load(config.DOMAIN_REFERENCE_PATH)
    assert ref.ndim == 2
    assert ref.shape[1] == 768, "reference must hold 768-d ConvNeXt embeddings"
    assert ref.shape[0] >= 1000, "reference is suspiciously small"
    assert np.isfinite(ref.astype(np.float32)).all(), "reference contains non-finite values"


def test_calibration_record_is_present():
    meta_path = os.path.join(os.path.dirname(config.DOMAIN_REFERENCE_PATH),
                             "domain_reference_meta.json")
    assert os.path.exists(meta_path), "calibration record is missing"


# ----------------------------------------------------------------------
# Decision rules
# ----------------------------------------------------------------------
def test_decision_rules():
    """The decision rule behaves exactly as calibrated.

    The gate is decided by the distance to the HAM10000 manifold alone. The
    MobileNetV2 second opinion is advisory only: measured on 100 real
    photographs its cancer probability is LOWER on genuine lesion photographs
    (median 0.074) than on ordinary photographs (median 0.123), so using it to
    refuse images refused genuine lesions preferentially.
    """
    # Clearly inside the lesion domain.
    assert not domain.is_out_of_domain(0.30)
    assert not domain.is_out_of_domain(config.DOMAIN_SUSPICIOUS_DISTANCE)

    # Clearly outside.
    assert domain.is_out_of_domain(config.DOMAIN_MAX_DISTANCE + 0.01)
    assert domain.is_out_of_domain(0.95)

    # The ambiguous band is split at its midpoint: the lower half is analysed,
    # the upper half is refused. The split is derived from the two existing
    # thresholds, so no new tuned number is introduced.
    split = config.DOMAIN_BAND_SPLIT
    assert config.DOMAIN_SUSPICIOUS_DISTANCE < split < config.DOMAIN_MAX_DISTANCE
    assert not domain.is_out_of_domain(config.DOMAIN_SUSPICIOUS_DISTANCE + 0.001)
    assert not domain.is_out_of_domain(split - 0.001)
    assert domain.is_out_of_domain(split + 0.001)


def test_second_opinion_cannot_change_the_decision():
    """The advisory second opinion can neither refuse nor admit an image.

    A genuine lesion photograph must not be refused because an unrelated
    cancer classifier happened to score it low, and an ordinary photograph
    must not be admitted because that classifier scored it high.
    """
    for score in (0.30, 0.71, 0.755, 0.80, 0.85, 0.95):
        expected = domain.is_out_of_domain(score)
        assert domain.is_out_of_domain(score, None) is expected
        assert domain.is_out_of_domain(score, 0.0) is expected
        assert domain.is_out_of_domain(score, 0.21) is expected
        assert domain.is_out_of_domain(score, 0.99) is expected


def test_band_split_is_derived_from_the_existing_thresholds():
    """The operating point is derived, not a separately tuned constant."""
    assert config.DOMAIN_BAND_SPLIT == pytest.approx(
        (config.DOMAIN_SUSPICIOUS_DISTANCE + config.DOMAIN_MAX_DISTANCE) / 2.0)
    # The calibrated thresholds themselves are unchanged.
    assert config.DOMAIN_SUSPICIOUS_DISTANCE == 0.70
    assert config.DOMAIN_MAX_DISTANCE == 0.82


# ----------------------------------------------------------------------
# In-domain behaviour
# ----------------------------------------------------------------------
@pytest.mark.skipif(not LESION_IMAGES, reason="no test-images/ folder available")
def test_real_lesion_photographs_stay_in_domain():
    """Genuine lesion photographs are never refused."""
    if not domain.is_available():
        pytest.skip("domain reference not available")
    for path in LESION_IMAGES:
        info = domain.check_domain(_open(path))
        assert info["in_domain"], \
            f"{os.path.basename(path)} was wrongly refused (score={info['score']})"


@pytest.mark.skipif(not LESION_IMAGES, reason="no test-images/ folder available")
def test_gate_does_not_change_in_domain_predictions():
    """The gate must not alter the model's output for an in-domain image."""
    import inference
    if not domain.is_available():
        pytest.skip("domain reference not available")

    images = [_open(p) for p in LESION_IMAGES]
    probs, metas = inference.predict_batch(images)
    probs2, metas2, _ = inference.predict_batch_with_embeddings(images)
    assert np.allclose(probs, probs2, atol=1e-6), "embedding forward changed the probabilities"
    assert [m["verdict"] for m in metas] == [m["verdict"] for m in metas2]

    for meta in metas:
        info = domain.check_domain(images[metas.index(meta)])
        if info["in_domain"]:
            assert meta["cancer_probability"] is not None
            assert meta["verdict"] in ("affected", "not_affected")


# ----------------------------------------------------------------------
# Out-of-domain behaviour
# ----------------------------------------------------------------------
def test_out_of_domain_image_is_refused():
    """An ordinary photograph is refused and no probability is produced."""
    if not domain.is_available():
        pytest.skip("domain reference not available")

    import app as app_module
    second_opinion = app_module.mobilenet_cancer_probability
    if app_module.load_mobilenet_model() is None:
        pytest.skip("MobileNetV2 second opinion unavailable")

    info = domain.check_domain(_scene_image(), second_opinion=second_opinion)
    assert not info["in_domain"], \
        f"synthetic scene was accepted (score={info['score']})"

    payload = domain.not_analysable_result(info)
    assert payload["success"] is True
    assert payload["analysable"] is False
    assert payload["verdict"] == "not_analysable"
    assert payload["cancer_probability"] is None
    assert payload["cancer_detected"] is None
    assert payload["confidence"] is None
    assert payload["risk_level"] == "unknown"
    assert payload["heatmap"] == ""


@pytest.mark.skipif(not LESION_IMAGES, reason="no test-images/ folder available")
def test_api_refuses_out_of_domain_image():
    """/predict must not report a cancer verdict for an ordinary photograph."""
    try:
        import app as app_module
    except Exception as exc:
        pytest.skip(f"app could not be imported: {exc}")

    if app_module.load_mobilenet_model() is None:
        pytest.skip("MobileNetV2 second opinion unavailable")
    if not domain.is_available():
        pytest.skip("domain reference not available")

    client = app_module.app.test_client()
    buf = io.BytesIO()
    _scene_image().save(buf, format="JPEG")
    buf.seek(0)

    resp = client.post("/predict", data={"file": (buf, "classroom.jpg")},
                        content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["analysable"] is False
    assert body["verdict"] == "not_analysable"
    assert body["cancer_probability"] is None
    assert body["cancer_detected"] is None
    assert body["heatmap"] == ""
    assert body["domain_in_domain"] is False
    assert body["domain_score"] is not None


@pytest.mark.skipif(not LESION_IMAGES, reason="no test-images/ folder available")
def test_api_keeps_in_domain_images_analysable():
    """/predict still produces a full verdict for a genuine lesion photograph."""
    try:
        import app as app_module
    except Exception as exc:
        pytest.skip(f"app could not be imported: {exc}")

    if not domain.is_available():
        pytest.skip("domain reference not available")

    client = app_module.app.test_client()
    for path in LESION_IMAGES:
        with open(path, "rb") as fh:
            resp = client.post("/predict", data={"file": (fh, os.path.basename(path))},
                                content_type="multipart/form-data")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["success"] is True
        assert body["analysable"] is True, \
            f"{os.path.basename(path)} was refused (score={body.get('domain_score')})"
        assert body["verdict"] in ("affected", "not_affected")
        assert body["cancer_probability"] is not None
        assert body["domain_in_domain"] is True


# ----------------------------------------------------------------------
# Standalone runner
# ----------------------------------------------------------------------
def _run_standalone():
    failures = []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:
            failures.append((name, exc))
            print(f"FAIL  {name}: {exc}")
    print()
    print(f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
