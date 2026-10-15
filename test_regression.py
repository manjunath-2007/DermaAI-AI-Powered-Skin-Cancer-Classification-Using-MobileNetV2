#!/usr/bin/env python3
"""
DermaAI Automated Regression Tests

Tests both ConvNeXt-Tiny and MobileNetV2 models for:
- Single-image prediction
- Batch prediction
- Grad-CAM heatmap generation (validity + localization diagnostics)
- Input validation and error handling
- API endpoint health

Output is plain ASCII ([PASS] / [FAIL] / [ERROR] / [WARN]) so the
script runs correctly on Windows consoles with cp1252/cp850 codepages.

Exit code: 0 if every test passed, 1 if any test failed or errored.
"""

import os
import io
import sys
import base64
import warnings

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.disable(logging.CRITICAL)

# Benign Keras notice when the Grad-CAM grad-model is built from the
# MobileNetV2 graph; the overlay is still generated correctly.
warnings.filterwarnings("ignore", message="The structure of .inputs.")

import numpy as np
from PIL import Image

from app import app

client = app.test_client()

# Test images (external clinical photos - NOT validated ground truth)
TEST_IMAGES = [
    ("test-images/cancer-melanoma.jpg", "melanoma"),
    ("test-images/noncancer-nevus.jpg", "nevus"),
    ("test-images/noncancer-congenital-nevus.jpg", "congenital_nevus"),
    ("test-images/noncancer-seborrheic-keratosis.jpg", "seborrheic_keratosis"),
]

# ----------------------------------------------------------------------
# Result tracking
# ----------------------------------------------------------------------
RESULTS = []          # (model, test_name, status, detail)
CURRENT_MODEL = "unknown"


def record(status, name, detail=""):
    RESULTS.append((CURRENT_MODEL, name, status, detail))
    tag = {"PASS": "[PASS]", "FAIL": "[FAIL]", "ERROR": "[ERROR]", "WARN": "[WARN]"}[status]
    line = "  %-8s %s" % (tag, name)
    if detail:
        line += " -- " + detail
    print(line, flush=True)


def run_test(name, fn, *args):
    """Run one test, record PASS/FAIL/ERROR, never abort the whole suite."""
    try:
        extra = fn(*args)
        record("PASS", name, extra or "")
        return True
    except AssertionError as exc:
        record("FAIL", name, str(exc) or "assertion failed")
        return False
    except Exception as exc:
        record("ERROR", name, "%s: %s" % (type(exc).__name__, exc))
        return False


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def load_image(path):
    with open(path, "rb") as f:
        return f.read()


def test_health(model_name):
    """Test /health endpoint."""
    resp = client.get('/health')
    assert resp.status_code == 200, "status code %s (expected 200)" % resp.status_code
    data = resp.get_json()
    assert data['status'] == 'ok', "status not ok: %s" % data
    assert data['model_loaded'] is True, "model not loaded: %s" % data
    assert data['active_model'] == model_name, \
        "wrong active model: %s (expected %s)" % (data['active_model'], model_name)
    return "model=%s loaded=True" % model_name


def validate_heatmap_png(heatmap_b64):
    """Decode a base64 heatmap, verify it is a valid PNG, return (w, h)."""
    import cv2
    raw = base64.b64decode(heatmap_b64)
    arr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None, "heatmap is not a decodable image"
    h, w = arr.shape[:2]
    assert w > 0 and h > 0, "heatmap has zero dimension"
    return w, h


def test_single_prediction(image_data, image_name):
    """Test /predict endpoint."""
    resp = client.post(
        '/predict',
        data={'file': (io.BytesIO(image_data), image_name)},
        content_type='multipart/form-data'
    )
    assert resp.status_code == 200, "status code %s (expected 200)" % resp.status_code
    data = resp.get_json()
    assert data['success'] is True, "success flag false: %s" % data
    assert data['verdict'] in ('affected', 'not_affected'), "invalid verdict: %s" % data['verdict']
    assert 'cancer_probability' in data, "missing cancer_probability"
    assert 0.0 <= data['cancer_probability'] <= 1.0, "probability out of range: %s" % data['cancer_probability']
    assert 'risk_level' in data and data['risk_level'] in ('low', 'medium', 'high'), \
        "invalid risk_level: %s" % data.get('risk_level')
    assert 'confidence' in data, "missing confidence"

    # Grad-CAM heatmap must be present and a decodable PNG
    assert 'heatmap' in data and data['heatmap'], "missing/empty heatmap"
    assert len(data['heatmap']) > 100, "heatmap too small (likely empty)"
    dims = validate_heatmap_png(data['heatmap'])
    return "verdict=%s prob=%.4f heatmap=%dx%d" % (
        data['verdict'], data['cancer_probability'], dims[0], dims[1])


def test_heatmap_alignment(image_data, image_name):
    """
    Verify the heatmap is dimensionally aligned with the model input AND
    actually registered to the image (not shifted/flipped), then analyse
    WHERE the model attends.

    WARNING: matching dimensions (224x224) only prove the overlay has the
    same size as the resized preview. They do NOT prove the overlay lines
    up with the image content or that the model attends to the lesion.
    We therefore also (a) check pixel registration of the blended overlay
    against the resized original and (b) compute the activation centroid
    and corner concentration as localization diagnostics (see README -
    suspected corner-activation limitation).
    """
    resp = client.post(
        '/predict',
        data={'file': (io.BytesIO(image_data), image_name)},
        content_type='multipart/form-data'
    )
    assert resp.status_code == 200, "status code %s" % resp.status_code
    data = resp.get_json()
    assert data['success'] is True, "success flag false"
    assert data.get('heatmap'), "empty heatmap"

    import cv2
    raw = base64.b64decode(data['heatmap'])
    hm = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = hm.shape[:2]

    # 1) Dimensional alignment with the model input (224x224) and the
    #    frontend preview canvas (index.html draws the preview at 224x224).
    assert (w, h) == (224, 224), \
        "heatmap %dx%d does not match model input 224x224" % (w, h)

    # 2) Pixel registration: the overlay is 0.55*orig + 0.45*JET(cam), so
    #    it must correlate better with the resized original at zero shift
    #    than with a shifted copy. This proves the heatmap is overlaid on
    #    the correctly aligned image (no flip/offset), not merely that the
    #    dimensions match.
    orig = np.array(Image.open(io.BytesIO(image_data)).convert("RGB").resize((224, 224)))
    orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR).astype(np.float64)
    hm_f = hm.astype(np.float64)
    diff_same = float(np.abs(hm_f - orig_bgr).mean())
    shifted = np.roll(np.roll(orig_bgr, 56, axis=0), 56, axis=1)
    diff_shifted = float(np.abs(hm_f - shifted).mean())
    assert diff_same < diff_shifted, \
        "heatmap not registered to image (zero-shift diff %.2f >= shifted diff %.2f)" % (
            diff_same, diff_shifted)

    # 3) Localization diagnostics: estimate the "heat" in the overlay.
    #    The overlay is 0.55*original + 0.45*JET(cam). JET is blue where
    #    CAM is low and red/yellow where CAM is high, so (R - B) is a
    #    usable heat proxy: positive when hot colours dominate.
    b, g, r = hm[:, :, 0].astype(np.float64), hm[:, :, 1].astype(np.float64), hm[:, :, 2].astype(np.float64)
    heat = np.maximum(r - b, 0.0)
    flat = heat.flatten()
    if flat.max() <= 0:
        return "aligned 224x224; registered; activation empty"
    thr = np.percentile(flat, 90)
    hot = heat >= thr
    ys, xs = np.nonzero(hot)
    cy, cx = ys.mean() / h, xs.mean() / w  # normalized centroid in [0,1]
    center_dist = float(np.hypot(cx - 0.5, cy - 0.5))  # 0=center, ~0.707=corner

    # Corner share: outer 12.5% frame (uniform attention would give ~0.44).
    margin = 0.125
    m_h, m_w = int(h * margin), int(w * margin)
    frame = np.zeros((h, w), dtype=bool)
    frame[:m_h, :] = True
    frame[h - m_h:, :] = True
    frame[:, :m_w] = True
    frame[:, w - m_w:] = True
    corner_share = float((hot & frame).sum()) / float(max(hot.sum(), 1))

    detail = ("aligned 224x224; registered (diff %.2f < %.2f); "
              "hot-pixel centroid=(%.2f,%.2f) center_dist=%.2f; "
              "corner-frame share=%.2f" % (
                  diff_same, diff_shifted, cx, cy, center_dist, corner_share))
    # Strong evidence the model attends to borders/corners rather than the
    # lesion: hot pixels pushed toward a corner AND over-represented in the
    # outer frame (uniform baseline ~0.44). Non-fatal diagnostic warning.
    if center_dist > 0.30 and corner_share > 0.50:
        record("WARN", "heatmap_localization(%s)" % image_name,
               "activation concentrated near image corners/borders - "
               "localization may be spurious (see README limitation)")
    return detail


def test_batch_prediction(image_datas, image_names):
    """Test /predict_batch endpoint."""
    files = [(io.BytesIO(data), name) for data, name in zip(image_datas, image_names)]
    resp = client.post('/predict_batch', data={'files': files}, content_type='multipart/form-data')
    assert resp.status_code == 200, "status code %s (expected 200)" % resp.status_code
    data = resp.get_json()
    assert data['success'] is True, "batch success false: %s" % data
    assert data['count'] == len(image_names), \
        "count mismatch: %s vs %s" % (data['count'], len(image_names))
    assert len(data['predictions']) == len(image_names), "predictions length mismatch"
    for i, pred in enumerate(data['predictions']):
        assert pred['verdict'] in ('affected', 'not_affected'), \
            "invalid verdict in batch[%d]: %s" % (i, pred['verdict'])
        assert 'cancer_probability' in pred, "missing prob in batch[%d]" % i
        assert 0.0 <= pred['cancer_probability'] <= 1.0, "prob out of range in batch[%d]" % i
    return "%d images, all verdicts valid" % len(image_names)


def test_error_handling():
    """Test error handling for invalid inputs."""
    # No file
    resp = client.post('/predict', data={}, content_type='multipart/form-data')
    assert resp.status_code == 400, "no file should be 400, got %s" % resp.status_code
    assert 'error' in resp.get_json(), "missing error message"

    # Invalid file type
    resp = client.post('/predict', data={'file': (io.BytesIO(b'notanimage'), 'test.txt')},
                       content_type='multipart/form-data')
    assert resp.status_code == 400, "invalid type should be 400, got %s" % resp.status_code

    # Corrupted image
    resp = client.post('/predict', data={'file': (io.BytesIO(b'corrupted'), 'test.jpg')},
                       content_type='multipart/form-data')
    assert resp.status_code == 400, "corrupted should be 400, got %s" % resp.status_code

    # File too large
    large = b'x' * (11 * 1024 * 1024)
    resp = client.post('/predict', data={'file': (io.BytesIO(large), 'large.jpg')},
                       content_type='multipart/form-data')
    assert resp.status_code == 413, "too large should be 413, got %s" % resp.status_code

    # Batch: too many files
    batch_files = [(io.BytesIO(b'fake'), 'test%d.jpg' % i) for i in range(65)]
    resp = client.post('/predict_batch', data={'files': batch_files},
                       content_type='multipart/form-data')
    assert resp.status_code == 400, "too many batch should be 400, got %s" % resp.status_code

    # Batch: total size too large
    large = io.BytesIO(b'x' * (60 * 1024 * 1024))
    resp = client.post('/predict_batch', data={'files': [(large, 'large.jpg')]},
                       content_type='multipart/form-data')
    assert resp.status_code == 413, "batch too large should be 413, got %s" % resp.status_code

    return "all 6 validation cases return correct status codes"


def run_all_tests(model_name):
    """Run complete test suite for a model."""
    global CURRENT_MODEL
    CURRENT_MODEL = model_name
    print("\n" + "=" * 72)
    print("Testing model: %s" % model_name)
    print("=" * 72, flush=True)

    run_test("health_endpoint", test_health, model_name)

    # Load all test images
    image_datas = [load_image(path) for path, _ in TEST_IMAGES]

    # Single predictions (includes Grad-CAM validity check).
    # Upload under the REAL filename (with extension) - the server rejects
    # files without a known image extension, so the label alone is not enough.
    for (path, label), data in zip(TEST_IMAGES, image_datas):
        upload_name = os.path.basename(path)
        run_test("single_prediction(%s)" % label,
                 test_single_prediction, data, upload_name)
        run_test("heatmap_alignment(%s)" % label,
                 test_heatmap_alignment, data, upload_name)

    # Batch prediction
    upload_names = [os.path.basename(path) for path, _ in TEST_IMAGES]
    run_test("batch_prediction", test_batch_prediction, image_datas, upload_names)

    # Error handling
    run_test("error_handling", test_error_handling)


def main():
    print("DermaAI Automated Regression Tests")
    print("=" * 72)
    print("Interpreter: %s" % sys.executable)

    # Test ConvNeXt-Tiny (default)
    import config
    config.DEFAULT_MODEL = 'convnext_tiny'
    run_all_tests('convnext_tiny')

    # Test MobileNetV2: flip the shared config attribute, then reload the
    # app module so its module-level active_model / startup load pick up
    # the new selection. (inference.py is intentionally NOT reloaded: it
    # caches the loaded ConvNeXt model and holds its own config reference.)
    import importlib
    import app as app_module
    config.DEFAULT_MODEL = 'mobilenetv2'
    importlib.reload(app_module)
    global client
    client = app_module.app.test_client()
    run_all_tests('mobilenetv2')

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    passed = [r for r in RESULTS if r[2] == "PASS"]
    failed = [r for r in RESULTS if r[2] == "FAIL"]
    errored = [r for r in RESULTS if r[2] == "ERROR"]
    warned = [r for r in RESULTS if r[2] == "WARN"]

    for model in ("convnext_tiny", "mobilenetv2"):
        m_all = [r for r in RESULTS if r[0] == model]
        m_pass = [r for r in m_all if r[2] == "PASS"]
        m_fail = [r for r in m_all if r[2] in ("FAIL", "ERROR")]
        print("%-14s %2d passed, %2d failed/errored" % (
            model, len(m_pass), len(m_fail)))

    if failed or errored:
        print("\nFailed / errored tests:")
        for model, name, status, detail in failed + errored:
            print("  [%s] %s :: %s -- %s" % (status, model, name, detail))
    if warned:
        print("\nWarnings (non-fatal diagnostics):")
        for model, name, status, detail in warned:
            print("  [WARN] %s :: %s -- %s" % (model, name, detail))

    print("\nTOTAL: %d passed, %d failed, %d errored, %d warnings" % (
        len(passed), len(failed), len(errored), len(warned)))

    if failed or errored:
        print("REGRESSION SUITE: FAILED")
        return 1
    print("REGRESSION SUITE: ALL TESTS PASSED")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print("\n[ERROR] test harness error: %s: %s" % (type(exc).__name__, exc))
        import traceback
        traceback.print_exc()
        sys.exit(1)
