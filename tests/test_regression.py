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

When run under pytest, tests use fixtures for model selection and
image data.  When run standalone (``python tests/test_regression.py``),
the ``main()`` function exercises both models end to end.
"""

import os
import io
import sys
import base64
import warnings

# Add parent directory to sys.path so 'app' can be imported without manual PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.disable(logging.CRITICAL)

# Benign Keras notice when the Grad-CAM grad-model is built from the
# MobileNetV2 graph; the overlay is still generated correctly.
warnings.filterwarnings("ignore", message="The structure of .inputs.")

import numpy as np
from PIL import Image
import pytest

from app import app

# Default test client (ConvNeXt-Tiny, the default model).
client = app.test_client()

# Test images (external clinical photos - NOT validated ground truth)
TEST_IMAGES = [
    ("test-images/cancer-melanoma.jpg", "melanoma"),
    ("test-images/noncancer-nevus.jpg", "nevus"),
    ("test-images/noncancer-congenital-nevus.jpg", "congenital_nevus"),
    ("test-images/noncancer-seborrheic-keratosis.jpg", "seborrheic_keratosis"),
]

# ----------------------------------------------------------------------
# Result tracking (standalone runner)
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
# Helpers
# ----------------------------------------------------------------------
def load_image(path):
    with open(path, "rb") as f:
        return f.read()


def validate_heatmap_png(heatmap_b64):
    """Decode a base64 heatmap, verify it is a valid PNG, return (w, h)."""
    import cv2
    raw = base64.b64decode(heatmap_b64)
    arr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None, "heatmap is not a decodable image"
    h, w = arr.shape[:2]
    assert w > 0 and h > 0, "heatmap has zero dimension"
    return w, h


# ----------------------------------------------------------------------
# Pytest fixtures
# ----------------------------------------------------------------------
@pytest.fixture(params=["convnext_tiny", "mobilenetv2"])
def model_name(request):
    """Parameterize over both supported models."""
    return request.param


@pytest.fixture
def app_client(model_name):
    """Create a Flask test client for the requested model.

    Reloads the app module when the model changes so that
    ``app.active_model`` and the lazy model loader pick up the
    requested backend.  ``inference.py`` is intentionally not reloaded
    (it caches the ConvNeXt model); only ``app.py`` is reloaded to flip
    the active model selection.
    """
    import config
    import importlib
    import app as app_module

    if app_module.active_model != model_name:
        config.DEFAULT_MODEL = model_name
        importlib.reload(app_module)

    return app_module.app.test_client()


@pytest.fixture(params=TEST_IMAGES, ids=[label for _, label in TEST_IMAGES])
def image_pair(request):
    """Yield (image_bytes, filename) for one test image."""
    path, label = request.param
    return load_image(path), os.path.basename(path)


@pytest.fixture
def all_images():
    """Yield (image_datas, image_names) for all test images."""
    datas = [load_image(path) for path, _ in TEST_IMAGES]
    names = [os.path.basename(path) for path, _ in TEST_IMAGES]
    return datas, names


# ----------------------------------------------------------------------
# Pytest tests
# ----------------------------------------------------------------------
def test_health(app_client, model_name):
    """Test /health endpoint."""
    resp = app_client.get('/health')
    assert resp.status_code == 200, "status code %s (expected 200)" % resp.status_code
    data = resp.get_json()
    assert data['status'] == 'ok', "status not ok: %s" % data
    assert data['model_loaded'] is True, "model not loaded: %s" % data
    assert data['active_model'] == model_name, \
        "wrong active model: %s (expected %s)" % (data['active_model'], model_name)


def test_single_prediction(app_client, image_pair):
    """Test /predict endpoint."""
    image_data, image_name = image_pair
    resp = app_client.post(
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
    validate_heatmap_png(data['heatmap'])


def test_heatmap_alignment(app_client, image_pair):
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
    image_data, image_name = image_pair
    resp = app_client.post(
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
        return
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
        warnings.warn(
            "heatmap_localization(%s): activation concentrated near image "
            "corners/borders - localization may be spurious "
            "(see README limitation)" % image_name,
            stacklevel=2,
        )


def test_gradcam_heatmap_finite_and_non_degenerate(app_client, model_name, image_pair):
    """The returned heatmap is a valid, finite, non-degenerate PNG."""
    image_data, image_name = image_pair
    resp = app_client.post(
        '/predict',
        data={'file': (io.BytesIO(image_data), image_name)},
        content_type='multipart/form-data'
    )
    assert resp.status_code == 200, "status code %s" % resp.status_code
    data = resp.get_json()
    assert data['success'] is True, "success flag false"
    assert data.get('heatmap'), "missing/empty heatmap"

    import cv2, base64
    raw = base64.b64decode(data['heatmap'])
    arr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None, "heatmap is not a decodable image"
    h, w = arr.shape[:2]
    assert h > 0 and w > 0, "heatmap has zero dimension"
    assert np.isfinite(arr).all(), "heatmap contains non-finite values"
    assert float(arr.std()) > 0.0, "heatmap is degenerate (constant)"


def test_gradcam_gradients_respond_to_target_class(image_pair):
    """generate_gradcam('affected') vs ('not_affected') must produce different
    heatmaps, confirming gradients flow for the intended prediction target
    (ConvNeXt-Tiny, the primary model)."""
    image_data, image_name = image_pair
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    import inference, cv2, base64
    inference.load_convnext_model()
    hm_a = inference.generate_gradcam(img, target_class="affected")
    hm_n = inference.generate_gradcam(img, target_class="not_affected")
    assert hm_a and hm_n, "heatmap missing for one target class"

    def heat_proxy(b64):
        raw = base64.b64decode(b64)
        arr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return np.maximum(arr[:, :, 2].astype(float) - arr[:, :, 0].astype(float), 0.0)

    pa, pn = heat_proxy(hm_a), heat_proxy(hm_n)
    assert not np.allclose(pa, pn, atol=1.0), \
        "affected and not_affected heatmaps identical: gradients not target-specific"


def test_gradcam_activation_gradient_shapes_compatible(image_pair):
    """Activations and gradients captured from model.features have identical,
    compatible spatial shapes and non-trivial gradients (ConvNeXt-Tiny)."""
    image_data, image_name = image_pair
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    import inference, torch
    model = inference.load_convnext_model()
    device = inference.get_device()
    tensor = inference.preprocess_image(img).to(device)

    acts, grads = {}, {}
    def fwd(m, i, o): acts['v'] = o.detach()
    def bwd(m, gi, go): grads['v'] = go[0].detach()
    hf = model.features.register_forward_hook(fwd)
    hb = model.features.register_full_backward_hook(bwd)
    model.zero_grad()
    logits, _feats = model.forward_with_features(tensor)
    logits.backward()
    hf.remove(); hb.remove()

    act, grad = acts['v'], grads['v']
    assert act.shape == grad.shape, \
        "activation/gradient shape mismatch: %s vs %s" % (act.shape, grad.shape)
    assert act.shape[2:] == grad.shape[2:], \
        "spatial dims mismatch: %s vs %s" % (act.shape[2:], grad.shape[2:])
    assert bool((grad != 0).any().item()), "all gradients are zero"


def test_batch_prediction(app_client, all_images):
    """Test /predict_batch endpoint."""
    image_datas, image_names = all_images
    files = [(io.BytesIO(data), name) for data, name in zip(image_datas, image_names)]
    resp = app_client.post('/predict_batch', data={'files': files}, content_type='multipart/form-data')
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


def test_error_handling(app_client):
    """Test error handling for invalid inputs."""
    # No file
    resp = app_client.post('/predict', data={}, content_type='multipart/form-data')
    assert resp.status_code == 400, "no file should be 400, got %s" % resp.status_code
    assert 'error' in resp.get_json(), "missing error message"

    # Invalid file type
    resp = app_client.post('/predict', data={'file': (io.BytesIO(b'notanimage'), 'test.txt')},
                           content_type='multipart/form-data')
    assert resp.status_code == 400, "invalid type should be 400, got %s" % resp.status_code

    # Corrupted image
    resp = app_client.post('/predict', data={'file': (io.BytesIO(b'corrupted'), 'test.jpg')},
                           content_type='multipart/form-data')
    assert resp.status_code == 400, "corrupted should be 400, got %s" % resp.status_code

    # File too large
    large = b'x' * (11 * 1024 * 1024)
    resp = app_client.post('/predict', data={'file': (io.BytesIO(large), 'large.jpg')},
                           content_type='multipart/form-data')
    assert resp.status_code == 413, "too large should be 413, got %s" % resp.status_code

    # Batch: too many files
    batch_files = [(io.BytesIO(b'fake'), 'test%d.jpg' % i) for i in range(65)]
    resp = app_client.post('/predict_batch', data={'files': batch_files},
                           content_type='multipart/form-data')
    assert resp.status_code == 400, "too many batch should be 400, got %s" % resp.status_code

    # Batch: total size too large
    large = io.BytesIO(b'x' * (60 * 1024 * 1024))
    resp = app_client.post('/predict_batch', data={'files': [(large, 'large.jpg')]},
                           content_type='multipart/form-data')
    assert resp.status_code == 413, "batch too large should be 413, got %s" % resp.status_code


# ----------------------------------------------------------------------
# Standalone runner (no pytest required)
# ----------------------------------------------------------------------
def run_all_tests(model_name):
    """Run complete test suite for a model."""
    global CURRENT_MODEL
    CURRENT_MODEL = model_name
    print("\n" + "=" * 72)
    print("Testing model: %s" % model_name)
    print("=" * 72, flush=True)

    import config
    import importlib
    import app as app_module

    # Ensure the app module is configured for this model
    if app_module.active_model != model_name:
        config.DEFAULT_MODEL = model_name
        importlib.reload(app_module)
    model_client = app_module.app.test_client()

    run_test("health_endpoint", test_health, model_client, model_name)

    # Load all test images
    image_datas = [load_image(path) for path, _ in TEST_IMAGES]

    # Single predictions (includes Grad-CAM validity check).
    for (path, label), data in zip(TEST_IMAGES, image_datas):
        upload_name = os.path.basename(path)
        run_test("single_prediction(%s)" % label,
                 test_single_prediction, model_client, (data, upload_name))
        run_test("heatmap_alignment(%s)" % label,
                 test_heatmap_alignment, model_client, (data, upload_name))

    # Batch prediction
    upload_names = [os.path.basename(path) for path, _ in TEST_IMAGES]
    run_test("batch_prediction", test_batch_prediction, model_client,
             (image_datas, upload_names))

    # Error handling
    run_test("error_handling", test_error_handling, model_client)


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
