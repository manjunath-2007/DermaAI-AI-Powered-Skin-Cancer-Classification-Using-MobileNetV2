#!/usr/bin/env python3
"""Regression tests for EXIF orientation handling and Grad-CAM thread safety.

EXIF orientation
----------------
A phone camera stores the sensor data unrotated and records the intended
viewing direction in the EXIF orientation tag. The backend used to ignore
that tag, so a portrait photo was analysed rotated by 90 degrees while the
browser showed the user the upright picture - the prediction was computed for
a rotated lesion and the Grad-CAM overlay did not line up with the photo.
``inference.apply_exif_orientation`` now applies the tag.

Grad-CAM thread safety
-----------------------
The Grad-CAM heatmap used to be captured with forward/backward hooks that
were registered on the *shared* model module, combined with
``model.zero_grad()`` + ``target.backward()``. Two concurrent requests
therefore overwrote each other's activations and gradients. The heatmap is
now produced with ``torch.autograd.grad`` on the feature map, which touches
no shared state.

Run with either::

    python tests/test_exif_gradcam.py        # standalone runner
    pytest tests/test_exif_gradcam.py -v     # pytest runner
"""

import io
import os
import sys
import threading

# Add parent directory to sys.path so 'inference'/'app' can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
logging.disable(logging.CRITICAL)

import numpy as np
import pytest
from PIL import Image

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_IMAGES = os.path.join(APP_DIR, "test-images")

import inference

try:
    from app import app as flask_app
    _API_AVAILABLE = True
except Exception as exc:                                # pragma: no cover
    flask_app = None
    _API_AVAILABLE = False
    _API_IMPORT_ERROR = str(exc)


def _image(name):
    img = Image.open(os.path.join(TEST_IMAGES, name))
    img.load()
    return img.convert("RGB")


def _tagged_bytes(img, orientation, transpose):
    """Store the picture rotated and tag it so it displays the right way up.

    ``transpose`` is the PIL transpose that produces the stored pixels;
    ``orientation`` is the EXIF tag that undoes it on display.
    """
    stored = img.transpose(transpose)
    exif = stored.getexif()
    exif[274] = orientation
    buf = io.BytesIO()
    stored.save(buf, format="JPEG", exif=exif.tobytes(), quality=97)
    return buf.getvalue()


# ----------------------------------------------------------------------
# EXIF orientation
# ----------------------------------------------------------------------
def test_no_orientation_tag_returns_the_same_object():
    """An image without an orientation tag is passed through untouched."""
    for name in ("cancer-melanoma.jpg", "noncancer-nevus.jpg",
                 "noncancer-congenital-nevus.jpg", "noncancer-seborrheic-keratosis.jpg"):
        img = _image(name)
        out = inference.apply_exif_orientation(img)
        assert out is img, "%s: image was copied although it carries no rotation" % name
        assert np.array_equal(np.array(out.convert("RGB")),
                              np.array(img.convert("RGB")))


def test_orientation_is_applied_and_idempotent():
    """The tag is applied once, and applying it again changes nothing."""
    img = _image("cancer-melanoma.jpg")
    tagged = Image.open(io.BytesIO(_tagged_bytes(img, 6, Image.Transpose.ROTATE_90)))
    tagged.load()
    assert tagged.getexif().get(274) == 6

    once = inference.apply_exif_orientation(tagged)
    twice = inference.apply_exif_orientation(once)
    assert once.size == tagged.size          # square source: same dimensions
    assert twice.getexif().get(274) in (None, 1), "orientation tag was not cleared"
    a = np.array(once.convert("RGB"))
    b = np.array(twice.convert("RGB"))
    assert np.array_equal(a, b), "applying the orientation twice changed the image"


def test_api_honours_exif_orientation():
    """A rotated-but-tagged upload must give the same answer as the upright one.

    Before the fix the tag was ignored: the backend analysed the rotated
    pixels while the browser displayed the upright picture, which changed the
    prediction (a melanoma photo scored 0.12 instead of 0.69).
    """
    if not _API_AVAILABLE:
        pytest.skip("Flask app could not be imported: %s" % _API_IMPORT_ERROR)

    client = flask_app.test_client()
    img = _image("cancer-melanoma.jpg")
    with open(os.path.join(TEST_IMAGES, "cancer-melanoma.jpg"), "rb") as fh:
        plain_bytes = fh.read()

    base = client.post("/predict", data={"file": (io.BytesIO(plain_bytes), "plain.jpg")},
                       content_type="multipart/form-data").get_json()
    assert base["success"] is True
    assert base["analysable"] is True

    for orientation, transpose in ((3, Image.Transpose.ROTATE_180),
                                   (6, Image.Transpose.ROTATE_90),
                                   (8, Image.Transpose.ROTATE_270)):
        blob = _tagged_bytes(img, orientation, transpose)
        resp = client.post("/predict", data={"file": (io.BytesIO(blob), "rot.jpg")},
                           content_type="multipart/form-data").get_json()
        assert resp["success"] is True
        assert resp["analysable"] is True
        assert abs(resp["cancer_probability"] - base["cancer_probability"]) < 0.05, (
            "orientation %d changed the prediction: %s vs baseline %s"
            % (orientation, resp["cancer_probability"], base["cancer_probability"]))
        assert resp["verdict"] == base["verdict"]


# ----------------------------------------------------------------------
# Grad-CAM thread safety
# ----------------------------------------------------------------------
def test_gradcam_registers_no_hooks_on_the_shared_model():
    """No forward/backward hook may survive a Grad-CAM call.

    Hooks live on the shared model module, so a leftover hook would fire for
    every other request as well.
    """
    model = inference.load_convnext_model()

    def hook_count(module):
        """Number of hooks registered on a module (PyTorch renamed the dicts)."""
        total = 0
        for attr in ("_forward_hooks", "_backward_hooks", "_full_backward_hooks",
                     "_forward_pre_hooks", "_state_dict_hooks"):
            total += len(getattr(module, attr, {}) or {})
        return total

    before = hook_count(model.features)
    inference.generate_gradcam(_image("cancer-melanoma.jpg"), target_class="affected")
    assert hook_count(model.features) == before


def test_gradcam_leaves_no_parameter_gradients():
    """Grad-CAM must not write to the shared parameters' .grad attributes."""
    model = inference.load_convnext_model()
    for p in model.parameters():
        p.grad = None
    inference.generate_gradcam(_image("cancer-melanoma.jpg"), target_class="affected")
    with_grad = [name for name, p in model.named_parameters() if p.grad is not None]
    assert not with_grad, "parameters still hold gradients: %s" % with_grad[:5]


def test_concurrent_gradcam_matches_the_sequential_result():
    """Concurrent heatmaps must equal the heatmaps produced one at a time."""
    names = ("cancer-melanoma.jpg", "noncancer-nevus.jpg",
             "noncancer-congenital-nevus.jpg", "noncancer-seborrheic-keratosis.jpg")
    images = [_image(n) for n in names]
    reference = [inference.generate_gradcam(im, target_class="affected") for im in images]
    assert all(reference), "a sequential Grad-CAM call produced no heatmap"

    results = [None] * len(images)
    barrier = threading.Barrier(len(images))

    def worker(index):
        barrier.wait()
        results[index] = inference.generate_gradcam(images[index], target_class="affected")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(images))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i, got in enumerate(results):
        assert got, "concurrent Grad-CAM call %d produced no heatmap" % i
        assert got == reference[i], \
            "concurrent heatmap %d differs from the sequential reference" % i


def test_gradcam_is_deterministic_for_the_same_image():
    """The same image must always produce the same heatmap."""
    img = _image("cancer-melanoma.jpg")
    first = inference.generate_gradcam(img, target_class="affected")
    second = inference.generate_gradcam(img, target_class="affected")
    assert first == second


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
            print("[PASS] %s" % name)
        except Exception as exc:
            failures.append((name, exc))
            print("[FAIL] %s -- %s" % (name, exc))
    print("\n%d failure(s)" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
