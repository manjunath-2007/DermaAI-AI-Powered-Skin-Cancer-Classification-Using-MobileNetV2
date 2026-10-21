import os
import io
import base64
import logging

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image, UnidentifiedImageError

import config
import quality
import domain
from inference import apply_exif_orientation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_BATCH_FILE_SIZE

# Model state
model_tf = None  # MobileNetV2 (TensorFlow)
model_pt = None  # ConvNeXt-Tiny (PyTorch)
active_model = config.DEFAULT_MODEL  # "convnext_tiny" or "mobilenetv2"

# ──────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────

def load_mobilenet_model():
    """Load MobileNetV2 Keras model."""
    global model_tf
    if model_tf is not None:
        return model_tf

    if not os.path.exists(config.MOBILENET_PATH):
        log.error("MobileNetV2 model file not found: %s", config.MOBILENET_PATH)
        return None

    try:
        import tensorflow as tf
        model_tf = tf.keras.models.load_model(config.MOBILENET_PATH)
        log.info("MobileNetV2 loaded. input=%s output=%s", model_tf.input_shape, model_tf.output_shape)
        return model_tf
    except Exception as exc:
        log.error("MobileNetV2 load failed: %s", exc)
        return None


def load_convnext_model():
    """Load ConvNeXt-Tiny PyTorch model via inference module."""
    global model_pt
    if model_pt is not None:
        return model_pt

    try:
        import inference
        model_pt = inference.load_convnext_model()
        log.info("ConvNeXt-Tiny loaded on %s", inference.get_device())
        return model_pt
    except Exception as exc:
        log.error("ConvNeXt-Tiny load failed: %s", exc)
        return None


def load_active_model():
    """Load the currently active model based on config."""
    if active_model == "convnext_tiny":
        return load_convnext_model()
    elif active_model == "mobilenetv2":
        return load_mobilenet_model()
    else:
        log.error("Unknown model: %s", active_model)
        return None


# ──────────────────────────────────────────────────────────────────────
# MobileNetV2 (TF) helpers
# ──────────────────────────────────────────────────────────────────────

MOBILENET_INPUT_SIZE = config.MOBILENET_INPUT_SIZE
MOBILENET_TEMPERATURE = config.MOBILENET_TEMPERATURE
MOBILENET_THRESHOLD = config.MOBILENET_THRESHOLD


def mobilenet_calibrated_prob(raw_prob: float) -> float:
    """Apply temperature scaling for MobileNetV2."""
    import math
    logit = math.log(raw_prob / (1.0 - raw_prob + 1e-7) + 1e-7)
    return 1.0 / (1.0 + math.exp(-logit / MOBILENET_TEMPERATURE))


def mobilenet_grad_cam(img_array: np.ndarray, target_class: str = "affected") -> str:
    """Return a base64-encoded PNG heatmap blended over the input image.
    
    Args:
        img_array: Preprocessed image array (1, H, W, 3) in [0, 1]
        target_class: "affected" (cancer) or "not_affected" (non-cancer)
    """
    import tensorflow as tf
    import cv2

    if model_tf is None:
        return ""

    # Find the last Conv layer in MobileNetV2 (before GlobalAveragePooling2D)
    last_conv = next(
        (l for l in reversed(model_tf.layers) if isinstance(l, tf.keras.layers.Conv2D)),
        None,
    )
    if last_conv is None:
        return ""

    # Build a model that outputs:
    # 1. The last conv layer's output (spatial feature map)
    # 2. The pre-sigmoid logits (dense layer output before sigmoid)
    dense_layer = model_tf.get_layer('dense')
    logits_model = tf.keras.Model(
        inputs=model_tf.inputs,
        outputs=[last_conv.output, dense_layer.input]
    )

    with tf.GradientTape() as tape:
        conv_out, logits = logits_model(img_array, training=False)
        # logits shape: (1, 1) - pre-sigmoid cancer logit
        # Use class-specific target on LOGITS (not probabilities)
        if target_class == "affected":
            loss = logits[:, 0]
        else:
            loss = -logits[:, 0]

    grads = tape.gradient(loss, conv_out)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1).numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam /= cam.max()

    # Optional: Gaussian smoothing before resize
    cam = cv2.GaussianBlur(cam, (3, 3), 0)
    if cam.max() > 0:
        cam /= cam.max()

    cam_resized = cv2.resize(cam, MOBILENET_INPUT_SIZE)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # img_array is in [0, 1], convert back to [0, 255]
    orig = np.uint8(img_array[0] * 255)
    blended = np.uint8(0.55 * orig + 0.45 * heatmap)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def mobilenet_preprocess(img: Image.Image) -> np.ndarray:
    """Resize to 224×224, convert to RGB, normalise to [0,1] (matches training preprocessing)."""
    img = apply_exif_orientation(img)
    img = img.convert("RGB").resize(MOBILENET_INPUT_SIZE)
    arr = np.array(img, dtype="float32") / 255.0  # [0, 1] - matches training
    return np.expand_dims(arr, axis=0)


def mobilenet_cancer_probability(image: Image.Image) -> float | None:
    """Cancer probability from the independent MobileNetV2 model.

    Used only as a second opinion for images that land in the ambiguous band of
    the input-domain gate. Returns None when the model cannot be loaded so the
    caller can fall back to keeping the image in domain (refusing a genuine
    lesion photo because a helper model is missing is the worse error).
    """
    model = load_mobilenet_model()
    if model is None:
        return None
    try:
        arr = mobilenet_preprocess(image)
        raw_prob = float(model.predict(arr, verbose=0)[0][0])
        return float(mobilenet_calibrated_prob(raw_prob))
    except Exception as exc:
        log.warning("MobileNetV2 second opinion failed: %s", exc)
        return None


def mobilenet_predict_single(image: Image.Image) -> dict:
    """Run inference on a single image using MobileNetV2."""
    if model_tf is None:
        raise RuntimeError("MobileNetV2 model not loaded")

    arr = mobilenet_preprocess(image)
    raw_prob = float(model_tf.predict(arr, verbose=0)[0][0])
    cancer_prob = mobilenet_calibrated_prob(raw_prob)
    verdict = "affected" if cancer_prob >= MOBILENET_THRESHOLD else "not_affected"
    detected = verdict == "affected"
    confidence = cancer_prob if cancer_prob >= 0.5 else 1.0 - cancer_prob
    risk = "high" if cancer_prob >= config.RISK_HIGH else ("medium" if cancer_prob >= config.RISK_MEDIUM else "low")

    heatmap_b64 = ""
    if config.ENABLE_GRADCAM_SINGLE:
        try:
            import cv2  # noqa: F401
            target = "affected" if detected else "not_affected"
            heatmap_b64 = mobilenet_grad_cam(arr, target_class=target)
        except Exception as exc:
            log.warning("Grad-CAM skipped: %s", exc)

    return {
        "cancer_probability": round(cancer_prob, 4),
        "verdict": verdict,
        "cancer_detected": detected,
        "confidence": round(confidence, 4),
        "risk_level": risk,
        "heatmap": heatmap_b64,
        "prediction": "Malignant (Cancer Detected)" if detected else "Benign (No Cancer Detected)",
    }


def mobilenet_predict_batch(images: list[Image.Image]) -> list[dict]:
    """Run inference on a batch of images using MobileNetV2."""
    if model_tf is None:
        raise RuntimeError("MobileNetV2 model not loaded")

    results = []
    for img in images:
        arr = mobilenet_preprocess(img)
        raw_prob = float(model_tf.predict(arr, verbose=0)[0][0])
        cancer_prob = mobilenet_calibrated_prob(raw_prob)
        verdict = "affected" if cancer_prob >= MOBILENET_THRESHOLD else "not_affected"
        detected = verdict == "affected"
        confidence = cancer_prob if cancer_prob >= 0.5 else 1.0 - cancer_prob
        risk = "high" if cancer_prob >= config.RISK_HIGH else ("medium" if cancer_prob >= config.RISK_MEDIUM else "low")

        results.append({
            "cancer_probability": round(cancer_prob, 4),
            "verdict": verdict,
            "cancer_detected": detected,
            "confidence": round(confidence, 4),
            "risk_level": risk,
            # Batch predictions never carry a Grad-CAM heatmap, but the key is
            # present so every item of a batch has the same shape as the
            # single-image response.
            "heatmap": "",
            "prediction": "Malignant (Cancer Detected)" if detected else "Benign (No Cancer Detected)",
        })
    return results


# ──────────────────────────────────────────────────────────────────────
# ConvNeXt-Tiny (PyTorch) helpers - delegate to inference module
# ──────────────────────────────────────────────────────────────────────

def convnext_predict_single(image: Image.Image) -> dict:
    """Run inference on a single image using ConvNeXt-Tiny."""
    import inference
    return inference.predict_single(image)


def convnext_predict_batch(images: list[Image.Image]) -> list[dict]:
    """Run inference on a batch of images using ConvNeXt-Tiny."""
    import inference
    _, metas = inference.predict_batch(images)
    for meta in metas:
        meta["prediction"] = "Malignant (Cancer Detected)" if meta["cancer_detected"] else "Benign (No Cancer Detected)"
        # Batch predictions never carry a Grad-CAM heatmap, but the key is
        # present so every item of a batch has the same shape as the
        # single-image response.
        meta["heatmap"] = ""
    return metas


# ──────────────────────────────────────────────────────────────────────
# Unified helpers
# ──────────────────────────────────────────────────────────────────────

def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXT


# ──────────────────────────────────────────────────────────────────────
# Flask routes
# ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    model_ready = (active_model == "convnext_tiny" and model_pt is not None) or \
                  (active_model == "mobilenetv2" and model_tf is not None)
    return render_template("index.html", model_ready=model_ready, active_model=active_model)


@app.route("/health")
def health():
    info = {
        "status": "ok",
        "active_model": active_model,
        "model_loaded": False,
    }
    if active_model == "convnext_tiny":
        import inference
        info["model_loaded"] = inference.is_model_loaded()
        if info["model_loaded"]:
            info.update(inference.get_model_info())
    elif active_model == "mobilenetv2":
        info["model_loaded"] = model_tf is not None
    return jsonify(info)


@app.route("/predict", methods=["POST"])
def predict():
    """Single image prediction endpoint."""
    model = load_active_model()
    if model is None:
        return jsonify(success=False, error="Model not loaded on server."), 503

    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify(success=False, error="No file uploaded. Use field name 'file'."), 400
    if not allowed(file.filename):
        return jsonify(success=False, error="Only JPG/JPEG/PNG files are accepted."), 400

    raw = file.read()
    if len(raw) > config.MAX_FILE_SIZE:
        return jsonify(success=False, error="File exceeds 10 MB limit."), 413

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError):
        return jsonify(success=False, error="Cannot read the uploaded file as an image."), 400

    # Apply the EXIF orientation so the image is analysed the way the browser
    # shows it to the user. Without this a portrait phone photo is analysed
    # rotated by 90 degrees, which changes the prediction and misaligns the
    # Grad-CAM overlay.
    img = apply_exif_orientation(img)

    # Image quality check (warnings only — never rejects a decodable image)
    quality_result = quality.check_image_quality(img)

    # Input-domain gate: refuse to emit a cancer verdict for images that are
    # not dermoscopic skin lesions (ordinary photos such as classroom scenes).
    # It runs before the prediction so no heatmap is generated for an image
    # that is going to be refused anyway.
    domain_result = domain.check_domain(img, second_opinion=mobilenet_cancer_probability)
    if not domain_result["in_domain"]:
        log.info("Input-domain gate rejected an out-of-domain image (score=%s)",
                 domain_result["score"])
        result = domain.not_analysable_result(domain_result)
    else:
        try:
            if active_model == "convnext_tiny":
                result = convnext_predict_single(img)
            else:
                result = mobilenet_predict_single(img)
        except Exception as exc:
            log.error("Prediction error: %s", exc)
            return jsonify(success=False, error=f"Prediction failed: {exc}"), 500
        result["analysable"] = True
        result["domain_in_domain"] = True
        result["domain_score"] = domain_result["score"]

    result["success"] = True
    result["quality_warnings"] = quality_result["warnings"]
    result["quality_metrics"] = quality_result["metrics"]
    result["quality_is_valid"] = quality_result["is_valid"]
    log.info("Prediction: verdict=%s cancer_prob=%s risk=%s model=%s",
             result["verdict"], result["cancer_probability"],
             result["risk_level"], active_model)
    return jsonify(result)


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """Batch prediction endpoint (up to MAX_BATCH_SIZE images)."""
    model = load_active_model()
    if model is None:
        return jsonify(success=False, error="Model not loaded on server."), 503

    files = request.files.getlist("files")
    if not files:
        return jsonify(success=False, error="No files uploaded. Use field name 'files'."), 400

    if len(files) > config.MAX_BATCH_SIZE:
        return jsonify(success=False, error=f"Maximum {config.MAX_BATCH_SIZE} images per batch."), 400

    total_size = 0
    images = []
    quality_results = []
    for file in files:
        if file.filename == "":
            continue
        if not allowed(file.filename):
            return jsonify(success=False, error=f"Invalid file type: {file.filename}. Only JPG/JPEG/PNG accepted."), 400

        raw = file.read()
        total_size += len(raw)
        if total_size > config.MAX_BATCH_FILE_SIZE:
            return jsonify(success=False, error=f"Total batch size exceeds {config.MAX_BATCH_FILE_SIZE // (1024*1024)} MB limit."), 413

        try:
            img = Image.open(io.BytesIO(raw))
            img.load()
            # Apply the EXIF orientation so the image is analysed the way the
            # browser shows it to the user (see /predict).
            img = apply_exif_orientation(img)
            images.append(img)
        except (UnidentifiedImageError, OSError):
            return jsonify(success=False, error=f"Cannot read {file.filename} as an image."), 400

        # Quality check (warnings only — never rejects a decodable image)
        quality_results.append(quality.check_image_quality(img))

    if not images:
        return jsonify(success=False, error="No valid images in batch."), 400

    try:
        if active_model == "convnext_tiny":
            results = convnext_predict_batch(images)
        else:
            results = mobilenet_predict_batch(images)
    except Exception as exc:
        log.error("Batch prediction error: %s", exc)
        return jsonify(success=False, error=f"Batch prediction failed: {exc}"), 500

    # Attach quality info to each per-image result
    for pred, qr in zip(results, quality_results):
        pred["quality_warnings"] = qr["warnings"]
        pred["quality_metrics"] = qr["metrics"]
        pred["quality_is_valid"] = qr["is_valid"]

    # Input-domain gate: refuse to emit a cancer verdict for images that are
    # not dermoscopic skin lesions (ordinary photos such as classroom scenes).
    domain_results = domain.check_domains(images, second_opinion=mobilenet_cancer_probability)
    for pred, dom in zip(results, domain_results):
        pred["analysable"] = True
        pred["domain_in_domain"] = dom["in_domain"]
        pred["domain_score"] = dom["score"]
        if not dom["in_domain"]:
            log.info("Input-domain gate rejected an out-of-domain image (score=%s)",
                     dom["score"])
            pred.update(domain.not_analysable_result(dom))

    response = {
        "success": True,
        "model": active_model,
        "count": len(results),
        "predictions": results,
    }
    log.info("Batch prediction: %d images, model=%s", len(results), active_model)
    return jsonify(response)


@app.errorhandler(413)
def too_large(_):
    return jsonify(success=False, error="File exceeds size limit."), 413


# ──────────────────────────────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────────────────────────────

load_active_model()

if __name__ == "__main__":
    host = config.FLASK_HOST
    port = config.FLASK_PORT
    app.run(host=host, port=port, debug=config.FLASK_DEBUG, threaded=True)