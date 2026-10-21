"""PyTorch inference backend for ConvNeXt-Tiny model.

Loads the trained checkpoint and provides batched inference with
optional Grad-CAM explanation generation.
"""

import io
import os
import base64
import logging
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
from PIL import Image, ImageOps

import config

log = logging.getLogger(__name__)

# EXIF tag 274 = Orientation
EXIF_ORIENTATION_TAG = 274

# ──────────────────────────────────────────────────────────────────────
# Model definition (must match train_convnext_tiny.py exactly)
# ──────────────────────────────────────────────────────────────────────
STAGE_OF_WIDTH = {96: 1, 192: 2, 384: 3, 768: 4}


class DermaConvNeXt(nn.Module):
    """ConvNeXt-Tiny + GAP + Dropout + Linear(1).

    ImageNet normalisation is a real layer inside the graph.
    Feed raw RGB in [0, 255] and nothing else.
    """

    def __init__(self, dropout=0.30, pretrained=False):
        super().__init__()
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        if pretrained:
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            net = convnext_tiny(weights=weights)
        else:
            net = convnext_tiny(weights=None)

        self.register_buffer("mean", torch.tensor(config.CONVNEXT_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(config.CONVNEXT_STD).view(1, 3, 1, 1))
        self.features = net.features
        self.avgpool = net.avgpool
        self.norm = net.classifier[0]
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(768, 1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_u8: torch.Tensor) -> torch.Tensor:
        """x_u8: (B,3,H,W) float32 in [0,255] raw RGB."""
        x = x_u8.float().div_(255.0)
        x = (x - self.mean) / self.std
        x = self.features(x)
        x = self.avgpool(x)
        x = self.norm(x)
        x = torch.flatten(x, 1)
        return self.head(self.dropout(x)).squeeze(1)

    def forward_with_features(self, x_u8: torch.Tensor):
        """Forward pass returning both logits and final spatial feature map (before GAP).
        
        Returns:
            logits: (B,) cancer logits
            features: (B, 768, 7, 7) spatial feature map before global average pooling
        """
        x = x_u8.float().div_(255.0)
        x = (x - self.mean) / self.std
        features = self.features(x)  # (B, 768, 7, 7)
        x = self.avgpool(features)
        x = self.norm(x)
        x = torch.flatten(x, 1)
        logits = self.head(self.dropout(x)).squeeze(1)
        return logits, features

    def forward_with_embedding(self, x_u8: torch.Tensor):
        """Forward pass returning the logit and the 768-d penultimate embedding.

        The embedding is the exact vector the classifier head consumes: global
        average pooled, layer-normalised and flattened. It is the feature space
        the input-domain gate measures against.

        Returns:
            logits: (B,) cancer logits
            embedding: (B, 768) penultimate features
        """
        x = x_u8.float().div_(255.0)
        x = (x - self.mean) / self.std
        x = self.features(x)
        x = self.avgpool(x)
        x = self.norm(x)
        embedding = torch.flatten(x, 1)
        logits = self.head(self.dropout(embedding)).squeeze(1)
        return logits, embedding

    @staticmethod
    def _child_width(child):
        widest = 0
        for m in child.modules():
            if isinstance(m, nn.Conv2d):
                widest = max(widest, m.out_channels)
        if widest in STAGE_OF_WIDTH:
            return widest
        if widest % 4 == 0 and widest // 4 in STAGE_OF_WIDTH:
            return widest // 4
        return widest

    def feature_stage_map(self):
        return {i: ("stem" if i == 0 else "stage%d" % self.STAGE_OF_WIDTH[self._child_width(c)])
                for i, c in enumerate(self.features)}


# ──────────────────────────────────────────────────────────────────────
# Global model instance (loaded once at startup)
# ──────────────────────────────────────────────────────────────────────
_model: Optional[DermaConvNeXt] = None
_device: torch.device = torch.device("cpu")
_last_conv_layer: Optional[nn.Module] = None
_grad_model: Optional[nn.Module] = None


def get_device() -> torch.device:
    """Return the configured device (CUDA if available, else CPU)."""
    global _device
    if config.CONVNEXT_DEVICE == "cuda" and torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")
    return _device


def load_convnext_model() -> DermaConvNeXt:
    """Load the ConvNeXt-Tiny model from checkpoint. Called once at startup."""
    global _model, _device, _last_conv_layer, _grad_model

    if _model is not None:
        return _model

    device = get_device()
    log.info("Loading ConvNeXt-Tiny model on %s...", device)

    # Initialize model architecture
    model = DermaConvNeXt(dropout=0.30, pretrained=False).to(device)
    model.eval()

    # Find checkpoint
    ckpt_path = config.CONVNEXT_CKPT
    if not os.path.exists(ckpt_path):
        # Fallback to best model
        ckpt_path = config.CONVNEXT_BEST

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No ConvNeXt checkpoint found at {config.CONVNEXT_CKPT} or {config.CONVNEXT_BEST}"
        )

    # Load state dict
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        log.info("Loaded checkpoint: val_auc=%.4f, stage=%s, epoch=%d",
                 checkpoint.get("val_auc", -1),
                 checkpoint.get("stage", "?"),
                 checkpoint.get("epoch", -1))
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        log.info("Loaded checkpoint: val_auc=%.4f, stage=%s, epoch=%d",
                 checkpoint.get("best_auc", -1),
                 checkpoint.get("stage", "?"),
                 checkpoint.get("epoch", -1))
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)

    model.eval()
    _model = model

    # Grad-CAM target layer: model.features (final spatial feature map before
    # global average pooling). Kept for reference/debugging - the heatmap is
    # produced with torch.autograd.grad on the feature map itself, so no hook
    # is registered on this module.
    _last_conv_layer = model.features

    log.info("ConvNeXt-Tiny model loaded successfully (params: %.2f M)",
             sum(p.numel() for p in model.parameters()) / 1e6)
    log.info("Grad-CAM target layer: %s (type: %s)",
             "model.features", type(_last_conv_layer).__name__)
    return model


def apply_exif_orientation(image: Image.Image) -> Image.Image:
    """Return the image with its EXIF orientation applied.

    Cameras (and phone galleries) store the sensor data unrotated and record
    the intended viewing direction in the EXIF orientation tag. PIL does not
    apply that tag on its own, so without this helper a portrait photo taken
    with a phone is analysed rotated by 90 degrees while the browser shows the
    user the upright picture. The ConvNeXt model is strongly orientation
    sensitive, so this changes the prediction, and the Grad-CAM overlay would
    not line up with the photo the user sees.

    Images without an orientation tag are returned unchanged (no copy, no
    pixel change), so this is a no-op for the HAM10000 images and for the
    project's own clinical test images.
    """
    try:
        exif = image.getexif()
    except Exception:                                   # pragma: no cover - broken EXIF
        return image
    if not exif:
        return image
    try:
        orientation = exif.get(EXIF_ORIENTATION_TAG)
    except Exception:                                   # pragma: no cover - broken EXIF
        return image
    if orientation is None or orientation == 1:
        return image
    try:
        return ImageOps.exif_transpose(image)
    except Exception as exc:                           # pragma: no cover - broken EXIF
        log.warning("Could not apply EXIF orientation (%s) - image left unrotated", exc)
        return image


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess PIL image to tensor (B, 3, 224, 224) in [0, 255]."""
    image = apply_exif_orientation(image)
    img = image.convert("RGB").resize(config.CONVNEXT_INPUT_SIZE)
    arr = np.array(img, dtype=np.uint8)  # (H, W, 3) in [0, 255]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)  # (1, 3, H, W)
    return tensor.float()


def preprocess_batch(images: List[Image.Image]) -> torch.Tensor:
    """Preprocess a batch of PIL images."""
    tensors = [preprocess_image(img).squeeze(0) for img in images]
    return torch.stack(tensors, dim=0).float()  # (B, 3, H, W)


def _metas_from_probs(probs: np.ndarray) -> List[Dict]:
    """Build the per-image metadata dicts from cancer probabilities."""
    metas = []
    for p in probs:
        prob = float(p)
        calibrated = prob  # No temperature scaling for ConvNeXt
        verdict = "affected" if calibrated >= config.CONVNEXT_THRESHOLD else "not_affected"
        detected = verdict == "affected"
        confidence = calibrated if calibrated >= 0.5 else 1.0 - calibrated
        risk = "high" if calibrated >= config.RISK_HIGH else ("medium" if calibrated >= config.RISK_MEDIUM else "low")

        metas.append({
            "cancer_probability": round(calibrated, 4),
            "verdict": verdict,
            "cancer_detected": detected,
            "confidence": round(confidence, 4),
            "risk_level": risk,
        })
    return metas


@torch.no_grad()
def _forward_batch(images: List[Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
    """Single forward pass returning cancer probabilities and penultimate embeddings."""
    model = load_convnext_model()
    device = get_device()

    # Preprocess
    batch_tensor = preprocess_batch(images).to(device, non_blocking=True)

    # Inference with autocast for speed
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        logits, embedding = model.forward_with_embedding(batch_tensor)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        embeddings = embedding.float().cpu().numpy()

    return probs, embeddings


@torch.no_grad()
def predict_batch(images: List[Image.Image]) -> Tuple[np.ndarray, List[Dict]]:
    """Run inference on a batch of images.

    Returns:
        probs: numpy array of cancer probabilities (N,)
        metas: list of dicts with metadata per image
    """
    probs, _ = _forward_batch(images)
    return probs, _metas_from_probs(probs)


@torch.no_grad()
def predict_batch_with_embeddings(images: List[Image.Image]) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
    """Run inference on a batch of images and also return the penultimate embeddings.

    Same probabilities and metadata as predict_batch(); the extra return value is
    the (N, 768) feature matrix used by the input-domain gate.

    Returns:
        probs: numpy array of cancer probabilities (N,)
        metas: list of dicts with metadata per image
        embeddings: numpy array of 768-d penultimate features (N, 768)
    """
    probs, embeddings = _forward_batch(images)
    return probs, _metas_from_probs(probs), embeddings


def predict_single(image: Image.Image) -> Dict:
    """Run inference on a single image with optional Grad-CAM."""
    probs, metas = predict_batch([image])
    meta = metas[0]

    # Grad-CAM (optional)
    heatmap_b64 = ""
    if config.ENABLE_GRADCAM_SINGLE:
        try:
            target = "affected" if meta["cancer_detected"] else "not_affected"
            heatmap_b64 = generate_gradcam(image, target_class=target)
        except Exception as exc:
            log.warning("Grad-CAM generation failed: %s", exc)

    meta["heatmap"] = heatmap_b64
    meta["prediction"] = "Malignant (Cancer Detected)" if meta["cancer_detected"] else "Benign (No Cancer Detected)"
    return meta


def generate_gradcam(image: Image.Image, target_class: str = "affected") -> str:
    """Generate Grad-CAM heatmap for a single image. Returns base64 PNG.

    Args:
        image: PIL Image
        target_class: "affected" (cancer) or "not_affected" (non-cancer)
                     Determines which class logit to use for gradients.

    Thread safety
    -------------
    The heatmap is derived from an explicit ``torch.autograd.grad`` call on the
    feature map that the forward pass returned. Nothing is registered on the
    shared model and no parameter ``.grad`` is written, so two concurrent
    requests cannot overwrite each other's activations or gradients.

    The previous implementation registered a forward and a full-backward hook
    on the shared ``model.features`` module and used ``model.zero_grad()`` plus
    ``target.backward()``. That was not thread safe: with two concurrent
    requests both hooks fire for both forward passes, so each request read
    whichever activations the other request had produced last, and one
    request's ``zero_grad()`` could wipe the gradients the other was using.
    The autograd formulation produces bit-identical heatmaps (verified
    max |difference| = 0 on the clinical test images) without the shared
    state.
    """
    model = load_convnext_model()
    device = get_device()

    # Preprocess image
    tensor = preprocess_image(image).to(device)

    # Forward pass - the returned feature map is the gradient target, so no
    # hook is needed and the graph is built only for this request.
    logits, features = model.forward_with_features(tensor)

    # Select target-class logit for gradient flow
    # Model output is (B,) after squeeze(1)
    if target_class == "affected":
        target = logits[:, 0] if logits.dim() > 1 else logits
    else:
        target = -logits[:, 0] if logits.dim() > 1 else -logits

    # Gradient of the target logit with respect to the final spatial feature
    # map. autograd.grad does not touch the parameters' .grad attributes.
    (grad,) = torch.autograd.grad(target, features)

    act = features.detach()
    grad = grad.detach()

    # Log shapes and statistics
    log.info("Grad-CAM: target_layer=model.features, act_shape=%s, grad_shape=%s",
             tuple(act.shape), tuple(grad.shape))
    log.info("Grad-CAM: act_abs_mean=%.4f grad_abs_mean=%.4f",
             act.abs().mean().item(), grad.abs().mean().item())

    # Global average pooling of gradients -> channel weights
    weights = grad.mean(dim=(2, 3))  # (1, C)

    # Weighted combination of activation maps
    cam = (act * weights.view(-1, 1, 1)).sum(dim=1)  # (1, H, W)
    cam = cam.squeeze(0)  # remove batch dim -> (H, W)
    cam = F.relu(cam)

    if cam.max() > 0:
        cam = cam / cam.max()

    # Convert to numpy and resize
    cam_np = cam.detach().cpu().numpy()

    # NO GaussianBlur - evaluate unblurred map
    log.info("Grad-CAM: cam_range=[%.4f, %.4f] cam_mean=%.4f hotspot=%s",
             cam_np.min(), cam_np.max(), cam_np.mean(),
             np.unravel_index(cam_np.argmax(), cam_np.shape))

    cam_resized = cv2.resize(cam_np, config.CONVNEXT_INPUT_SIZE)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    orig = np.array(image.convert("RGB").resize(config.CONVNEXT_INPUT_SIZE), dtype=np.uint8)
    blended = np.uint8(0.55 * orig + 0.45 * heatmap)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ──────────────────────────────────────────────────────────────────────
# Temperature scaling (for MobileNetV2 compatibility)
# ──────────────────────────────────────────────────────────────────────
def calibrated_prob(raw_prob: float, temperature: float = 1.0) -> float:
    """Apply temperature scaling: p_cal = sigmoid(logit(p) / T)."""
    import math
    logit = math.log(raw_prob / (1.0 - raw_prob + 1e-7) + 1e-7)
    return 1.0 / (1.0 + math.exp(-logit / temperature))


# ──────────────────────────────────────────────────────────────────────
# Initialization check
# ──────────────────────────────────────────────────────────────────────
def is_model_loaded() -> bool:
    return _model is not None


def get_model_info() -> Dict[str, Any]:
    """Return model info for health/debug endpoints."""
    if _model is None:
        return {"loaded": False}
    return {
        "loaded": True,
        "architecture": "ConvNeXt-Tiny",
        "framework": "PyTorch",
        "device": str(get_device()),
        "input_size": config.CONVNEXT_INPUT_SIZE,
        "threshold": config.CONVNEXT_THRESHOLD,
        "params_m": round(sum(p.numel() for p in _model.parameters()) / 1e6, 2),
    }