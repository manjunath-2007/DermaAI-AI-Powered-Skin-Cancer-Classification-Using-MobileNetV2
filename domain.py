"""Input-domain gate for DermaAI.

Why this exists
---------------
The deployed ConvNeXt-Tiny is a binary dermoscopic-lesion classifier trained
only on HAM10000 (dermoscopic close-ups of skin). It has no notion of "this
image is not a skin image at all" - every input is forced into one of two
classes. On ordinary photographs the network behaves as a "dark blob on a light
field" detector, so scenes that happen to contain that structure (fruit in a
bowl, a car in a street, a classroom) are scored as malignant lesions with
high confidence. The image-quality checks in quality.py cannot catch this:
blur, brightness and overexposure say nothing about whether the picture is a
skin lesion at all.

How the gate works
------------------
The deployed model itself provides the domain descriptor: its 768-d penultimate
embedding (global average pooled, layer-normalised) is the exact feature space
the classifier head consumes. The gate measures the mean cosine distance from
an input embedding to its k nearest neighbours in the HAM10000 TRAIN-split
embedding reference, taking the worst case over {original, horizontal flip}
because the model is orientation sensitive.

Distance thresholds alone cannot separate every ordinary photo from every real
lesion photograph (some clinical photographs of a nose or an ear legitimately
sit far from the dermoscopic manifold). For the band between "clearly inside"
and "clearly outside" the gate therefore asks a second, independent opinion -
the already-deployed MobileNetV2 model - and only rejects when that model also
says the picture is not a lesion.

Nothing here changes the model, its weights, the 0.525 decision threshold or
any stored prediction array. The gate is a separate, additive safety layer.
"""

import logging
import os
import threading
from typing import Callable, List, Optional

import numpy as np
from PIL import Image

import config
import inference

log = logging.getLogger(__name__)

_lock = threading.Lock()
_reference: Optional[np.ndarray] = None
_reference_norm: Optional[np.ndarray] = None

REASON_OUTSIDE = "outside_lesion_domain"


# ──────────────────────────────────────────────────────────────────────
# Reference embedding matrix
# ──────────────────────────────────────────────────────────────────────

def _load_reference() -> Optional[np.ndarray]:
    """Load (and cache) the HAM10000 reference embeddings."""
    global _reference, _reference_norm

    if _reference is not None:
        return _reference

    with _lock:
        if _reference is not None:
            return _reference

        path = config.DOMAIN_REFERENCE_PATH
        if not os.path.exists(path):
            log.warning("Domain reference not found at %s - input-domain gate is inactive",
                        path)
            return None
        try:
            ref = np.load(path).astype(np.float32)
        except Exception as exc:                      # pragma: no cover - IO failure
            log.warning("Domain reference could not be read (%s) - gate is inactive", exc)
            return None

        if ref.ndim != 2 or ref.shape[1] != 768:
            log.warning("Domain reference has unexpected shape %s - gate is inactive",
                        tuple(ref.shape))
            return None

        _reference = ref
        _reference_norm = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-9)
        log.info("Domain reference loaded: %d embeddings from %s", ref.shape[0], path)
        return _reference


def is_available() -> bool:
    """True when the gate can be evaluated (reference present, gate enabled)."""
    if not config.DOMAIN_GATE_ENABLED:
        return False
    return _load_reference() is not None


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


# ──────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────

def _knn_cosine_distance(query: np.ndarray, reference_norm: np.ndarray, k: int) -> np.ndarray:
    """Mean cosine distance to the k nearest reference rows."""
    q = _normalize(query.astype(np.float32))
    out = np.empty(len(q), dtype=np.float32)
    k = int(min(k, reference_norm.shape[0]))
    chunk = 256
    for start in range(0, len(q), chunk):
        sims = q[start:start + chunk] @ reference_norm.T
        part = np.partition(sims, -k, axis=1)[:, -k:]
        out[start:start + chunk] = 1.0 - part.mean(axis=1)
    return out


def _flipped(image: Image.Image) -> Image.Image:
    return image.convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)


def domain_scores(images: List[Image.Image]) -> List[float]:
    """Distance of each image from the HAM10000 training manifold.

    Higher means more out-of-domain. Returns an empty list when the gate is
    unavailable so callers can fall back to the plain prediction.
    """
    if not config.DOMAIN_GATE_ENABLED or not images:
        return []

    reference = _load_reference()
    if reference is None:
        return []

    # Analyse the image the way the user sees it (EXIF orientation applied).
    images = [inference.apply_exif_orientation(img) for img in images]

    # Both views in one batched forward pass.
    views = []
    for img in images:
        views.append(img)
    for img in images:
        views.append(_flipped(img))

    try:
        _, _, embeddings = inference.predict_batch_with_embeddings(views)
    except Exception as exc:                          # pragma: no cover - inference failure
        log.warning("Domain scoring failed (%s) - gate is inactive for this request", exc)
        return []

    n = len(images)
    orig_emb = embeddings[:n]
    flip_emb = embeddings[n:]
    distances = np.maximum(
        _knn_cosine_distance(orig_emb, _reference_norm, config.DOMAIN_K),
        _knn_cosine_distance(flip_emb, _reference_norm, config.DOMAIN_K),
    )
    return [float(d) for d in distances]


def domain_score(image: Image.Image) -> Optional[float]:
    """Distance of a single image from the HAM10000 training manifold."""
    scores = domain_scores([image])
    return scores[0] if scores else None


# ──────────────────────────────────────────────────────────────────────
# Decision
# ──────────────────────────────────────────────────────────────────────

def is_out_of_domain(score: float,
                     second_opinion_cancer_prob: Optional[float] = None) -> bool:
    """Decide whether an image lies outside the lesion domain.

    Args:
        score: distance from the HAM10000 training manifold (domain_score()).
        second_opinion_cancer_prob: cancer probability from the independent
            MobileNetV2 model. ACCEPTED FOR COMPATIBILITY BUT NOT USED for the
            decision - see the note below.

    The gate uses a single signal: the distance from the HAM10000 training
    manifold in the deployed model's own feature space.

    The MobileNetV2 second opinion is deliberately NOT part of the decision.
    It is a *cancer* classifier, not a lesion-presence detector, and measured
    on 100 real photographs (30 genuine lesion photographs, 70 ordinary
    photographs spanning posters, documents, screenshots, classrooms, buildings,
    scenery, furniture, books, electronics, animals, food, vehicles, selfies
    and body photographs) its probability points the wrong way:

        genuine lesion photographs : median 0.074
        ordinary photographs       : median 0.123

    Refusing an image whose second opinion was below MOBILENET_THRESHOLD
    therefore refused genuine lesions preferentially: 6 of the 8 genuine
    lesions inside the ambiguous band were refused, including two images the
    deployed model itself calls highly malignant (ConvNeXt probability 0.82 and
    0.92). That rule was the dominant source of false rejections, not the hard
    cut-off.

    Measured error rates on the same 100 photographs (FR = genuine lesion
    wrongly refused, FA = ordinary photograph wrongly accepted, FC = ordinary
    photograph accepted and reported as cancer):

        old rule (>max, or band and second opinion below threshold)
                                              FR 23.3%  FA 17.1%  FC 0
        this rule (distance only)                FR  6.7%  FA 12.9%  FC 0

    Nothing here changes the model, its weights, the 0.525 decision threshold
    or any stored prediction array. The gate is a separate, additive safety
    layer.
    """
    if score is None:
        return False

    # Deep inside the lesion domain: always analyse. Every HAM10000
    # validation and test lesion scores at most 0.636.
    if score <= config.DOMAIN_SUSPICIOUS_DISTANCE:
        return False

    # Everything above the split point is treated as outside the lesion
    # domain. The split is the midpoint of the ambiguous band, derived from
    # DOMAIN_SUSPICIOUS_DISTANCE and DOMAIN_MAX_DISTANCE.
    return score > config.DOMAIN_BAND_SPLIT


def check_domain(image: Image.Image,
                 second_opinion: Optional[Callable[[Image.Image], Optional[float]]] = None
                 ) -> dict:
    """Evaluate the input-domain gate for a single image.

    Args:
        image: PIL image.
        second_opinion: optional callable(image) -> cancer probability from an
            independent model. It is ADVISORY ONLY: the value is reported in
            the result for auditing but it does not influence the decision,
            because it is not a valid lesion-presence signal (see
            is_out_of_domain). It must return None if it cannot produce an
            answer.

    Returns:
        dict with keys: in_domain (bool), score (float|None),
                        reason (str|None), second_opinion_cancer_prob (float|None)
    """
    if not config.DOMAIN_GATE_ENABLED:
        return {"in_domain": True, "score": None, "reason": None,
                "second_opinion_cancer_prob": None}

    score = domain_score(image)
    if score is None:
        return {"in_domain": True, "score": None, "reason": None,
                "second_opinion_cancer_prob": None}

    # The decision is made from the distance alone.
    out = is_out_of_domain(score)

    # The second opinion is still computed for images near the boundary so it
    # is available for auditing, but it can no longer refuse an image.
    second_prob = None
    if second_opinion is not None and score > config.DOMAIN_SUSPICIOUS_DISTANCE:
        try:
            second_prob = second_opinion(image)
        except Exception as exc:                      # pragma: no cover - helper failure
            log.warning("Second opinion failed (%s) - advisory value only", exc)
            second_prob = None

    if out:
        log.info("Input-domain gate rejected image (score=%.4f)", score)

    return {
        "in_domain": not out,
        "score": round(score, 4),
        "reason": REASON_OUTSIDE if out else None,
        "second_opinion_cancer_prob": (round(second_prob, 4)
                                       if second_prob is not None else None),
    }


def check_domains(images: List[Image.Image],
                  second_opinion: Optional[Callable[[Image.Image], Optional[float]]] = None
                  ) -> List[dict]:
    """Evaluate the gate for a batch of images (one forward pass for all views).

    The decision is made from the distance alone; the second opinion is an
    advisory diagnostic only (see is_out_of_domain).
    """
    if not config.DOMAIN_GATE_ENABLED or not images:
        return [{"in_domain": True, "score": None, "reason": None,
                 "second_opinion_cancer_prob": None} for _ in images]

    scores = domain_scores(images)
    if not scores:
        return [{"in_domain": True, "score": None, "reason": None,
                 "second_opinion_cancer_prob": None} for _ in images]

    results = []
    for img, score in zip(images, scores):
        out = is_out_of_domain(score)
        second_prob = None
        if second_opinion is not None and score > config.DOMAIN_SUSPICIOUS_DISTANCE:
            try:
                second_prob = second_opinion(img)
            except Exception as exc:                  # pragma: no cover - helper failure
                log.warning("Second opinion failed (%s) - advisory value only", exc)
                second_prob = None
        results.append({
            "in_domain": not out,
            "score": round(score, 4),
            "reason": REASON_OUTSIDE if out else None,
            "second_opinion_cancer_prob": (round(second_prob, 4)
                                           if second_prob is not None else None),
        })
    return results


# ──────────────────────────────────────────────────────────────────────
# Response payload
# ──────────────────────────────────────────────────────────────────────

NOT_ANALYSABLE_TEXT = (
    "This image cannot be analysed: it does not look like a close-up photo of "
    "a skin lesion. No result was produced. Please upload a photo of the skin "
    "area that fills the frame, in focus and in good light."
)


def not_analysable_result(domain_info: dict) -> dict:
    """Build the API payload for an image that is outside the lesion domain.

    Deliberately carries no probability, no confidence and no risk level: the
    model has nothing meaningful to say about this image and a consumer must
    not be able to mistake the payload for a diagnosis.
    """
    return {
        "success": True,
        "analysable": False,
        "verdict": "not_analysable",
        "prediction": "Invalid Image — Please upload a suitable skin-lesion image",
        "cancer_probability": None,
        "cancer_detected": None,
        "confidence": None,
        "risk_level": "unknown",
        "heatmap": "",
        "domain_in_domain": False,
        "domain_score": domain_info.get("score"),
        "domain_reason": domain_info.get("reason"),
    }
