# DermaAI — Skin Cancer Detection

A lightweight Flask web application for AI-powered skin lesion classification. Upload a dermoscopic or clinical image of a skin lesion and receive an instant binary classification (Cancer Detected / Not Detected) with a confidence score and a Grad-CAM attention heatmap.

## Problem statement

Skin cancer is highly treatable when caught early, but timely access to dermatoscopic screening is limited in many settings. DermaAI addresses this gap as a decision-support prototype that lets a user upload a photo of a skin lesion and receive an immediate, explainable binary indication (cancer vs. non-cancer) with a confidence score and a Grad-CAM attention heatmap. It is intended to support — never replace — professional dermatological evaluation.


> **Responsible-use disclaimer — read before use.** DermaAI is a student/research decision-support prototype. It is **not** a clinically validated diagnostic system, is **not** FDA-cleared, CE-marked, or approved for medical use, and must **not** be used to diagnose, treat, or triage any skin condition. Always consult a qualified dermatologist for any skin concern. Images are analysed in memory and are never stored or transmitted. The Grad-CAM heatmaps are interpretability aids only: their **lesion-localization reliability has not been established**, and they may highlight image artefacts (hair, ruler markings, background, borders) rather than the lesion itself.

## Features

- **Single-page interface** — drag-and-drop or click to upload
- **In-memory processing** — images are analysed and never stored
- **No authentication required** — no login, no user accounts, no database
- **Grad-CAM explainability** — attention heatmap overlaid on the resized photo
- **Input-domain gate** — refuses ordinary photographs (classroom scenes, cars, food)
  that the lesion-only model would otherwise report as a skin lesion
- **EXIF orientation handling** — a photo taken in portrait is analysed the way the
  browser displays it, not rotated
- **Real-time confidence visualization** — animated confidence bar with color-coded results
- **Responsive design** — works on desktop and mobile browsers
- **Medical disclaimer included** — clear decision-support warning

## Technology Stack

| Layer | Technology |
|-------|------------|
| Backend | Flask 3.1 |
| ML Framework | PyTorch 2.11 (ConvNeXt-Tiny), TensorFlow / Keras 2.20 (MobileNetV2) |
| Models | ConvNeXt-Tiny (default), MobileNetV2 (legacy) |
| Input Resolution | 224 × 224 RGB |
| Frontend | Vanilla HTML5, CSS3, ES6 JavaScript |
| Styling | Custom CSS (no frameworks) |

## System architecture

DermaAI is a single-page Flask application. The backend (`app.py`) loads one of two model backends at startup, selected by `config.DEFAULT_MODEL`:

- **ConvNeXt-Tiny (PyTorch)** — `inference.py` defines the architecture (ConvNeXt-Tiny backbone + global average pooling + dropout + linear head with ImageNet normalisation as an in-graph layer), loads the PyTorch checkpoint, and produces predictions plus Grad-CAM heatmaps from the final spatial feature map. The heatmap gradient is taken with `torch.autograd.grad` on that feature map, so nothing is registered on the shared model and concurrent requests cannot contaminate each other's heatmap.
- **MobileNetV2 (TensorFlow/Keras)** — `app.py` loads the Keras `.h5` model and runs prediction with temperature-scaled calibration plus Grad-CAM from the last Conv2D layer.

Both backends share `config.py` (paths, thresholds, input size, risk bands) and `quality.py` (lightweight blur/dark/overexposure screening that runs on decoded uploads and never rejects a decodable image). A third module, `domain.py`, adds the input-domain gate: it measures how far an upload sits from the HAM10000 training manifold in the deployed model's own 768-d feature space and refuses images that are not dermoscopic skin lesions, so no diagnosis is reported for an ordinary photograph. The frontend (`templates/index.html` + `static/`) uploads the image to `/predict`, displays the verdict/confidence/risk, shows quality warnings, and renders the Grad-CAM heatmap side-by-side with the resized photo.

## Project directory structure

```
skin-cancer-app/
├── app.py                      # Flask application entry point (+ routes)
├── config.py                   # Single source of truth: model, paths, thresholds
├── inference.py                # ConvNeXt-Tiny PyTorch backend (inference + Grad-CAM)
├── quality.py                  # Image-quality screening (blur/dark/overexposure)
├── domain.py                   # Input-domain gate (refuses non-lesion photographs)
├── requirements.txt            # Python dependencies
├── model/
│   ├── skin_cancer_mobilenetv2.h5   # Trained MobileNetV2 model (~22 MB)
│   ├── model_card.json              # Model provenance, thresholds, metrics
│   └── threshold.txt                # MobileNetV2 decision threshold (0.22)
├── results/convnext_tiny/      # ConvNeXt-Tiny checkpoints, metrics, curves
│   ├── convnext_tiny_best.pt        # ConvNeXt-Tiny PRIMARY checkpoint (stage-2 epoch 7, ~111 MB)
│   ├── domain_reference.npy         # HAM10000 train-split embeddings for the domain gate (~11 MB)
│   ├── domain_reference_meta.json   # Domain-gate provenance + calibration record
│   └── checkpoints/stage2_epoch13.pt # ConvNeXt-Tiny FALLBACK checkpoint (stage-2 epoch 13, ~324 MB)
├── scripts/
│   └── build_domain_reference.py    # Rebuilds domain_reference.npy + calibration record
├── test-images/                # External clinical photos used by the regression tests
├── tests/                      # pytest suite (test_quality.py, test_domain_gate.py,
│                               # test_exif_gradcam.py, test_regression.py)
├── templates/
│   └── index.html              # Single-page frontend
├── static/
│   ├── css/style.css           # Styling
│   └── js/app.js               # Frontend logic
└── README.md
```

## Installation

### Prerequisites
- Python 3.10 – 3.13
- pip (or conda)

### Using venv (recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/dermaai.git
cd dermaai

# Create and activate virtual environment
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# macOS / Linux / Git Bash
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Using conda

```bash
conda create -n dermaai python=3.10 -y
conda activate dermaai
pip install -r requirements.txt
```

## Running the Application

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

The server will start on port 5000. The model loads automatically at startup (first run may take a few seconds). The active backend is `DEFAULT_MODEL` in `config.py` (`"convnext_tiny"` or `"mobilenetv2"`); edit it and restart to switch.

## API Reference

### `POST /predict`

Classify a skin lesion image.

**Request** (multipart/form-data)

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | PNG, JPG or JPEG (max 10 MB) |

**Response** (JSON)

```json
{
  "success": true,
  "analysable": true,
  "cancer_detected": false,
  "verdict": "not_affected",
  "prediction": "Benign (No Cancer Detected)",
  "confidence": 0.9737,
  "cancer_probability": 0.0263,
  "risk_level": "low",
  "domain_in_domain": true,
  "domain_score": 0.4646,
  "heatmap": "<base64 PNG>"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `analysable` | boolean | `false` when the image is outside the lesion domain and no result was produced |
| `cancer_detected` | boolean | `true` if calibrated probability >= model threshold; `null` when `analysable` is `false` |
| `verdict` | string | `affected`, `not_affected` or `not_analysable` |
| `prediction` | string | Human-readable result |
| `confidence` | float | Confidence for the returned class (0–1); `null` when `analysable` is `false` |
| `cancer_probability` | float | Calibrated probability of malignancy (0–1); `null` when `analysable` is `false` |
| `risk_level` | string | `low`, `medium`, `high`, or `unknown` when `analysable` is `false` |
| `domain_in_domain` | boolean | `false` if the input-domain gate refused the image |
| `domain_score` | float | Distance of the image from the HAM10000 training manifold (higher = further out) |
| `heatmap` | string | Grad-CAM overlay, base64-encoded PNG (224 × 224) |

### Input-domain gate

The model is trained only on HAM10000 dermoscopic images and has no "this is not a skin
image" class, so ordinary photographs that contain a dark blob on a light field (fruit
in a bowl, a car, a classroom scene) used to be scored as malignant lesions. Before any
prediction is reported, the backend measures how far the image sits from the HAM10000
training manifold in the model's own feature space and refuses images that are clearly
outside it. The decision uses that distance alone — the MobileNetV2 second opinion is
reported for auditing but can no longer refuse or admit an image, because it is a
*cancer* classifier and was measured to point the wrong way (its probability is lower on
genuine lesion photographs than on ordinary photographs).

A refused image produces **no verdict and no probability**:

```json
{
  "success": true,
  "analysable": false,
  "verdict": "not_analysable",
  "prediction": "Invalid Image — Please upload a suitable skin-lesion image",
  "cancer_probability": null,
  "cancer_detected": null,
  "confidence": null,
  "risk_level": "unknown",
  "domain_in_domain": false,
  "domain_score": 0.8556,
  "domain_reason": "outside_lesion_domain",
  "heatmap": ""
}
```

Measured on 100 real photographs (30 genuine skin-lesion photographs and 70 ordinary
photographs spanning posters, documents, screenshots, classrooms, buildings, scenery,
furniture, books, electronics, animals, food, vehicles, selfies and body photographs):
**61 of the 70 ordinary photographs are refused and none of them is reported as cancer**
(9 are analysed and reported as *not affected*), while **28 of the 30 genuine lesion
photographs are analysed**. All 3005 held-out HAM10000 lesions and all 14 genuine lesion
photographs of the project test set keep their original verdict. The gate can be turned
off with `DOMAIN_GATE_ENABLED = False` in `config.py`; the reference matrix can be
rebuilt with `python scripts/build_domain_reference.py`. See
`DERMAI_ODD_FALSE_POSITIVE_REPORT.md` for the full investigation.

### Image orientation (EXIF)

Phone cameras store the sensor data unrotated and record the intended viewing
direction in the EXIF orientation tag; browsers apply that tag when they display
the photo. `inference.apply_exif_orientation()` applies the same tag on the
server, so the picture is analysed in the orientation the user actually sees.

Without it, a portrait photo was analysed rotated by 90 degrees: the melanoma
test photo scored **0.6885** upright and **0.1194** when the same picture was
stored sideways, i.e. the verdict flipped from *Affected* to *Not Affected*, and
the Grad-CAM overlay did not line up with the photo on screen. With the tag
honoured, the rotated upload returns the same probability (0.687 / 0.6987 /
0.6851 for orientations 6 / 3 / 8) as the upright one.

Images without an orientation tag are passed through untouched (no copy, no
pixel change), so the HAM10000 data and the published metrics are unaffected.

**Error Responses**

| Status | Cause |
|--------|-------|
| 400 | Missing file, empty filename, or unsupported format |
| 413 | File exceeds 10 MB limit |
| 503 | Model not loaded on server |
| 500 | Prediction failed |

## Model Information

| | ConvNeXt-Tiny (default) | MobileNetV2 (legacy) |
|---|---|---|
| Framework | PyTorch | TensorFlow / Keras |
| Checkpoint | `results/convnext_tiny/convnext_tiny_best.pt` (stage-2 epoch 7) | `model/skin_cancer_mobilenetv2.h5` |
| Input | 224×224 RGB, raw [0,255] (ImageNet norm inside graph) | 224×224 RGB, [0,1] |
| Decision threshold | 0.525 (argmax F1 on validation) | 0.22 (argmax F1 on validation) |
| Calibration | none (raw sigmoid probability) | temperature scaling, T = 1.0 (identity) |
| Grad-CAM target | model.features (final spatial feature map before global average pooling) | last Conv2D of MobileNetV2 (Conv_1) |

All paths, thresholds and switches live in `config.py`.

### Model checkpoint requirements

To run the application you need the trained checkpoint(s) for the backend you select:

| Backend | Checkpoint | Size | How to obtain |
|---------|-----------|------|---------------|
| MobileNetV2 (legacy) | `model/skin_cancer_mobilenetv2.h5` | ~22 MB | Committed to the repository (under GitHub's 100 MB file limit). |
| ConvNeXt-Tiny (default, primary) | `results/convnext_tiny/convnext_tiny_best.pt` | ~111 MB | **Exceeds GitHub's 100 MB hard file limit.** Must be fetched via Git LFS or obtained separately (see below). |
| ConvNeXt-Tiny (default, fallback) | `results/convnext_tiny/checkpoints/stage2_epoch13.pt` | ~324 MB | **Exceeds GitHub's 100 MB hard file limit.** Fallback: `stage2_epoch13.pt` is used automatically if the primary is absent. Its weights are not the ones behind the published metrics. |

`inference.load_convnext_model()` tries `convnext_tiny_best.pt` (stage-2 epoch 7) first and falls back to `checkpoints/stage2_epoch13.pt` (stage-2 epoch 13), so the default ConvNeXt-Tiny backend works with just the primary checkpoint present. The primary checkpoint is the one that produced every published ConvNeXt-Tiny metric and the 0.525 decision threshold, so the deployed application and the reported results describe the same weights.

**ConvNeXt checkpoint (>100 MB) options:**
1. **Git LFS** — this repository tracks large checkpoints with Git LFS. After cloning, run `git lfs install` then `git lfs pull` to download the checkpoint files. GitHub Free includes 1 GB of LFS storage and 1 GB of bandwidth per month; confirm this is sufficient before enabling LFS, as additional usage may require a paid plan.
2. **Manual / separate transfer** — obtain the `.pt` checkpoint from your training output or a shared location and place it at the path above before running the app. The application will report a clear error (`503` / startup log) if no checkpoint is found.

Do not commit passwords, API keys, tokens, or patient data. No patient data or PHI should be added to `test-images/` or anywhere else in the repository.

## Testing

Automated regression suite for both models — single-image prediction,
batch prediction, Grad-CAM validity, heatmap dimension/registration
checks, input validation and the `/health` endpoint:

```bash
pytest tests/
```

The suite has four files:

| File | Covers |
|---|---|
| `tests/test_regression.py` | both models: prediction, batch, heatmap validity and registration, error handling |
| `tests/test_quality.py` | blur / dark / overexposure screening and the API quality fields |
| `tests/test_domain_gate.py` | the input-domain gate: calibration, refusal payload, in-domain behaviour |
| `tests/test_exif_gradcam.py` | EXIF orientation handling and Grad-CAM thread safety |

Equivalent standalone runner (no pytest required):

```bash
python tests/test_regression.py
```

Output uses plain ASCII markers (`[PASS]` / `[FAIL]` / `[ERROR]` /
`[WARN]`) so it works on Windows consoles with cp1252/cp850 codepages.
The script exits `0` only when every test passes, and `1` if any test
fails or errors (failures are listed individually in the summary).

### Verified results (2026-10-20, Python 3.13, torch 2.11.0+cu128, TF 2.20.0)

```
pytest tests/ -q
73 passed, 17 warnings in 93.85s (exit code 0)
  tests/test_regression.py    38 tests   (both models, prediction, batch,
                                         heatmap validity, error handling)
  tests/test_domain_gate.py   10 tests   (gate calibration, decision rules,
                                         second-opinion independence, refusal
                                         payload, in-domain behaviour)
  tests/test_quality.py       18 tests   (blur / dark / overexposure)
  tests/test_exif_gradcam.py   7 tests   (EXIF orientation, Grad-CAM safety)

python tests/test_regression.py
convnext_tiny  11 passed,  0 failed/errored
mobilenetv2    11 passed,  0 failed/errored
TOTAL: 22 passed, 0 failed, 0 errored, 0 warnings
REGRESSION SUITE: ALL TESTS PASSED  (exit code 0)
```

> Run the suite with the live server **stopped** on a low-memory machine. This
> host has 7.7 GB of RAM, and running the pytest suite while the Flask server
> holds the same models exhausted memory and produced a Windows access
> violation inside the PyTorch forward pass of the batch test. The crash is
> inside the ConvNeXt forward pass, which runs before the gate's decision
> logic; with the server stopped the same suite passes 73/73.

- Passed for **both** models: `/health`, single-image prediction on all 4
  test images (verdict + probability + decodable 224×224 heatmap),
  batch prediction (4 images), and all 6 error-handling cases
  (no file → 400, bad extension → 400, corrupted → 400, >10 MB → 413,
  >64 batch files → 400, >50 MB batch → 413).
- Heatmap alignment: every returned heatmap is 224×224 (matches the model
  input and the frontend preview canvas) **and** pixel-registered — the
  blended overlay correlates with the resized original at zero shift and
  not with a shifted copy (verified for all 8 image/model combinations).
- One non-fatal `[WARN]`: on the melanoma photo the MobileNetV2 heatmap
  concentrates near the bottom-right corner (hot-pixel centroid (0.75, 0.83),
  corner-frame share 0.59 vs ≈0.44 for uniform attention). See the
  corner-activation limitation below.

The same checks were also verified over real HTTP (Flask dev server,
werkzeug multipart handling) rather than only the Flask test client, and
the frontend was verified to serve and to draw the preview and heatmap
side by side at 224×224.

## Medical Disclaimer

> **This tool is for decision support only and is not a medical diagnosis.**  
> Always consult a qualified dermatologist for any skin concern.  
> Images are analysed in memory and are never stored or transmitted.

## Limitations

- Binary classification only (malignant vs. benign) — no lesion type identification
- Trained on dermoscopic images; clinical photos may reduce accuracy
- The input-domain gate refuses images outside the HAM10000 distribution. It can refuse
  a genuine lesion photo if that photo also contains enough of a non-skin scene to move
  it off the manifold, and it lets a small number of ordinary photographs through when
  they are indistinguishable from real lesion photographs in this feature space (they
  are then reported as benign, never as cancer). Measured on 30 genuine lesion
  photographs and 70 ordinary photographs, **2 genuine lesions are refused** (6.7%) and
  **9 ordinary photographs are analysed** (12.9%) — none of them reported as cancer.
  The weakest categories are scanned documents (1 of 5 refused), posters (3 of 5) and
  books/screenshots (4 of 5), because a flat page of text is genuinely close to the
  dermoscopic manifold in this feature space. See `DERMAI_ODD_FALSE_POSITIVE_REPORT.md`
  §5 and §8, and `DERMAI_UNRESOLVED_ISSUES.md` (ISSUE-20).
- Thresholds (0.525 ConvNeXt-Tiny / 0.22 MobileNetV2) are fixed operating
  points chosen on one validation split; clinical validation required
- No uncertainty quantification beyond the confidence score
- CPU-only inference for MobileNetV2 (TensorFlow GPU not supported on native
  Windows ≥ 2.11); ConvNeXt-Tiny uses CUDA when available and falls back to CPU
- **Suspected corner-activation limitation (Grad-CAM):** matching dimensions between the preview and the heatmap (both 224×224) prove the overlay is aligned, but they do **not** prove the model attends to the lesion. **The Grad-CAM lesion-localization reliability has not been established.** Measured on the test images, the MobileNetV2 heatmap for the melanoma photo concentrates near the image corners/borders (hot-pixel centroid (0.75, 0.83), corner-frame share 0.59 vs ≈0.44 expected for uniform attention), which suggests the model may respond to artefacts (e.g. rulers, markings, background) rather than the lesion itself. Treat the heatmap as an interpretability aid, not evidence of correct reasoning. ConvNeXt-Tiny heatmaps on the same images had hot-pixel centroids near the image centre (all within 0.27 of centre), though centrality alone likewise does not prove lesion localization.
- Test images in `test-images/` are external clinical photos and are
  **not** validated ground truth; the regression suite verifies mechanics
  (endpoints, formats, thresholds, heatmap validity), not clinical accuracy.

## Example Usage

```bash
# Using curl
curl -X POST -F "file=@lesion.jpg" http://localhost:5000/predict

# Using Python requests
import requests
with open('lesion.jpg', 'rb') as f:
    resp = requests.post('http://localhost:5000/predict', files={'file': f})
print(resp.json())
```

## License

This project is provided for educational and research purposes.  
Model weights and code may be subject to separate licensing — verify before commercial use.

## Contributing

Issues and pull requests are welcome. Please ensure:

1. No patient data or PHI is committed
2. Model files remain under GitHub's 100 MB file limit
3. Medical disclaimer is preserved
4. Changes are tested locally before submitting