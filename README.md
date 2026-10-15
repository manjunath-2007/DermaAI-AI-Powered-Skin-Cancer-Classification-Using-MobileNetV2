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

- **ConvNeXt-Tiny (PyTorch)** — `inference.py` defines the architecture (ConvNeXt-Tiny backbone + global average pooling + dropout + linear head with ImageNet normalisation as an in-graph layer), loads the PyTorch checkpoint, and produces predictions plus Grad-CAM heatmaps via hooks on the final spatial feature map.
- **MobileNetV2 (TensorFlow/Keras)** — `app.py` loads the Keras `.h5` model and runs prediction with temperature-scaled calibration plus Grad-CAM from the last Conv2D layer.

Both backends share `config.py` (paths, thresholds, input size, risk bands) and `quality.py` (lightweight blur/dark/overexposure screening that runs on decoded uploads and never rejects a decodable image). The frontend (`templates/index.html` + `static/`) uploads the image to `/predict`, displays the verdict/confidence/risk, shows quality warnings, and renders the Grad-CAM heatmap side-by-side with the resized photo.

## Project directory structure

```
skin-cancer-app/
├── app.py                      # Flask application entry point (+ routes)
├── config.py                   # Single source of truth: model, paths, thresholds
├── inference.py                # ConvNeXt-Tiny PyTorch backend (inference + Grad-CAM)
├── quality.py                  # Image-quality screening (blur/dark/overexposure)
├── test_regression.py          # Standalone regression runner (both models)
├── requirements.txt            # Python dependencies
├── model/
│   ├── skin_cancer_mobilenetv2.h5   # Trained MobileNetV2 model (~22 MB)
│   ├── model_card.json              # Model provenance, thresholds, metrics
│   └── threshold.txt                # MobileNetV2 decision threshold (0.22)
├── results/convnext_tiny/      # ConvNeXt-Tiny checkpoints, metrics, curves
│   ├── convnext_tiny_best.pt        # ConvNeXt-Tiny best checkpoint (~111 MB)
│   └── checkpoints/stage2_epoch13.pt # ConvNeXt-Tiny stage-2 checkpoint (~324 MB)
├── test-images/                # External clinical photos used by the regression tests
├── tests/                      # pytest suite (test_quality.py, test_regression.py)
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
  "cancer_detected": false,
  "verdict": "not_affected",
  "prediction": "Benign (No Cancer Detected)",
  "confidence": 0.9737,
  "cancer_probability": 0.0263,
  "risk_level": "low",
  "heatmap": "<base64 PNG>"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `cancer_detected` | boolean | `true` if calibrated probability >= model threshold |
| `verdict` | string | `affected` or `not_affected` |
| `prediction` | string | Human-readable result |
| `confidence` | float | Confidence for the returned class (0–1) |
| `cancer_probability` | float | Calibrated probability of malignancy (0–1) |
| `risk_level` | string | `low`, `medium` or `high` |
| `heatmap` | string | Grad-CAM overlay, base64-encoded PNG (224 × 224) |

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
| Checkpoint | `results/convnext_tiny/checkpoints/stage2_epoch13.pt` | `model/skin_cancer_mobilenetv2.h5` |
| Input | 224×224 RGB, raw [0,255] (ImageNet norm inside graph) | 224×224 RGB, [0,1] |
| Decision threshold | 0.525 (argmax F1 on validation) | 0.22 (argmax F1 on validation) |
| Calibration | none (raw sigmoid probability) | temperature scaling, T = 1.0 (identity) |
| Grad-CAM target | last Conv2d of ConvNeXt stage 4 | last Conv2D of MobileNetV2 |

All paths, thresholds and switches live in `config.py`.

### Model checkpoint requirements

To run the application you need the trained checkpoint(s) for the backend you select:

| Backend | Checkpoint | Size | How to obtain |
|---------|-----------|------|---------------|
| MobileNetV2 (legacy) | `model/skin_cancer_mobilenetv2.h5` | ~22 MB | Committed to the repository (under GitHub's 100 MB file limit). |
| ConvNeXt-Tiny (default) | `results/convnext_tiny/convnext_tiny_best.pt` | ~111 MB | **Exceeds GitHub's 100 MB hard file limit.** Must be fetched via Git LFS or obtained separately (see below). |
| ConvNeXt-Tiny (default, primary) | `results/convnext_tiny/checkpoints/stage2_epoch13.pt` | ~324 MB | **Exceeds GitHub's 100 MB hard file limit.** Fallback: `convnext_tiny_best.pt` is used automatically if this is absent. |

`inference.load_convnext_model()` tries `stage2_epoch13.pt` first and falls back to `convnext_tiny_best.pt`, so the default ConvNeXt-Tiny backend works with just the best checkpoint present.

**ConvNeXt checkpoint (>100 MB) options:**
1. **Git LFS** — this repository tracks large checkpoints with Git LFS. After cloning, run `git lfs install` then `git lfs pull` to download the checkpoint files. GitHub Free includes 1 GB of LFS storage and 1 GB of bandwidth per month; confirm this is sufficient before enabling LFS, as additional usage may require a paid plan.
2. **Manual / separate transfer** — obtain the `.pt` checkpoint from your training output or a shared location and place it at the path above before running the app. The application will report a clear error (`503` / startup log) if no checkpoint is found.

Do not commit passwords, API keys, tokens, or patient data. No patient data or PHI should be added to `test-images/` or anywhere else in the repository.

## Testing

Automated regression suite for both models — single-image prediction,
batch prediction, Grad-CAM validity, heatmap dimension/registration
checks, input validation and the `/health` endpoint:

```bash
python test_regression.py
```

Output uses plain ASCII markers (`[PASS]` / `[FAIL]` / `[ERROR]` /
`[WARN]`) so it works on Windows consoles with cp1252/cp850 codepages.
The script exits `0` only when every test passes, and `1` if any test
fails or errors (failures are listed individually in the summary).

### Verified results (2026-10-13, Python 3.13, torch 2.11.0+cu128, TF 2.20.0)

```
convnext_tiny   11 passed,  0 failed/errored
mobilenetv2     11 passed,  0 failed/errored
TOTAL: 22 passed, 0 failed, 0 errored, 1 warning
REGRESSION SUITE: ALL TESTS PASSED  (exit code 0)
```

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