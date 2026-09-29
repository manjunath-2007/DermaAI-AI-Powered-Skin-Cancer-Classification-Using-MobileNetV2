# DermaAI — Skin Cancer Detection

A lightweight Flask web application for AI-powered skin lesion classification. Upload a dermoscopic or clinical image of a skin lesion and receive an instant binary classification (Cancer Detected / Not Detected) with a confidence score.

## Features

- **Single-page interface** — drag-and-drop or click to upload
- **In-memory processing** — images are analysed and never stored
- **No authentication required** — no login, no user accounts, no database
- **Real-time confidence visualization** — animated confidence bar with color-coded results
- **Responsive design** — works on desktop and mobile browsers
- **Medical disclaimer included** — clear decision-support warning

## Technology Stack

| Layer | Technology |
|-------|------------|
| Backend | Flask 3.1 |
| ML Framework | TensorFlow / Keras 2.21 |
| Model | MobileNetV2 (transfer learning) |
| Input Resolution | 224 × 224 RGB |
| Frontend | Vanilla HTML5, CSS3, ES6 JavaScript |
| Styling | Custom CSS (no frameworks) |

## Project Structure

```
skin-cancer-app/
├── app.py                      # Flask application entry point
├── requirements.txt            # Python dependencies
├── model/
│   └── skin_cancer_mobilenetv2.h5   # Trained MobileNetV2 model (~22 MB)
├── templates/
│   └── index.html              # Single-page frontend
├── static/
│   ├── css/
│   │   └── style.css           # Styling
│   └── js/
│       └── app.js              # Frontend logic
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

The server will start on port 5000. The model loads automatically at startup (first run may take a few seconds).

## API Reference

### `POST /predict`

Classify a skin lesion image.

**Request** (multipart/form-data)

| Field | Type | Description |
|-------|------|-------------|
| `image` | File | PNG, JPG, JPEG, BMP, or TIFF (max 16 MB) |

**Response** (JSON)

```json
{
  "cancer_detected": true,
  "label": "Cancer Detected",
  "confidence": 87.5
}
```

| Field | Type | Description |
|-------|------|-------------|
| `cancer_detected` | boolean | `true` if probability > threshold (0.23) |
| `label` | string | Human-readable result |
| `confidence` | float | Confidence percentage (0–100) |

**Error Responses**

| Status | Cause |
|--------|-------|
| 400 | Missing file, empty filename, or unsupported format |
| 413 | File exceeds 16 MB limit |
| 503 | Model not loaded on server |
| 500 | Prediction failed |

## Model Information

- **Architecture**: MobileNetV2 with custom classification head
- **Training**: Transfer learning on dermoscopic image dataset
- **Input**: 224 × 224 × 3 (RGB), normalized to [0, 1]
- **Output**: Single sigmoid unit → probability of malignancy
- **Decision Threshold**: 0.23 (configurable via `THRESHOLD` in `app.py`)
- **File Size**: ~22 MB (commits directly, no Git LFS required)

The model file `model/skin_cancer_mobilenetv2.h5` is included in the repository.

## Medical Disclaimer

> **This tool is for decision support only and is not a medical diagnosis.**  
> Always consult a qualified dermatologist for any skin concern.  
> Images are analysed in memory and are never stored or transmitted.

## Limitations

- Binary classification only (malignant vs. benign) — no lesion type identification
- Trained on dermoscopic images; clinical photos may reduce accuracy
- Threshold (0.23) is a fixed operating point; clinical validation required
- No uncertainty quantification beyond confidence score
- CPU-only inference (TensorFlow GPU not supported on native Windows ≥ 2.11)

## Example Usage

```bash
# Using curl
curl -X POST -F "image=@lesion.jpg" http://localhost:5000/predict

# Using Python requests
import requests
with open('lesion.jpg', 'rb') as f:
    resp = requests.post('http://localhost:5000/predict', files={'image': f})
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