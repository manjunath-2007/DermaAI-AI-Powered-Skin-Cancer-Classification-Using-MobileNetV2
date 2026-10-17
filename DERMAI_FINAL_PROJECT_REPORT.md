# DermaAI — Final Project Report

**Project:** DermaAI — AI-Powered Skin Cancer Classification Using MobileNetV2 and ConvNeXt-Tiny  
**Local root:** C:\Users\Manjunath\OneDrive\Desktop\MINI3\skin-cancer-app-simple\skin-cancer-app\  
**GitHub:** https://github.com/manjunath-2007/DermaAI-AI-Powered-Skin-Cancer-Classification-Using-MobileNetV2.git  
**Current commit:** 94b4af0b15c1f4dd550580e12dafb87fe3211635  
**Branch:** main (up to date with origin/main)  
**Report date:** 2026-10-17  
**Python:** 3.13.3 | **Flask:** 3.1.3 | **PyTorch:** 2.11.0+cu128 | **TensorFlow:** 2.20.0  
**Purpose:** Consolidated technical source document for updating the research paper and project paperwork.

---

## Evidence labels used

- **VERIFIED BY EXECUTION** — directly tested in this session (test suites run against the live codebase).
- **SUPPORTED BY EXISTING ARTIFACTS** — backed by files present in the repository (JSON logs, model_card, metrics, source code).
- **INFERRED FROM CODE** — derived by reading source; not independently executed.
- **NOT VERIFIED** — claimed nowhere in artifacts; no evidence found.
- **BLOCKED** — cannot be completed because a specific dependency/data is missing.

---

## 1. Abstract

DermaAI is a Flask-based web application for binary skin-lesion classification (Cancer vs Non-Cancer) using two model backends: MobileNetV2 (TensorFlow/Keras) and ConvNeXt-Tiny (PyTorch). The application provides single-image and batch prediction endpoints, image-quality screening (blur/dark/overexposure warnings), Grad-CAM heatmap generation for interpretability, and client-side PDF report generation via jsPDF. Models were trained on HAM10000 with a lesion-level stratified 70/15/15 split (random_state=42). The system is a research/demo decision-support tool and is not a substitute for professional medical diagnosis.

**Software test results (VERIFIED BY EXECUTION, this session):**
- pytest suite: 56 passed, 36 warnings (0 failed).
- Standalone regression suite: 22 passed, 0 failed, 0 errored.

**Model-performance metrics (SUPPORTED BY EXISTING ARTIFACTS, not re-derived):**
- MobileNetV2: validation F1 0.645 @ threshold 0.22; test F1 0.585 @ threshold 0.22; test ROC-AUC 0.862.
- ConvNeXt-Tiny: validation F1 0.698 @ threshold 0.525; test F1 0.630 @ threshold 0.525; test ROC-AUC 0.892.

---

## 2. Problem statement

Skin cancer (particularly melanoma) requires early and accurate detection for effective treatment. Visual inspection by clinicians is the primary screening method but is subject to human variability and limited specialist availability in many regions. There is a need for an accessible, decision-support tool that can triage skin-lesion images and flag potential malignancies for further dermatological review. DermaAI addresses this by providing a web-based image classifier with interpretability (Grad-CAM) and quality screening, intended as a decision aid rather than a diagnostic device.

---

## 3. Objectives

1. Implement a Flask web application serving two model backends (MobileNetV2 and ConvNeXt-Tiny) for binary skin-lesion classification.
2. Provide single-image and batch prediction via REST-like JSON endpoints (/predict, /predict_batch).
3. Apply image-quality screening (blur, dark, overexposure) as non-rejecting warnings.
4. Generate Grad-CAM heatmaps for single-image predictions as an interpretability aid.
5. Generate client-side PDF reports via jsPDF.
6. Document model evaluation metrics, limitations, and known issues without fabricating results.
7. Keep production model weights unchanged; do not retrain.

---

## 4. Project scope and limitations

**In scope:**
- Flask backend with /, /health, /predict, /predict_batch routes.
- Two inference backends: MobileNetV2 (Keras .h5) and ConvNeXt-Tiny (PyTorch .pt).
- Image-quality screening module (quality.py) with configurable thresholds.
- Grad-CAM for both models (single-image only; disabled for batch).
- Frontend (templates/index.html, static/css/style.css, static/js/app.js) with upload, preview, prediction display, heatmap overlay, and PDF download.
- Test suites: tests/test_regression.py and tests/test_quality.py.

**Out of scope / limitations:**
- No patient-level identity in HAM10000 artifacts; patient-level split separation cannot be assessed (BLOCKED).
- No external validation on an independent dataset (NOT VERIFIED).
- No confidence calibration applied to production models (MobileNetV2 temperature scaling investigated but not applied; ConvNeXt uses raw sigmoid) (SUPPORTED BY EXISTING ARTIFACTS).
- Image-quality thresholds are heuristic; no human-labelled calibration dataset found (SUPPORTED BY EXISTING ARTIFACTS).
- Grad-CAM lesion localization not established; MobileNetV2 shows corner-activation bias (SUPPORTED BY EXISTING ARTIFACTS + VERIFIED BY EXECUTION via test warnings).
- Browser-based PDF download not tested with a real browser (NOT VERIFIED).
- Application is a decision-support demo, not a medical device.

---

## 5. System architecture

**INFERRED FROM CODE** (app.py, inference.py, quality.py, config.py):

```
Browser (index.html + app.js + jsPDF)
        |
   HTTP multipart/form-data
        |
   Flask app (app.py)
   ├── /health              -> status, active_model, model_loaded
   ├── /predict             -> single image -> model backend + quality + Grad-CAM
   ├── /predict_batch       -> multiple images -> model backend + quality
   └── /                    -> render index.html with model_ready flag

Model backends:
   MobileNetV2 (app.py: mobilenet_* functions, TensorFlow/Keras .h5)
      -> mobilenet_preprocess (resize 224x224, /255 [0,1])
      -> mobilenet_predict_single / mobilenet_predict_batch
      -> mobilenet_grad_cam (last Conv2D layer, blended overlay)
      -> mobilenet_calibrated_prob (temperature scaling, T=1.0 = identity)

   ConvNeXt-Tiny (inference.py: PyTorch, .pt checkpoint)
      -> preprocess_image (resize 224x224, raw [0,255] RGB, ImageNet norm inside graph)
      -> predict_single / predict_batch
      -> generate_gradcam (model.features = final spatial feature map before GAP)

Quality module (quality.py):
   -> check_image_quality: blur (Laplacian variance), dark (mean brightness), overexposed (near-white ratio)
   -> Warnings only; never rejects a decodable image.

Configuration (config.py): single source of truth for paths, thresholds, model selection, Flask settings.
```

**Model ready check** (app.py:258): active model must be loaded before the frontend enables prediction.

---

## 6. Technology stack

**SUPPORTED BY EXISTING ARTIFACTS** (requirements.txt, source code, model_card.json, metrics.json):

| Layer | Technology |
|-------|-----------|
| Web framework | Flask 3.1.3 |
| CORS | flask_cors |
| MobileNetV2 backend | TensorFlow 2.20.0, Keras |
| ConvNeXt-Tiny backend | PyTorch 2.11.0+cu128, torchvision |
| Image processing | PIL (Pillow), OpenCV (cv2), NumPy |
| Frontend | HTML5, CSS, vanilla JavaScript, jspdf 2.5.1 (CDN) |
| Testing | pytest 9.1.1 |
| GPU | NVIDIA GeForce RTX 2050, 4 GB VRAM, CUDA 12.8 (per metrics.json) |
| Model files | model/skin_cancer_mobilenetv2.h5 (~21.83 MB), results/convnext_tiny/convnext_tiny_best.pt |

---

## 7. Project directory structure

**INFERRED FROM CODE** (directory listing of the project root):

```
skin-cancer-app/
├── app.py                         # Flask application + routes + MobileNetV2 inference
├── config.py                      # Single source of truth: paths, thresholds, Flask settings
├── inference.py                   # ConvNeXt-Tiny PyTorch backend (model def, load, predict, Grad-CAM)
├── quality.py                     # Image-quality screening (blur/dark/overexposure)
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation (modified during audit)
├── model/
│   ├── skin_cancer_mobilenetv2.h5 # Trained MobileNetV2 Keras model (~21.83 MB)
│   ├── model_card.json            # MobileNetV2 metrics, calibration, provenance
│   └── threshold.txt              # Threshold reference
├── results/
│   └── convnext_tiny/
│       ├── convnext_tiny_best.pt  # Best ConvNeXt-Tiny checkpoint
│       ├── checkpoints/           # Stage checkpoints (stage2_epoch13.pt, etc.)
│       ├── metrics.json           # ConvNeXt-Tiny training + test metrics
│       ├── test_proba.npy / test_true.npy  # Independent test predictions + labels
│       ├── training_log.txt, evaluation_log.txt
│       ├── classification_report.txt, confusion_matrix_*.csv, roc_curve_*.csv
│       └── val_threshold_sweep.csv, history_stage*.csv
├── tests/
│   ├── test_regression.py         # Standalone + pytest regression suite (both models)
│   └── test_quality.py            # Quality unit tests + API compatibility tests
├── test-images/                   # External clinical photos (NOT HAM10000 ground truth)
├── static/                        # CSS, JS
├── templates/                     # index.html (frontend + jsPDF)
├── scripts/                       # Calibration / utility scripts (untracked)
├── research/                      # Research artifacts (untracked)
├── venv/                          # Virtual environment
├── PROJECT_AUDIT.md               # Audit report (untracked)
├── PROJECT_CHANGELOG.md           # Changelog (untracked)
├── PROJECT_COMPLETION_REPORT.md   # Phase completion report (untracked)
├── DERMAI_FINAL_PROJECT_REPORT.md # This report (untracked)
└── regression_test_results.txt    # Historical regression output (untracked)
```

---

## 8. Dataset provenance

**SUPPORTED BY EXISTING ARTIFACTS** (model_card.json, metrics.json):

- Source: HAM10000 via `kagglehub skin-cancer-mnist-ham10000`.
- Label source: HAM10000_metadata.csv `dx` column (filenames never used as labels) — metrics.json.
- Dataset root (per logs): `D:\User\Downloads\archive` — not present in the project; path is local to the original training machine (INFERRED FROM CODE, not re-verified).
- Split: lesion-level stratified 70/15/15, random_state=42 (grouped by `lesion_id`).
- Split reproducibility: `split_reproduces_notebook_exactly: true` (model_card.json); `matches_production_split: true` for train/val/test (metrics.json).
- Counts: train 7010 images / 5229 lesions; val 1500 images / 1120 lesions; test 1505 images / 1121 lesions.
- Class counts (test): Non-Cancer 1217, Cancer 288.
- Lesion overlap between splits: 0 (metrics.json) — no lesion-level overlap.
- Test-images/ directory contains external clinical photographs; md5 + pixel + perceptual-hash comparison against all 10015 HAM10000 images concluded "Not from HAM10000" (model_card.json `known_provenance_test_images`). These filenames are NOT dataset ground truth and cannot be verified.

**Patient-level separation:** patient identifiers are not available in the provided artifacts; patient-level leakage cannot be assessed (BLOCKED). The completion report correctly makes no claim of patient-level separation.

---

## 9. Dataset labels and preprocessing

**Label mapping (shared by both models, SUPPORTED BY EXISTING ARTIFACTS):**
- Cancer (label 1): mel, bcc, akiec
- Non-Cancer (label 0): nv, bkl, vasc, df

**MobileNetV2 preprocessing (INFERRED FROM CODE, app.py:160):**
- `img.convert("RGB").resize((224,224))`, float32 / 255.0 → [0,1].
- Matches training preprocessing per model_card.json (`input_preprocessing` field).
- No ImageNet normalization inside graph; division by 255 only.

**ConvNeXt-Tiny preprocessing (INFERRED FROM CODE, inference.py:176):**
- `img.convert("RGB").resize((224,224))`, raw uint8 [0,255] RGB.
- ImageNet mean/std normalization is INSIDE the model graph (`self.mean`, `self.std` buffers).
- Config documents: "raw RGB [0, 255] -> ImageNet norm INSIDE model".

**Class balancing (metrics.json):** WeightedRandomSampler 50/50 on TRAIN split only.

---

## 10. Dataset splits and possible leakage

**SUPPORTED BY EXISTING ARTIFACTS:**

- Lesion-level stratified split with zero lesion overlap between splits (metrics.json `lesion_overlap_between_splits: 0`).
- No separate split files (train.txt/val.txt/test.txt) found in the project; split info is encoded in model_card.json and metrics.json (PROJECT_COMPLETION_REPORT.md).
- **Patient-level leakage:** NOT assessable — no patient-to-lesion mapping in artifacts (BLOCKED). Do not claim patient-level separation.
- **Duplicate images:** not explicitly checked beyond the zero-lesion-overlap statement (INFERRED FROM CODE).
- **Test-set integrity:** metrics.json states test set never used for threshold selection (model_card.json threshold note: "Test set never used"). ConvNeXt threshold selected on validation only via argmax F1 over grid 0.05..0.95 step 0.005.

---

## 11. MobileNetV2 implementation

**INFERRED FROM CODE** (app.py, model_card.json):

- Architecture: MobileNetV2 (ImageNet) + GlobalAveragePooling2D + Dense(256,relu) + Dropout + Dense(1,sigmoid).
- Framework: TensorFlow/Keras .h5 (~21.83 MB).
- Input: 224×224 RGB, [0,1] after /255.
- Output: single sigmoid logit; p=1 means Cancer.
- Threshold: config.py `MOBILENET_THRESHOLD = 0.22` (argmax F1 on validation).
- **DISCREPANCY:** model_card.json lists `"app_py_current_threshold": 0.5`, but config.py (current source) uses 0.22. The model_card.json value appears outdated relative to config.py. Code verified at config.py:36 = 0.22.
- Temperature scaling: `MOBILENET_TEMPERATURE = 1.0` (no scaling; identity). Calibration estimate T=1.2543 was computed on validation but NOT applied (model_card.json `calibration_estimate_not_applied`).
- Calibrated probability: `mobilenet_calibrated_prob()` applies sigmoid(logit(p)/T) with T=1.0 → returns raw probability unchanged.
- Grad-CAM: last Conv2D layer (Conv_1), blended overlay 0.55*orig + 0.45*heatmap, enabled via `ENABLE_GRADCAM_SINGLE=True`.
- Batch prediction: no Grad-CAM (`ENABLE_GRADCAM_BATCH=False`).

**Metrics (SUPPORTED BY EXISTING ARTIFACTS, model_card.json):**
| Split | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|----------|-----------|--------|----|---------|
| Validation | 0.22 | 0.832 | 0.554 | 0.771 | 0.645 | 0.895 |
| Test | 0.22 | 0.798 | 0.482 | 0.743 | 0.585 | 0.862 |
| Test @ 0.50 | 0.50 | 0.849 | 0.638 | 0.490 | 0.554 | 0.862 |

---

## 12. ConvNeXt-Tiny implementation

**INFERRED FROM CODE** (inference.py, metrics.json, config.py):

- Architecture: ConvNeXt-Tiny (torchvision, weights=None for backbone) + GAP + Dropout(0.30) + Linear(768,1). ImageNet normalization is a real layer inside the graph; feed raw RGB [0,255].
- Framework: PyTorch 2.11.0+cu128.
- Checkpoint: results/convnext_tiny/convnext_tiny_best.pt (fallback: checkpoints/stage2_epoch13.pt).
- Input: 224×224 RGB, raw [0,255] tensor (B,3,224,224).
- Output: sigmoid(logit); p=1 means Cancer.
- Threshold: config.py `CONVNEXT_THRESHOLD = 0.525` (argmax F1 on validation).
- **DISCREPANCY:** metrics.json lists `"app_py_current_threshold": 0.5` and `"app_py_changed": false`, but config.py (current source) uses 0.525. The metrics.json value appears outdated relative to config.py. Code verified at config.py:23 = 0.525; inference.py:213 uses `config.CONVNEXT_THRESHOLD`.
- Temperature scaling: none; `calibrated = prob` (raw sigmoid). `calibrated_prob()` function exists but is not applied in predict path.
- Grad-CAM: target layer is `model.features` (final spatial feature map before global average pooling), hooks capture activations + gradients, blended overlay. No GaussianBlur applied to CAM (unlike MobileNetV2).
- Batch prediction: no Grad-CAM.

**Metrics (SUPPORTED BY EXISTING ARTIFACTS, metrics.json):**
| Split | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|----------|-----------|--------|----|---------|
| Validation | 0.525 | 0.885 | 0.729 | 0.670 | 0.698 | 0.913 |
| Test | 0.525 | 0.860 | 0.639 | 0.622 | 0.630 | 0.892 |

Test melanoma sensitivity: 0.549 (per-class detection rate for mel: 90/164).
Best checkpoint: val_auc 0.9133, stage2, epoch 7 (early stopping on val ROC-AUC, patience 6, triggered).

---

## 13. Classification and inference pipeline

**INFERRED FROM CODE** (app.py):

1. Startup: `load_active_model()` loads the configured default model (ConvNeXt-Tiny).
2. Client POSTs image to `/predict` (single) or `/predict_batch` (batch).
3. Server validates: file present, allowed extension (png/jpg/jpeg), size limit (10 MB single / 50 MB batch total / 64 images max).
4. Image decoded with PIL; un-decodable → 400.
5. Quality check (`quality.check_image_quality`) runs; warnings are advisory only, never reject.
6. Inference:
   - ConvNeXt-Tiny: `inference.predict_single` / `predict_batch` → probs + metas; heatmap via `generate_gradcam` if enabled.
   - MobileNetV2: `mobilenet_predict_single` / `mobilenet_predict_batch` → calibrated prob, verdict, risk; heatmap via `mobilenet_grad_cam` if enabled.
7. Verdict: cancer_probability >= threshold → "affected" (cancer detected); else "not_affected".
8. Confidence: probability if >= 0.5 else 1 - probability.
9. Risk level: high (>=0.6), medium (>=0.3), low.
10. Response JSON includes: success, cancer_probability, verdict, cancer_detected, confidence, risk_level, prediction, heatmap (base64 PNG), quality_warnings, quality_metrics, quality_is_valid.
11. Errors: 400 (no file / bad type / corrupt image), 413 (too large), 503 (model not loaded), 500 (prediction failure).

---

## 14. Image-quality assessment

**INFERRED FROM CODE** (quality.py, config.py):

Checks run on the decoded PIL image (downscaled to 256×256 for speed; scale-invariant metrics):
- **Blur:** variance of Laplacian on grayscale. Threshold `QUALITY_BLUR_THRESHOLD = 100.0`. Below → blur warning.
- **Dark:** mean brightness < 0.20 → dark warning.
- **Bright/overexposed:** mean brightness > 0.95 OR >10% near-white pixels (>240/255) → bright warning (de-duplicated).
- **overexposure_ratio:** fraction of pixels > 240/255; threshold 0.10.

**Heuristic nature (SUPPORTED BY EXISTING ARTIFACTS):**
- `scripts/calibrate_quality.py` includes placeholders for manually reviewed images but none are present.
- No reliable human-labelled sharp/blurry/dark/overexposed examples available in the project.
- config.py documents: thresholds are heuristic and not scientifically calibrated against real skin photos.
- Real-image blur scores recorded as regression baseline in tests/test_quality.py; several real photos score below 100.0 and receive blur warnings even though they may not be genuinely blurry (test does not assert classification because ground-truth blur labels are unavailable).
- **Conclusion:** thresholds are heuristic; calibration requires labelled data (BLOCKED without it).

**Test results (VERIFIED BY EXECUTION):** tests/test_quality.py unit tests pass (14 passed); API compatibility tests pass (pytest: 16 quality-related tests passed total across both files).

---

## 15. Grad-CAM methodology and known issue

**INFERRED FROM CODE + VERIFIED BY EXECUTION:**

Methodology:
- MobileNetV2: gradients w.r.t. last Conv2D output; channel weights = spatial mean of gradients; CAM = ReLU(weighted sum of activations); GaussianBlur(3x3) applied; resized to 224×224; JET colormap; blended 0.55/0.45.
- ConvNeXt-Tiny: forward+backward hooks on `model.features`; channel weights = spatial mean of gradients; CAM = ReLU(weighted sum); NO GaussianBlur; resized to 224×224; JET colormap; blended 0.55/0.45.
- Target class: "affected" uses +logit; "not_affected" uses -logit.
- Grad-CAM generated for single-image predictions only (`ENABLE_GRADCAM_SINGLE=True`); disabled for batch (`ENABLE_GRADCAM_BATCH=False`).

**Known issue — MobileNetV2 corner activation (SUPPORTED BY EXISTING ARTIFACTS + VERIFIED BY EXECUTION):**
- README.md documents: "Suspected corner-activation limitation (Grad-CAM)." Measured on test images, the MobileNetV2 heatmap for the melanoma photo concentrates near image corners/borders (hot-pixel centroid (0.75, 0.83), corner-frame share 0.59 vs ~0.44 expected for uniform attention).
- pytest `test_heatmap_alignment[mobilenetv2-melanoma]` emitted a UserWarning: "activation concentrated near image corners/borders - localization may be spurious (see README limitation)" (VERIFIED BY EXECUTION, this session).
- ConvNeXt-Tiny heatmaps on same images had hot-pixel centroids near image centre (within 0.27 of centre), though centrality alone does not prove lesion localization.
- **Critical:** a blended overlay can be non-degenerate even if the raw CAM is empty/zero because the original image contributes variation. The tests validate the overlay mechanism and registration, NOT that the model attends to the lesion.
- **Do not claim Grad-CAM accurately localizes a lesion.** Treat heatmap as interpretability aid only.

**Tests (VERIFIED BY EXECUTION):**
- test_gradcam_heatmap_finite_and_non_degenerate: heatmap is valid, non-degenerate PNG (passed for both models, all images).
- test_gradcam_gradients_respond_to_target_class: affected vs not_affected heatmaps differ (passed, ConvNeXt-Tiny).
- test_gradcam_activation_gradient_shapes_compatible: activation/gradient shapes compatible, non-zero gradients (passed).
- test_heatmap_alignment: overlay dimensionally aligned (224×224) and pixel-registered to image (passed).

---

## 16. Flask routes and API endpoints

**INFERRED FROM CODE** (app.py):

| Route | Method | Description |
|-------|--------|-------------|
| / | GET | Render index.html with model_ready + active_model |
| /health | GET | `{status, active_model, model_loaded}` + model info (ConvNeXt) |
| /predict | POST | Single-image prediction; multipart field `file` |
| /predict_batch | POST | Batch prediction; multipart field `files` (max 64, 50 MB) |
| Error 413 | — | File exceeds size limit |

Response fields (single): success, cancer_probability, verdict, cancer_detected, confidence, risk_level, prediction, heatmap, quality_warnings, quality_metrics, quality_is_valid, active_model (implicit via config).

Error responses:
- 400: no file, invalid type, corrupt/un-decodable image, too many batch files.
- 413: file/batch too large.
- 503: model not loaded.
- 500: prediction exception.

**Security (INFERRED FROM CODE):**
- Extension whitelist: {png, jpg, jpeg}.
- MAX_FILE_SIZE = 10 MB; MAX_BATCH_FILE_SIZE = 50 MB; MAX_BATCH_SIZE = 64.
- Path traversal: files read into memory via BytesIO; never written to disk (INFERRED).
- Flask auto-escapes templates; no unsafe HTML rendering (INFERRED).
- Debug mode controlled by FLASK_DEBUG env (default false).
- No secrets in code (INFERRED).
- CORS enabled globally (flask_cors).
- DISCLAIMER in config.py: decision support only, not medical diagnosis; images analysed in memory, never stored or transmitted.

---

## 17. Frontend implementation

**INFERRED FROM CODE** (templates/index.html, static/css/style.css, static/js/app.js):

- Single image upload with preview canvas (224×224).
- Prediction display: verdict, probability, confidence, risk level, quality warnings.
- Heatmap overlay rendered from base64 PNG returned by backend.
- "Download PDF report" button (line 180) using jsPDF 2.5.1 from CDN (line 10).
- JS handler `$('pdf').onclick` (line 362): creates jsPDF, adds content, calls `doc.save('DermaScan-report-'+id+'.pdf')` (line 406).
- Fallback: `window.print()` if jspdf not loaded (line 364).
- Medical disclaimer rendered from backend config.

**PDF browser verification:** NOT VERIFIED. No browser automation was run; the `doc.save()` call triggers a browser download but this was not tested with a real browser. Do not claim browser-based PDF downloading was verified.

---

## 18. Batch predictions

**INFERRED FROM CODE + VERIFIED BY EXECUTION:**

- Endpoint: `/predict_batch`, multipart field `files`.
- Limits: max 64 images, total 50 MB.
- Each image: extension check, size accumulation, decode, quality check, inference.
- Results: per-image predictions with quality_warnings, quality_metrics, quality_is_valid attached.
- Order preserved; count returned.
- MobileNetV2 batch: no Grad-CAM (performance). ConvNeXt-Tiny batch: no Grad-CAM.
- **VERIFIED BY EXECUTION:** test_batch_prediction passed for both models (pytest + standalone).

---

## 19. Client-side PDF report generation

**INFERRED FROM CODE** (templates/index.html):

- Library: jspdf 2.5.1 loaded from cdnjs (https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js).
- Trigger: "Download PDF report" button; onclick handler builds a jsPDF document and calls `doc.save(...)`.
- Filename pattern: `DermaScan-report-<id>.pdf`.
- Backend: no PDF endpoint; PDF is entirely client-side.
- **NOT VERIFIED:** browser download not tested with a real browser. The mechanism exists in code but end-to-end browser behavior (download dialog, file saved) was not exercised.

---

## 20. Security review

**INFERRED FROM CODE:**

| Concern | Status | Basis |
|---------|--------|-------|
| File type validation | Pass | Extension whitelist {png,jpg,jpeg} |
| Size limits | Pass | 10 MB single, 50 MB batch, 64 images |
| Path traversal | Pass (inferred) | Files read into memory via BytesIO; never written to disk |
| Unsafe HTML rendering | Pass (inferred) | Flask/Jinja auto-escapes |
| Debug mode | Pass (default false) | FLASK_DEBUG env, default "false" |
| Secret exposure | Pass (inferred) | No secrets/keys in source code |
| Error handling | Pass | 400/413/503/500 with JSON error messages |
| CORS | Enabled | flask_cors globally |
| Image storage | Not stored | config DISCLAIMER: analysed in memory, never stored/transmitted |
| Model not loaded | 503 | load_active_model returns None → 503 |

**NOT VERIFIED:** no penetration test or dependency vulnerability scan (e.g., `pip audit`) was performed.

---

## 21. Software test results

**VERIFIED BY EXECUTION (this session, 2026-10-17):**

**pytest suite (`pytest tests/ -v`):** 56 passed, 36 warnings, 0 failed, 0 errored.
- tests/test_quality.py: 14 passed (unit + API tests).
- tests/test_regression.py: 42 passed (parameterized over convnext_tiny × mobilenetv2 × 4 test images).
- Warnings: PyTorch full-backward-hook notices, Keras input-structure notice, and heatmap_localization warnings for mobilenetv2-melanoma (corner-activation diagnostic — expected, non-fatal).

**Standalone regression suite (`python tests/test_regression.py`):** 22 passed, 0 failed, 0 errored, 0 warnings.
- convnext_tiny: 11 passed, 0 failed/errored.
- mobilenetv2: 11 passed, 0 failed/errored.

**Coverage (tested):** health endpoint, single prediction, batch prediction, Grad-CAM validity/alignment/gradient-responsiveness/shape-compatibility, error handling (no file / bad type / corrupt / too large / too many batch / batch too large), quality warnings (blur/dark/overexposed), invalid-image rejection.

**Coverage (not tested by suites):** missing-model handling (code path exists but not exercised), browser PDF download, external validation, patient-level leakage, confidence calibration effectiveness.

**Important distinction (software testing vs model-performance evaluation):** Passing software tests confirm the application code runs correctly (endpoints respond, predictions return valid fields, heatmaps generate, errors are handled). They do NOT validate clinical accuracy, diagnostic safety, or model performance on real patients. Model metrics come from HAM10000 test-split artifacts (model_card.json, metrics.json) and are separate from software test results.

---

## 22. Model evaluation metrics

**SUPPORTED BY EXISTING ARTIFACTS (not re-derived, not re-run):**

MobileNetV2 (model_card.json, threshold 0.22):
- Validation F1: 0.6451; Test F1: 0.5847; Test ROC-AUC: 0.8624.
- Validation: acc 0.832, prec 0.554, rec 0.771 (n=1500).
- Test: acc 0.798, prec 0.482, rec 0.743 (n=1505; tn=987, fp=230, fn=74, tp=214).
- Test @ 0.50: acc 0.849, prec 0.638, rec 0.490, f1 0.554 (n=1505).
- Per-class test (threshold 0.22): Non-Cancer f1 0.867 (support 1217); Cancer f1 0.585 (support 288).
- Mean probability per dx (test): mel 0.410 (p_ge_0.5=0.409), bcc 0.550 (0.578), akiec 0.591 (0.634); nv 0.080 (0.029), bkl 0.291 (0.21), vasc 0.236 (0.19), df 0.381 (0.406).

ConvNeXt-Tiny (metrics.json, threshold 0.525):
- Validation F1: 0.6982; Test F1: 0.6303; Test ROC-AUC: 0.8924.
- Validation: acc 0.885, prec 0.729, rec 0.670 (n=1500; tn=1129, fp=74, fn=98, tp=199).
- Test: acc 0.860, prec 0.639, rec 0.622 (n=1505; tn=1116, fp=101, fn=109, tp=179).
- Per-class test: Non-Cancer f1 0.914 (support 1217); Cancer f1 0.630 (support 288).
- Per-dx detection rate (test): mel 0.549 (90/164), bcc 0.723 (60/83), akiec 0.707 (29/41); nv 0.047 (47/1002), bkl 0.191 (31/162), vasc 0.333 (7/21), df 0.500 (16/32).
- Test melanoma sensitivity: 0.549.

**Metric attribution:** MobileNetV2 metrics are from model_card.json; ConvNeXt-Tiny metrics are from metrics.json. Results are NOT cross-attributed.

**Reproducibility of metrics:** ConvNeXt-Tiny test_proba.npy and test_true.npy exist, so metrics could be re-derived. MobileNetV2 full test predictions/labels are NOT available (only single_image_predictions for 6 images in model_card.json), so MobileNetV2 test metrics cannot be independently reproduced from available .npy files (LIMITATION).

---

## 23. Confidence calibration

**SUPPORTED BY EXISTING ARTIFACTS:**

- MobileNetV2: temperature scaling investigated (T=1.2543, NLL minimised on validation; target sensitivity 0.95). val_log_loss_raw 0.3196 → calibrated 0.3128. **NOT applied to app.py.** `MOBILENET_TEMPERATURE = 1.0` (identity). Threshold remains 0.22.
- ConvNeXt-Tiny: no temperature scaling applied or investigated in artifacts. `calibrated = prob` (raw sigmoid). Threshold 0.525.
- `calibrated_prob()` function exists in inference.py but is not called in the predict path.

**Conclusion:** Confidence calibration has NOT been applied to the production models. Output probabilities are raw sigmoid values (MobileNetV2 T=1.0; ConvNeXt identity).

---

## 24. Image-quality calibration

**SUPPORTED BY EXISTING ARTIFACTS:**

- Thresholds: QUALITY_BLUR_THRESHOLD=100.0, QUALITY_BRIGHTNESS_DARK=0.20, QUALITY_BRIGHTNESS_BRIGHT=0.95, QUALITY_OVEREXPOSURE_RATIO=0.10.
- `scripts/calibrate_quality.py` has placeholders for manually reviewed images; none present.
- No reliable human-labelled sharp/blurry/dark/overexposed dataset available.
- config.py documents thresholds as heuristic, not scientifically calibrated against real skin photos.
- Real-image blur scores recorded as regression baseline; several real photos score below 100.0 and receive blur warnings without ground-truth blur labels.

**Conclusion:** Image-quality thresholds are heuristic. Calibration requires a human-labelled dataset (BLOCKED without it).

---

## 25. External validation

**NOT VERIFIED / BLOCKED:**

- No independent external dataset was used for evaluation.
- test-images/ contains external clinical photographs, but model_card.json concludes they are "Not from HAM10000" and their filenames are NOT ground truth (cannot be verified).
- No external evaluation metrics exist in artifacts.
- **External validation is not established.** Do not claim external validation.

---

## 26. Bugs actually fixed

**None during this session.** No code changes were made to the application. The only working-tree modification is README.md (pre-existing: removal of incorrect `test_regression.py` root reference, update of Grad-CAM target description, update of test instructions). These README changes were made prior to this session and are documented in git diff. No bugs were fixed, no model weights or thresholds were modified, no retraining was performed.

---

## 27. Files changed

**Git status (VERIFIED BY EXECUTION, this session):**

Modified (tracked):
- README.md (8 insertions, 3 deletions) — pre-existing modification from prior audit session.

Untracked (new files, not committed):
- PROJECT_AUDIT.md
- PROJECT_CHANGELOG.md
- PROJECT_COMPLETION_REPORT.md
- DERMAI_FINAL_PROJECT_REPORT.md (this report)
- regression_test_results.txt
- research/
- scripts/

No model files, source code (app.py, config.py, inference.py, quality.py), or test files were modified during this session.

---

## 28. Installation and execution instructions

**INFERRED FROM CODE + config.py:**

Prerequisites: Python 3.13, venv at `<project>\venv`.

```powershell
# Activate virtual environment
C:\Users\Manjunath\OneDrive\Desktop\MINI3\skin-cancer-app-simple\skin-cancer-app\venv\Scripts\Activate.ps1

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run the application
python app.py
# Default: http://0.0.0.0:5000 (host/port overridable via DERMA_HOST / DERMA_PORT)
# Debug: set DERMA_DEBUG=true
```

**Run tests:**
```powershell
# Full pytest suite
& "C:\...\venv\Scripts\python.exe" -m pytest tests -v

# Standalone regression suite (no pytest required)
& "C:\...\venv\Scripts\python.exe" tests\test_regression.py
```

**Model selection:** config.py `DEFAULT_MODEL = "convnext_tiny"` (recommended) or `"mobilenetv2"`. ConvNeXt device: `CONVNEXT_DEVICE = "cuda"` (falls back to CPU if CUDA unavailable).

---

## 29. Demonstration procedure

**INFERRED FROM CODE (not executed with a browser this session):**

1. Start `python app.py` (venv activated).
2. Open `http://localhost:5000/` in a browser.
3. Confirm `/health` returns `{"status": "ok", "model_loaded": true, ...}`.
4. Upload a skin-lesion image (jpg/png/jpeg, ≤10 MB).
5. Observe: verdict, cancer probability, confidence, risk level, quality warnings, heatmap overlay.
6. Click "Download PDF report" to generate a client-side PDF via jsPDF (browser download not tested).
7. For batch: use `/predict_batch` programmatically with up to 64 images (≤50 MB total).

---

## 30. Reproducibility

**Split reproducibility:** SUPPORTED BY EXISTING ARTIFACTS — model_card.json and metrics.json both state the lesion-level 70/15/15 split (random_state=42) reproduces the notebook exactly.

**Metric reproducibility:**
- ConvNeXt-Tiny: test_proba.npy and test_true.npy exist; metrics can be re-derived from these artifacts.
- MobileNetV2: full test predictions/labels not available as .npy files; only 6 single_image_predictions in model_card.json. Test metrics cannot be independently reproduced from available prediction files (LIMITATION).

**Code reproducibility:** The application runs from source with the venv; model files (.h5, .pt) are present and unmodified. Training code is not required for inference.

**Not reproduced this session:** training (no training jobs run), temperature-scaling calibration application, quality-threshold calibration, external validation.

---

## 31. Known limitations

1. **Grad-CAM lesion localization not established** (MobileNetV2 corner-activation bias; blended overlay can be non-degenerate even with empty raw CAM). Heatmap is an interpretability aid only. (SUPPORTED BY EXISTING ARTIFACTS + VERIFIED BY EXECUTION)
2. **Confidence calibration not applied** to either production model. (SUPPORTED BY EXISTING ARTIFACTS)
3. **Image-quality thresholds are heuristic**; no human-labelled calibration dataset. (SUPPORTED BY EXISTING ARTIFACTS)
4. **Patient-level separation not assessable**; no patient-to-lesion mapping in artifacts. (BLOCKED)
5. **External validation not established.** (NOT VERIFIED / BLOCKED)
6. **Browser PDF download not tested.** (NOT VERIFIED)
7. **MobileNetV2 test metrics not independently reproducible** (no .npy prediction files). (LIMITATION)
8. **Threshold discrepancies:** model_card.json and metrics.json claim app.py uses threshold 0.5 for both models, but config.py uses 0.22 (MobileNetV2) and 0.525 (ConvNeXt-Tiny). Code takes precedence; artifact files appear outdated. (INFERRED FROM CODE vs artifacts)
9. **test-images/ are external clinical photos, not HAM10000 ground truth.** (SUPPORTED BY EXISTING ARTIFACTS)
10. **Application is a research/demo decision-support tool, not a medical diagnosis device.** (config.py DISCLAIMER)

---

## 32. Future work

(INFERRED FROM CODE + artifacts; none claimed as complete)

- Apply temperature-scaling calibration (T=1.2543 estimate exists for MobileNetV2) and re-evaluate.
- Calibrate image-quality thresholds with a human-labelled dataset.
- Investigate and mitigate MobileNetV2 Grad-CAM corner-activation bias.
- Add patient-level split verification if HAM10000 patient mapping becomes available.
- Perform external validation on an independent dermatology dataset.
- Test browser-based PDF download end-to-end.
- Add per-class (7-class dx) evaluation in addition to binary Cancer/Non-Cancer.
- Consider model uncertainty / abstention for low-confidence predictions.

---

## 33. Exact Git status

**VERIFIED BY EXECUTION (git status --porcelain, this session):**

```
 M README.md
?? PROJECT_AUDIT.md
?? PROJECT_CHANGELOG.md
?? PROJECT_COMPLETION_REPORT.md
?? DERMAI_FINAL_PROJECT_REPORT.md
?? regression_test_results.txt
?? research/
?? scripts/
```

Branch: main (up to date with origin/main).  
Current commit: 94b4af0 feat: complete DermaAI image quality warning system.  
Previous commit: 880d17a Initial commit - DermaAI skin cancer detection app.  
No commit, push, or deployment occurred during this session.

---

## 34. Final completion assessment

**Software:** All tests pass (pytest 56 passed / standalone regression 22 passed). Application starts, /health returns ok, single and batch predictions work, error handling is correct, quality warnings function, Grad-CAM generates valid heatmaps. Model files unchanged.

**Documentation:** PROJECT_COMPLETION_REPORT.md and DERMAI_FINAL_PROJECT_REPORT.md (this report) document the final state. README.md has pre-existing modifications (uncommitted).

**Honest limitations:** Grad-CAM localization not established; confidence calibration not applied; quality thresholds heuristic; no patient-level separation evidence; no external validation; browser PDF download not tested; MobileNetV2 test metrics not independently reproducible from .npy files.

**Bottom line:** DermaAI is a functioning research/demo decision-support application for binary skin-lesion classification. It is not a substitute for professional medical diagnosis. Both reports exist; DERMAI_FINAL_PROJECT_REPORT.md is the consolidated source document for updating the research paper and project paperwork.

No commit, push, or deployment occurred.