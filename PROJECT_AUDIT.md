# DermaAI — PROJECT AUDIT
**Generated:** 2025 (Phase 1 of structured cleanup)  
**Auditor:** Amazon Q (senior ML/Python engineer review)  
**Scope:** `skin-cancer-app-simple/skin-cancer-app/`

---

## 1. Current Folder Structure (Before Cleanup)

```
skin-cancer-app/
├── app.py                          # Flask application (production)
├── inference.py                    # ConvNeXt-Tiny PyTorch backend (production)
├── config.py                       # Single source of truth for all settings (production)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation (good quality, up to date)
├── .gitignore                      # Git ignore rules (needs update)
│
├── test_regression.py              # Automated regression suite (production test)
├── train_convnext_tiny.py          # ConvNeXt-Tiny training script (research)
├── debug_gradcam.py                # Grad-CAM debug script → outputs to debug_gradcam/
├── investigate_gradcam.py          # Comprehensive Grad-CAM investigation → gradcam_investigation/
├── generate_visualizations.py      # Visualization generator → visualizations/
├── calibrate.py                    # MobileNetV2 temperature-scaling calibration tool
├── complete_c03_validation.py      # Experiment c03 validation (dermai_exp framework)
├── make_all_summaries.py           # Batch summary generator for experiments c01/c02/c03
├── make_c01_summary.py             # Summary for experiment c01_aux7
├── make_c03_summary.py             # Summary for experiment c03_heavy_mixup
│
├── audit_report.txt                # Previous pipeline audit (valuable research record)
├── test_output.txt                 # Last regression test run output (UTF-16 encoded)
├── gradcam_generation.log          # Log from generate_visualizations.py run
├── gradcam_investigation.log       # Log from investigate_gradcam.py run
│
├── dermai_convnext_tiny_colab.ipynb  # Colab training notebook (research, original)
│
├── WhatsApp Image 2026-09-28 at 8.01.26 PM (1).jpeg  # Personal test image (HAM ISIC_0027190, mel)
├── WhatsApp Image 2026-09-28 at 8.01.26 PM.jpeg      # Personal test image (HAM ISIC_0024698, nv)
│
├── model/
│   ├── skin_cancer_mobilenetv2.h5  # MobileNetV2 weights (~22 MB) — PRODUCTION
│   ├── model_card.json             # Full evaluation card with metrics, ROC, provenance
│   └── threshold.txt               # MobileNetV2 threshold (0.22)
│
├── results/
│   ├── convnext_tiny/              # ConvNeXt-Tiny training outputs
│   │   ├── checkpoints/
│   │   │   ├── stage2_epoch12.pt   # Training checkpoint
│   │   │   └── stage2_epoch13.pt   # Training checkpoint (ACTIVE — used by config.py)
│   │   ├── convnext_tiny_best.pt   # Best model checkpoint — PRODUCTION
│   │   ├── metrics.json            # Full evaluation metrics
│   │   ├── config.json             # Training configuration
│   │   ├── threshold.json          # Threshold selection record
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.png / .csv (val + test)
│   │   ├── roc_curve.png / .csv (val + test)
│   │   ├── training_curves.png
│   │   ├── training_log.txt
│   │   ├── evaluation_log.txt
│   │   ├── history_stage1.csv / history_stage2.csv
│   │   ├── val_threshold_sweep.csv
│   │   ├── val_proba.npy / val_true.npy
│   │   ├── val_proba_best_epoch.npy / val_true_best_epoch.npy
│   │   ├── test_proba.npy / test_true.npy
│   │   └── roc_curve_val.csv / roc_curve_test.csv
│   ├── experiments/
│   │   ├── a01_cnxt224/            # Experiment a01 (ConvNeXt-Tiny baseline)
│   │   ├── a01_cnxt224_base/       # Experiment a01 base variant
│   │   ├── b01_r224_ref14/         # Experiment b01 (resolution sweep)
│   │   ├── b02_r288/               # Experiment b02
│   │   ├── c01_aux7/               # Experiment c01 (auxiliary head)
│   │   ├── c02_pw4/                # Experiment c02 (patchwise)
│   │   ├── c03_heavy_mixup/        # Experiment c03 (heavy mixup) — best sweep result
│   │   ├── c04_cnxts/              # Experiment c04 (ConvNeXt-Small)
│   │   ├── drive_log.txt
│   │   ├── plan_b.json / plan_c.json
│   │   ├── sweep_log.txt
│   │   └── sweep_summary_plan_c.json
│   ├── _ckpt/                      # Shared checkpoint staging area (empty)
│   └── final/                      # Final results staging area (empty)
│
├── dermai_exp/                     # Experiment framework (Python package)
│   ├── __init__.py
│   ├── common.py                   # Shared paths, logging, seeding
│   ├── models.py                   # BinaryNet backbone builder (multi-arch)
│   ├── data.py                     # Dataset loading
│   ├── aug.py                      # Augmentation pipeline
│   ├── engine.py                   # Trainer class
│   ├── cv.py                       # Cross-validation utilities
│   ├── run_exp.py                  # Experiment runner
│   ├── sweep.py                    # Hyperparameter sweep
│   ├── analyze_run.py              # Run analysis
│   ├── drive_plan.py               # Drive-based experiment planning
│   ├── final_eval.py               # Final evaluation
│   ├── profile_conv.py             # Convolution profiling
│   └── profile_step.py             # Step profiling
│
├── debug_gradcam/                  # Output of debug_gradcam.py (20 PNG files)
├── gradcam_investigation/          # Output of investigate_gradcam.py (36 PNG + 1 CSV)
├── gradcam_outputs_new/            # Output of earlier Grad-CAM generation (9 files)
├── visualizations/                 # Output of generate_visualizations.py (48 PNG + 1 JSON)
│
├── templates/
│   └── index.html                  # Single-page frontend
├── static/
│   ├── css/style.css
│   └── js/app.js
│
├── test-images/                    # 4 external clinical photos for regression tests
│   ├── cancer-melanoma.jpg
│   ├── noncancer-congenital-nevus.jpg
│   ├── noncancer-nevus.jpg
│   └── noncancer-seborrheic-keratosis.jpg
│
└── tools/
    ├── verify_models.py            # MobileNetV2 verification harness (needs HAM10000)
    ├── trace_test_image_provenance.py  # Image provenance tracer
    └── results/                    # Output of verify_models.py
│
└── .kilo/                          # Kilo editor worktree (IDE artifact)
└── .qodo/                          # Qodo AI tooling cache (IDE artifact)
```

---

## 2. Purpose of Important Files

### Production Application (must never break)
| File | Purpose |
|------|---------|
| `app.py` | Flask app: routes `/`, `/health`, `/predict`, `/predict_batch`. Loads both models, runs Grad-CAM inline. |
| `inference.py` | ConvNeXt-Tiny PyTorch backend: `DermaConvNeXt` class, `load_convnext_model()`, `predict_single()`, `predict_batch()`, `generate_gradcam()`. |
| `config.py` | Single source of truth: all paths, thresholds, input sizes, Flask settings. |
| `model/skin_cancer_mobilenetv2.h5` | MobileNetV2 Keras weights (~22 MB). Required for MobileNetV2 inference. |
| `model/model_card.json` | Full evaluation card: metrics, ROC curve data, provenance, calibration. |
| `model/threshold.txt` | MobileNetV2 threshold (0.22). |
| `results/convnext_tiny/convnext_tiny_best.pt` | ConvNeXt-Tiny best checkpoint. Required for inference. |
| `results/convnext_tiny/checkpoints/stage2_epoch13.pt` | Active checkpoint referenced by `config.py:CONVNEXT_CKPT`. |
| `templates/index.html` | Frontend HTML. |
| `static/css/style.css` | Frontend CSS. |
| `static/js/app.js` | Frontend JavaScript. |
| `test-images/` | 4 external clinical photos. Required by `test_regression.py`. |

### Tests
| File | Purpose |
|------|---------|
| `test_regression.py` | Full regression suite: health, single prediction, batch, Grad-CAM validity, heatmap alignment, error handling. Tests both models. Last run: 22/22 PASS, 2 WARN. |
| `test_output.txt` | Saved output of last regression run (UTF-16 encoded). |

### Training
| File | Purpose |
|------|---------|
| `train_convnext_tiny.py` | Full ConvNeXt-Tiny training pipeline (GPU only). Requires HAM10000 dataset at `D:\`. Not needed for inference. |
| `dermai_convnext_tiny_colab.ipynb` | Original Colab training notebook. Research reproducibility artifact. |
| `dermai_exp/` | Experiment framework used for the sweep experiments (a01, b01, c01–c04). |

### Grad-CAM & Visualization Scripts
| File | Purpose |
|------|---------|
| `debug_gradcam.py` | Early debug script. Generates overlays to `debug_gradcam/`. Superseded by `investigate_gradcam.py`. |
| `investigate_gradcam.py` | Comprehensive investigation: side-by-side comparisons, occlusion sensitivity, CSV statistics. Outputs to `gradcam_investigation/`. |
| `generate_visualizations.py` | Generates full visualization set with statistics JSON. Outputs to `visualizations/`. |

### Research / Evaluation Scripts
| File | Purpose |
|------|---------|
| `calibrate.py` | Temperature-scaling calibration for MobileNetV2. Requires validation directory. Standalone tool. |
| `complete_c03_validation.py` | Runs full TTA evaluation on c03_heavy_mixup checkpoint. Requires HAM10000. |
| `make_all_summaries.py` | Generates summary.json for c01/c02/c03 from existing .npy files. Requires HAM10000. |
| `make_c01_summary.py` | Same as above, c01 only. Superseded by `make_all_summaries.py`. |
| `make_c03_summary.py` | Same as above, c03 only. Superseded by `make_all_summaries.py`. |
| `audit_report.txt` | Previous pipeline audit. Valuable research record documenting test image provenance, split integrity, and Grad-CAM findings. |
| `tools/verify_models.py` | MobileNetV2 verification harness. Requires HAM10000 at `D:\`. |
| `tools/trace_test_image_provenance.py` | Traces test image provenance against HAM10000. |

### Generated Outputs (Grad-CAM images)
| Directory | Contents | Source Script |
|-----------|----------|---------------|
| `debug_gradcam/` | 20 PNGs: original + overlay for both models, 4 images | `debug_gradcam.py` |
| `gradcam_investigation/` | 36 PNGs + 1 CSV: original, raw heatmap, overlay, comparison, occlusion | `investigate_gradcam.py` |
| `gradcam_outputs_new/` | 9 files: overlay + raw heatmap for 4 classes + statistics CSV | Earlier generation script |
| `visualizations/` | 48 PNGs + 1 JSON: full set with raw heat, JET, overlay for both models | `generate_visualizations.py` |

### Personal / Miscellaneous
| File | Notes |
|------|-------|
| `WhatsApp Image 2026-09-28 at 8.01.26 PM (1).jpeg` | Personal photo. Identified in model_card.json as HAM ISIC_0027190 (mel/Cancer). |
| `WhatsApp Image 2026-09-28 at 8.01.26 PM.jpeg` | Personal photo. Identified as HAM ISIC_0024698 (nv/Non-Cancer). |
| `.kilo/` | Kilo editor worktree — IDE artifact, not project code. |
| `.qodo/` | Qodo AI tooling cache — IDE artifact. |

---

## 3. Dependency Map

```
app.py
  └── imports: config, inference, flask, tensorflow, PIL, numpy, cv2

inference.py
  └── imports: config, torch, torchvision, numpy, cv2, PIL

config.py
  └── imports: os only (no project imports)

test_regression.py
  └── imports: app (and transitively: config, inference)
  └── reads: test-images/*.jpg

train_convnext_tiny.py
  └── standalone (no project imports)
  └── writes: results/convnext_tiny/

debug_gradcam.py
  └── imports: inference, app
  └── reads: test-images/*.jpg
  └── writes: debug_gradcam/

investigate_gradcam.py
  └── imports: config, inference, app
  └── reads: test-images/*.jpg
  └── writes: gradcam_investigation/

generate_visualizations.py
  └── imports: inference, app (via from app import ...)
  └── reads: test-images/*.jpg
  └── writes: visualizations/

calibrate.py
  └── standalone (tensorflow only)

complete_c03_validation.py
  └── imports: dermai_exp.common, dermai_exp.engine, dermai_exp.run_exp

make_all_summaries.py / make_c01_summary.py / make_c03_summary.py
  └── imports: dermai_exp.common, dermai_exp.run_exp

dermai_exp/
  └── common.py: standalone
  └── models.py: imports common
  └── engine.py: imports common, models, data, aug
  └── run_exp.py: imports common, models, data, engine, aug
  └── sweep.py: imports common, run_exp
  └── (all others): import from within dermai_exp

tools/verify_models.py
  └── standalone (tensorflow, sklearn, pandas)
  └── hardcoded paths: D:\User\Downloads\archive, local APP path
```

---

## 4. Files That Appear Duplicated or Redundant

| File | Issue |
|------|-------|
| `make_c01_summary.py` | Functionality fully covered by `make_all_summaries.py`. Older, less robust version. |
| `make_c03_summary.py` | Same — covered by `make_all_summaries.py`. |
| `debug_gradcam/` | Subset of outputs from `gradcam_investigation/` (fewer images, no statistics). `gradcam_investigation/` is the superset. |
| `gradcam_outputs_new/` | Older generation run. `visualizations/` is the comprehensive replacement. |
| `test_output.txt` | Saved test run output. Useful as a reference but regenerable. |
| `gradcam_generation.log` | Log file from a script run. Regenerable. |
| `gradcam_investigation.log` | Log file from a script run. Regenerable. |

---

## 5. Recommended Organization

```
skin-cancer-app/
├── app.py                          # KEEP (production)
├── inference.py                    # KEEP (production)
├── config.py                       # KEEP (production)
├── requirements.txt                # KEEP
├── README.md                       # UPDATE
├── PROJECT_AUDIT.md                # NEW (this file)
├── PROJECT_CHANGELOG.md            # NEW
├── RESEARCH_NOTES.md               # NEW
├── .gitignore                      # UPDATE
│
├── model/                          # KEEP AS-IS
│   ├── skin_cancer_mobilenetv2.h5
│   ├── model_card.json
│   └── threshold.txt
│
├── templates/                      # KEEP AS-IS
│   └── index.html
│
├── static/                         # KEEP AS-IS
│   ├── css/style.css
│   └── js/app.js
│
├── test-images/                    # KEEP AS-IS (required by tests)
│
├── tests/
│   └── test_regression.py          # MOVE from root
│
├── scripts/
│   ├── training/
│   │   └── train_convnext_tiny.py  # MOVE
│   ├── evaluation/
│   │   ├── calibrate.py            # MOVE
│   │   ├── complete_c03_validation.py  # MOVE
│   │   ├── make_all_summaries.py   # MOVE
│   │   └── make_c01_summary.py     # MOVE (keep for reference)
│   │   └── make_c03_summary.py     # MOVE (keep for reference)
│   ├── visualization/
│   │   ├── generate_visualizations.py  # MOVE
│   │   └── investigate_gradcam.py  # MOVE
│   └── debugging/
│       └── debug_gradcam.py        # MOVE
│
├── research/
│   ├── notebooks/
│   │   └── dermai_convnext_tiny_colab.ipynb  # MOVE
│   ├── documentation/
│   │   └── audit_report.txt        # MOVE
│   └── figures/
│       ├── gradcam_investigation/  # MOVE (comprehensive set)
│       └── visualizations/         # MOVE (comprehensive set)
│
├── results/                        # KEEP AS-IS (model checkpoints + metrics)
│   ├── convnext_tiny/
│   └── experiments/
│
├── dermai_exp/                     # KEEP AS-IS (experiment framework)
│
├── tools/                          # KEEP AS-IS
│
└── _quarantine/                    # NEW — cleanup candidates (not deleted yet)
    ├── debug_gradcam/              # Superseded by gradcam_investigation/
    ├── gradcam_outputs_new/        # Superseded by visualizations/
    ├── test_output.txt             # Regenerable
    ├── gradcam_generation.log      # Regenerable
    ├── gradcam_investigation.log   # Regenerable
    ├── WhatsApp Image *.jpeg       # Personal photos (documented in model_card.json)
    └── .kilo/                      # IDE artifact
```

---

## 6. Potential Cleanup Risks

| Risk | Mitigation |
|------|-----------|
| `test_regression.py` uses relative paths `test-images/...` | Must stay in project root OR paths must be updated. **Moving requires path fix.** |
| `investigate_gradcam.py` has hardcoded absolute path `sys.path.insert(0, r'C:\Users\Manjunath\...')` | This is a research script, not production. Path is only needed when run standalone. |
| `make_c01_summary.py` imports `dermai_exp.run_exp.context` at module level (not in `main()`) | Running it triggers HAM10000 dataset loading. Moving is safe; running requires dataset. |
| `results/convnext_tiny/checkpoints/stage2_epoch13.pt` is referenced by `config.py:CONVNEXT_CKPT` | Must not be moved or renamed. |
| `results/convnext_tiny/convnext_tiny_best.pt` is referenced by `config.py:CONVNEXT_BEST` | Must not be moved or renamed. |
| `model/skin_cancer_mobilenetv2.h5` is referenced by `config.py:MOBILENET_PATH` | Must not be moved or renamed. |
| `.gitignore` currently ignores `*.jpeg` and `*.jpg` globally | This would exclude `test-images/*.jpg` from git. The test images are needed for the regression suite. |

---

## 7. Proposed Actions Table

| File / Directory | Action | Reason |
|-----------------|--------|--------|
| `app.py` | **KEEP** | Production application |
| `inference.py` | **KEEP** | Production inference backend |
| `config.py` | **KEEP** | Production configuration |
| `requirements.txt` | **KEEP** | Dependencies |
| `README.md` | **UPDATE** | Needs structure section update |
| `model/` (all files) | **KEEP** | Production model files |
| `templates/index.html` | **KEEP** | Production frontend |
| `static/` | **KEEP** | Production frontend assets |
| `test-images/` | **KEEP** | Required by regression tests |
| `test_regression.py` | **MOVE → tests/** | Organize tests into dedicated folder |
| `train_convnext_tiny.py` | **MOVE → scripts/training/** | Training script, not production |
| `calibrate.py` | **MOVE → scripts/evaluation/** | Evaluation tool |
| `complete_c03_validation.py` | **MOVE → scripts/evaluation/** | Research evaluation |
| `make_all_summaries.py` | **MOVE → scripts/evaluation/** | Research evaluation |
| `make_c01_summary.py` | **MOVE → scripts/evaluation/** | Research (superseded but keep) |
| `make_c03_summary.py` | **MOVE → scripts/evaluation/** | Research (superseded but keep) |
| `generate_visualizations.py` | **MOVE → scripts/visualization/** | Visualization tool |
| `investigate_gradcam.py` | **MOVE → scripts/visualization/** | Visualization/research tool |
| `debug_gradcam.py` | **MOVE → scripts/debugging/** | Debug tool |
| `dermai_convnext_tiny_colab.ipynb` | **MOVE → research/notebooks/** | Research notebook |
| `audit_report.txt` | **MOVE → research/documentation/** | Research record |
| `gradcam_investigation/` | **MOVE → research/figures/** | Research outputs |
| `visualizations/` | **MOVE → research/figures/** | Research outputs |
| `results/` | **KEEP** | Model checkpoints + metrics |
| `dermai_exp/` | **KEEP** | Experiment framework |
| `tools/` | **KEEP** | Verification tools |
| `debug_gradcam/` | **ARCHIVE → _quarantine/** | Superseded by gradcam_investigation/ |
| `gradcam_outputs_new/` | **ARCHIVE → _quarantine/** | Superseded by visualizations/ |
| `test_output.txt` | **ARCHIVE → _quarantine/** | Regenerable log |
| `gradcam_generation.log` | **ARCHIVE → _quarantine/** | Regenerable log |
| `gradcam_investigation.log` | **ARCHIVE → _quarantine/** | Regenerable log |
| `WhatsApp Image *.jpeg` (2 files) | **ARCHIVE → _quarantine/** | Personal photos, documented in model_card.json |
| `.kilo/` | **ARCHIVE → _quarantine/** | IDE artifact |
| `.qodo/` | **KEEP** | Active IDE tooling |
| `PROJECT_AUDIT.md` | **CREATE** | This file |
| `PROJECT_CHANGELOG.md` | **CREATE** | Change log |
| `RESEARCH_NOTES.md` | **CREATE** | Research documentation |

---

## 8. Critical Path Dependencies (Do Not Break)

```
config.py
  CONVNEXT_CKPT  → results/convnext_tiny/checkpoints/stage2_epoch13.pt  [MUST EXIST]
  CONVNEXT_BEST  → results/convnext_tiny/convnext_tiny_best.pt           [MUST EXIST]
  MOBILENET_PATH → model/skin_cancer_mobilenetv2.h5                      [MUST EXIST]

test_regression.py
  TEST_IMAGES    → test-images/*.jpg                                      [MUST EXIST]
  imports        → app (which imports config, inference)                  [MUST BE IN ROOT]
```

**Conclusion:** `app.py`, `inference.py`, `config.py`, and `test_regression.py` must remain in the project root. All model files and test images must remain at their current relative paths.
---

## 9. Regression Test Results (2026-10-15)

### Test Execution Summary
`
ConvNeXt-Tiny: 11 passed, 0 failed
MobileNetV2:   11 passed, 0 failed
Total:         22 passed, 0 failed, 0 errors, 1 non-fatal warning
`

### Test Coverage
- Health endpoint validation (both models)
- Single-image prediction with Grad-CAM (4 test images x 2 models = 8 tests)
- Heatmap alignment & registration verification (8 tests)
- Batch prediction (2 tests)
- Error handling: invalid inputs, corrupted files, size limits, batch limits (2 tests)

### Non-Fatal Warning (Unresolved Limitation)
- **MobileNetV2 + melanoma image**: Heatmap activation concentrated at bottom-right corner (centroid 0.79, 0.86; corner-frame share 0.68). Regression test correctly emits [WARN] heatmap_localization(cancer-melanoma.jpg). This is a known limitation of the legacy MobileNetV2 Grad-CAM implementation (see PROJECT_CHANGELOG.md).

### Test Artifacts
- egression_test_results.txt - Full test output log (UTF-8)
- 	est_output.txt - Previous run output (UTF-16, legacy, archivable)

---

## 10. Current State vs. Audit Recommendations (2026-10-15)

### Already Organized (Partial)
| Target Location | Status | Notes |
|----------------|--------|-------|
| scripts/training/train_convnext_tiny.py | Exists | But duplicate remains in root |
| scripts/evaluation/ (5 files) | Exists | But duplicates remain in root |
| scripts/visualization/generate_visualizations.py | Exists | But duplicate remains in root; investigate_gradcam.py **missing** |
| scripts/debugging/debug_gradcam.py | Exists | But duplicate remains in root |
| 	ests/test_regression.py | Exists | But duplicate remains in root |
| esearch/notebooks/dermai_convnext_tiny_colab.ipynb | Exists | But duplicate remains in root |
| esearch/documentation/audit_report.txt | Exists | But duplicate remains in root |
| esearch/figures/ | Created | **Empty** - gradcam_investigation/ and isualizations/ not moved |

### Duplicates in Root (Require Cleanup)
| File | Should Be | Action |
|------|-----------|--------|
| 	est_regression.py | 	ests/ | Remove root copy |
| 	rain_convnext_tiny.py | scripts/training/ | Remove root copy |
| calibrate.py | scripts/evaluation/ | Remove root copy |
| complete_c03_validation.py | scripts/evaluation/ | Remove root copy |
| make_all_summaries.py | scripts/evaluation/ | Remove root copy |
| make_c01_summary.py | scripts/evaluation/ | Remove root copy |
| make_c03_summary.py | scripts/evaluation/ | Remove root copy |
| generate_visualizations.py | scripts/visualization/ | Remove root copy |
| debug_gradcam.py | scripts/debugging/ | Remove root copy |
| dermai_convnext_tiny_colab.ipynb | esearch/notebooks/ | Remove root copy |
| udit_report.txt | esearch/documentation/ | Remove root copy |

### Missing File (Not in Root or Organized Location)
| File | Expected Location | Status |
|------|-------------------|--------|
| investigate_gradcam.py | scripts/visualization/ | **MISSING** - Not found anywhere |

### Archive Candidates (to _quarantine/)
| Path | Type | Reason |
|------|------|--------|
| debug_gradcam/ | Dir (20 PNG) | Superseded by gradcam_investigation/ |
| gradcam_outputs_new/ | Dir (9 files) | Superseded by isualizations/ |
| 	est_output.txt | File | Legacy UTF-16 log, regenerable |
| gradcam_generation.log | File | Regenerable script log |
| gradcam_investigation.log | File | Regenerable script log (empty) |
| WhatsApp Image 2026-09-28 at 8.01.26 PM (1).jpeg | File | Personal photo, documented in model_card.json |
| WhatsApp Image 2026-09-28 at 8.01.26 PM.jpeg | File | Personal photo, documented in model_card.json |
| .kilo/ | Dir | IDE artifact |
### New Files Created During This Session
| File | Purpose |
|------|---------|
| PROJECT_AUDIT.md | This audit document |
| PROJECT_CHANGELOG.md | Change log with Grad-CAM fixes |
|
egression_test_results.txt | Latest test run output |
| scripts/, 	ests/,
esearch/ | New organizational directories |

---

## 11. Image-Quality Screening (2026-10-15)

### Files and Roles
| File | Role |
|------|------|
| `quality.py` (new) | `check_image_quality(img)` - runs on the already-decoded PIL Image, no re-decode. Returns `warnings`, `is_valid`, `metrics`. |
| `config.py` | `QUALITY_*` thresholds (single source of truth) - see the "Image quality checks" section. |
| `app.py` | `/predict` and `/predict_batch` attach `quality_warnings`, `quality_metrics`, `quality_is_valid`. |
| `templates/index.html` | Active frontend: warning panels near the preview and the result; `showQuality()` renders `d.quality_warnings`. |
| `static/js/app.js`, `static/css/style.css` | Alternative (unlinked) frontend kept in sync with `showQualityWarnings()` and `.quality-warnings` styles. |
| `tests/test_quality.py` (new) | Unit + API tests for this feature. |

### Design Decisions
- **Advisory only.** Warnings never reject an image; `is_valid` is `True` for every decodable image. Only un-decodable files fail, through the existing 400 error path.
- **No duplicate decoding.** The check consumes the PIL Image the endpoint already decoded; it performs one internal resize to 256x256 and one grayscale conversion.
- **Model pipeline untouched.** Quality metrics are computed on the analysis copy only. Preprocessing, thresholds, inference, and Grad-CAM behaviour are unchanged.
- **Batch parity.** `/predict_batch` attaches quality info per prediction, preserving result order.
- **Frontend compatibility.** Existing response fields are additive-only, so older clients keep working; the active frontend reads the new fields defensively and renders nothing when the list is empty.

### Configurable Thresholds
| Setting | Default | Meaning |
|---------|---------|---------|
| `QUALITY_ENABLED` | `True` | Master switch; when false the check is a no-op. |
| `QUALITY_ANALYSIS_SIZE` | `(256, 256)` | Downscale used for metric computation. |
| `QUALITY_BLUR_THRESHOLD` | `100.0` | Laplacian variance below this -> blur warning. |
| `QUALITY_BRIGHTNESS_DARK` | `0.20` | Mean brightness below this -> dark warning. |
| `QUALITY_BRIGHTNESS_BRIGHT` | `0.95` | Mean brightness above this -> overexposed warning. |
| `QUALITY_OVEREXPOSURE_RATIO` | `0.10` | Near-white pixel fraction above this -> overexposed warning. |

### Known Limitations
- Thresholds are coarse heuristics tuned on synthetic images plus the four sample photos; they will need retuning against a representative photo set.
- `blur_score` is resolution-sensitive as a raw number (fine texture changes after downscaling) even though the resulting classification is stable.
- The near-white overexposure test cannot distinguish blown highlights from genuinely pale skin.
- Quality checks are not covered by the Grad-CAM/heatmap regression tests, which were left untouched.
- The alternative frontend (`static/js/app.js` + `static/css/style.css`) is not linked by `templates/index.html`; it received compatible handling only and is not verified in a browser.