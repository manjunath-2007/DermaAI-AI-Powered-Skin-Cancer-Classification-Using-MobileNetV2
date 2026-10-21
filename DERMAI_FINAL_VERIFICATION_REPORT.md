# DERMAI — Final Verification Report

**Project:** DermaAI — AI-Powered Skin Cancer Classification (MobileNetV2 + ConvNeXt-Tiny)
**Project root:** `C:\Users\Manjunath\OneDrive\Desktop\MINI3\skin-cancer-app-simple\skin-cancer-app`
**Verification date:** 2026-10-20
**Git branch / commit at verification time:** `main` / `b38ed89` (working tree has uncommitted audit artifacts only)
**Platform:** Windows, Python 3.13.3, CPU-only verification (no GPU present on the verification machine)
**Scope:** Software-engineering audit and artifact verification. **No training was executed. No model weights, checkpoints, prediction arrays, or dataset files were modified.**

---

## 0. Evidence labelling convention

| Label | Meaning |
|---|---|
| **VERIFIED BY EXECUTION** | A command was run in this verification phase and its output recorded below. |
| **SUPPORTED BY EXISTING ARTIFACT** | A claim is stated by a file that exists on disk; the file was read but the underlying event was not reproduced. |
| **INFERRED FROM CODE** | Established by reading the implementation; no runtime observation. |
| **NOT VERIFIED** | No evidence in the project supports the claim either way. |
| **BLOCKED** | The task could not be completed because a required input does not exist. |
| **DISCREPANCY FOUND** | Two authoritative sources disagree; the conflict is documented, not silently resolved. |

---

## 1. Files actually inspected in this phase

### Application code
| File | Bytes | Purpose |
|---|---|---|
| `app.py` | 16,078 | Flask application (`/`, `/health`, `/predict`, `/predict_batch`) |
| `config.py` | 8,226 | Single source of truth for paths, thresholds, quality limits |
| `inference.py` | 14,433 | PyTorch ConvNeXt backend, checkpoint loading, Grad-CAM |
| `quality.py` | 4,452 | Image-quality screening |
| `model/model_card.json` | 1,326 lines | MobileNetV2 metadata + stored single-image predictions |
| `model/skin_cancer_mobilenetv2.h5` | 21.83 MB | Legacy production MobileNetV2 checkpoint |

### Model artifacts (read only, never modified)
| File | Bytes | Notes |
|---|---|---|
| `results/convnext_tiny/convnext_tiny_best.pt` | 111,354,291 | Epoch 7 — checkpoint used by the published evaluation |
| `results/convnext_tiny/checkpoints/stage2_epoch12.pt` | 324,158,987 | Epoch 12 |
| `results/convnext_tiny/checkpoints/stage2_epoch13.pt` | 324,158,987 | Epoch 13 — **checkpoint the deployed application loads** |

### Evaluation artifacts
`results/convnext_tiny/`: `metrics.json`, `threshold.json`, `config.json`, `evaluation_log.txt`, `training_log.txt`, `history_stage1.csv`, `history_stage2.csv`, `classification_report.txt`, `val_threshold_sweep.csv`, `confusion_matrix_val.csv`, `confusion_matrix_test.csv`, `roc_curve_val.csv`, `roc_curve_test.csv`, `test_proba.npy`, `test_true.npy`, `val_proba.npy`, `val_true.npy`, `val_proba_best_epoch.npy`, `val_true_best_epoch.npy`, `confusion_matrix.png`, `roc_curve.png`, `training_curves.png`.

### Training / experiment framework
`dermai_exp/common.py`, `dermai_exp/data.py`, `dermai_exp/engine.py`, `dermai_exp/models.py`, `dermai_exp/aug.py`, `scripts/training/train_convnext_tiny.py` (1,146 lines), `results/experiments/plan_c.json`, `results/experiments/plan_b.json`.

### Verification scripts and tests
`scripts/verify_lesion_split.py`, `scripts/verify_reported_metrics.py`, `tests/test_quality.py`, `tests/test_regression.py`.

### External data actually read
| Item | Location | Size |
|---|---|---|
| HAM10000 metadata | `D:\User\Downloads\archive\HAM10000_metadata.csv` | 563,277 bytes |
| HAM10000 images | `D:\User\Downloads\archive\HAM10000_images_part_1` + `_part_2` | 10,015 JPEG files |
| 224×224 decoded image cache | `D:\dermai_cache\ham10000_224x224_u8.npy` | 1,507,538,048 bytes (uint8, shape `(10015, 224, 224, 3)`) |
| 384×384 decoded image cache | `D:\dermai_cache\ham10000_master_384x384_u8.npy` | 4,430,315,648 bytes |
| EfficientNetB2 / MobileNetV2 `.keras` (outside project) | `D:\User\Downloads\*.keras` | 83.6 MB / 142.0 MB / 25.0 MB |

---

## 2. Commands executed in this phase

All commands were run with `venv\Scripts\python.exe` from the project root.

```
git -C <project> status / log --oneline -5 / branch -vv
python scripts/verify_lesion_split.py                      -> exit 0
python scripts/verify_reported_metrics.py                  -> exit 0
python -m pytest tests -q                                  -> 56 passed, 36 warnings in 52.22 s
python tests/test_regression.py                            -> 22 passed, 0 failed, 0 errored
python <convnext checkpoint resolution script>             -> see section 5
python <dual-checkpoint evaluation script>                 -> see section 5
```

Integrity check performed around the verifier runs:

```
SHA256(DERMAI_VERIFICATION_RESULTS.json) before = E7FE79AEDAE8CAFC41B28D737C234265D54EABA503774FEF0E43BCD995DEC518
SHA256(DERMAI_VERIFICATION_RESULTS.json) after  = E7FE79AEDAE8CAFC41B28D737C234265D54EABA503774FEF0E43BCD995DEC518
AUDIT_JSON_UNCHANGED = True
```

---

## 3. Lesion-level split verification

**Status: VERIFIED BY EXECUTION**

`scripts/verify_lesion_split.py` was re-run in this phase against the actual metadata file
`D:\User\Downloads\archive\HAM10000_metadata.csv` (10,015 images, 7,470 unique lesions).

| Split | Images | Lesions | Class counts | Matches reference |
|---|---|---|---|---|
| train | 7,010 | 5,229 | 5,641 non-cancer / 1,369 cancer | yes |
| val | 1,500 | 1,120 | 1,203 non-cancer / 297 cancer | yes |
| test | 1,505 | 1,121 | 1,217 non-cancer / 288 cancer | yes |

- Train ∩ Val lesions = **0**
- Train ∩ Test lesions = **0**
- Val ∩ Test lesions = **0**
- Duplicate `image_id` = **0**; duplicate (`image_id`, `lesion_id`) pairs = **0**
- Script exit code: **0** (`STATUS: PASSED`)

**Scope limit (must be stated alongside the above):** this proves the *split logic reproduces
byte-for-byte* from the committed code with seed 42. It is **not** proof that every historical
training run consumed this exact split — HAM10000 carries no patient identifier, so
patient-level separation is **NOT ASSESSABLE** (`INFERRED FROM CODE` + artifact inspection).

---

## 4. ConvNeXt-Tiny metric verification (published metrics)

**Status: VERIFIED BY EXECUTION**

`scripts/verify_reported_metrics.py` was re-run (exit 0). It recalculates purely from the stored
prediction arrays; it does not re-run any model.

**Test split** — `test_proba.npy` / `test_true.npy`, threshold 0.525, n = 1,505

| Metric | Value |
|---|---|
| Accuracy | 0.860465 |
| Precision | 0.639286 |
| Recall / cancer sensitivity | 0.621528 |
| Specificity | 0.917009 |
| F1 | 0.630282 |
| ROC-AUC | 0.892432 |
| Confusion | TN 1116, FP 101, FN 109, TP 179 |

**Validation split** — `val_proba.npy` / `val_true.npy`, threshold 0.525, n = 1,500

| Metric | Value |
|---|---|
| Accuracy | 0.885333 |
| Precision | 0.728938 |
| Recall | 0.670034 |
| Specificity | 0.938487 |
| F1 | 0.698246 |
| ROC-AUC | 0.913319 |
| Confusion | TN 1129, FP 74, FN 98, TP 199 |

These match `results/convnext_tiny/metrics.json` exactly, and match `metrics_at()` in
`scripts/training/train_convnext_tiny.py:470`.

---

## 5. ConvNeXt checkpoint-loading investigation — **RESOLVED**

> This was the open item carried into this phase. It is now resolved with direct evidence
> from the application code and from the checkpoint files themselves.

### 5.1 What the application loads

`config.py:21-22`

```python
CONVNEXT_CKPT = os.path.join(RESULTS_DIR, "checkpoints", "stage2_epoch13.pt")
CONVNEXT_BEST = os.path.join(RESULTS_DIR, "convnext_tiny_best.pt")
```

`inference.py:134-161` resolves the primary path first and falls back to `CONVNEXT_BEST` only
if the primary does not exist. The primary file **exists**, so:

- **The deployed application loads `results/convnext_tiny/checkpoints/stage2_epoch13.pt`.**
- The fallback `convnext_tiny_best.pt` is **not** used. **VERIFIED BY EXECUTION**

### 5.2 Which checkpoint produced the published metrics

`scripts/training/train_convnext_tiny.py:1112-1116`

```python
if args.eval_only:
    banner("EVAL ONLY - loading the saved best model, no training")
    ck = r.load_best()          # -> convnext_tiny_best.pt   (line 756-766)
    report(r, ck)               # writes val_proba.npy / test_proba.npy / metrics.json
```

`results/convnext_tiny/evaluation_log.txt:60-61` and `:137` record the same fact:

```
best model    : ...\results\convnext_tiny\convnext_tiny_best.pt
               stage=stage2 epoch=7 val_auc=0.9133 saved=2026-10-04T00:56:44.412477
model         : ...\results\convnext_tiny\convnext_tiny_best.pt
```

- **All published ConvNeXt metrics describe `convnext_tiny_best.pt` (stage2, epoch 7).**

### 5.3 The two checkpoints are NOT the same weights

Both files store a 184-tensor state dict for the identical architecture, but the tensors differ.

| Property | `convnext_tiny_best.pt` | `stage2_epoch13.pt` |
|---|---|---|
| Size | 111,354,291 bytes | 324,158,987 bytes |
| State-dict key | `model_state_dict` | `model` |
| Branch taken in `inference.py` | n/a (not loaded) | `elif "model" in checkpoint` (line 153) |
| stage / epoch | stage2 / **7** | stage2 / **13** |
| Own val ROC-AUC | 0.913331 | **0.908626** (recomputed) / 0.908643 (history CSV) |
| Embedded `val_auc` / `best_auc` field | 0.913331 | 0.913331 ← *running maximum carried from epoch 7, not this checkpoint's own score* |
| SHA-256 of the state dict | `1af4d17eee76911a1582fbe4d83cf419…` | `fb57b6d6be69dded9c870ecffe5d7449…` |
| Per-tensor equality vs the other file | **False** | **False** |

### 5.4 The live application model was proved to be the epoch-13 weights

`inference.load_convnext_model()` was executed directly. The SHA-256 fingerprint of the live
`model.state_dict()` is `fb57b6d6be69dded9c870ecffe5d7449…`:

- matches `stage2_epoch13.pt` → **True**
- matches `convnext_tiny_best.pt` → **False**

Application log line produced by the real load path:

```
Loaded checkpoint: val_auc=0.9133, stage=stage2, epoch=13
ConvNeXt-Tiny model loaded successfully (params: 27.82 M)
```

> Note the log message: it prints `val_auc=0.9133` for the epoch-13 checkpoint because the code
> reads `checkpoint.get("best_auc")` (`inference.py:157`). That value is the **training run's
> best-ever AUC from epoch 7**, not the epoch-13 model's own score. The epoch-13 model's own
> validation ROC-AUC is **0.9086**. This is a **misleading log line**, recorded as an issue below.

### 5.5 Empirical consequence — measured, not estimated

Both checkpoints were evaluated on the **same** frozen validation and test splits, using the
project's own `make_split()` (seed 42) and the project's own 224×224 decoded image cache, with
the same raw-RGB `[0,255]` input contract the application uses.

**Pipeline validation first:** the epoch-7 checkpoint reproduces the stored evaluation arrays
(max |Δ| = 5.5e-03 on val, 1.07e-02 on test — the residual is fp16-GPU vs fp32-CPU arithmetic),
and the reconstructed label order is **identical** to `val_true.npy` / `test_true.npy`. The
harness is therefore faithful. The epoch-13 checkpoint does **not** reproduce those arrays
(max |Δ| = 0.50 on val, 0.56 on test) — confirming it is a different model.

**Measured metrics at the frozen threshold 0.525:**

| Metric (test set, n = 1505) | Published — epoch 7 | Deployed — epoch 13 | Δ |
|---|---|---|---|
| Accuracy | 0.860465 | 0.864452 | **+0.003987** |
| Precision | 0.639286 | 0.676471 | +0.037185 |
| **Recall / cancer sensitivity** | 0.621528 | **0.559028** | **−0.062500** |
| Specificity | 0.917009 | 0.936730 | +0.019721 |
| F1 | 0.630282 | 0.612167 | −0.018114 |
| ROC-AUC | 0.892432 | 0.881342 | −0.011090 |

| Metric (validation set, n = 1500) | Published — epoch 7 | Deployed — epoch 13 | Δ |
|---|---|---|---|
| Accuracy | 0.885333 | 0.883333 | −0.002000 |
| F1 | 0.698246 | 0.671670 | −0.026576 |
| ROC-AUC | 0.913319 | 0.908626 | −0.004692 |

Machine-readable output: `scripts/convnext_checkpoint_verification.json`.

### 5.6 Conclusion of the investigation

1. The deployed application serves the **stage2 epoch-13** weights.
2. Every published ConvNeXt metric describes the **stage2 epoch-7** weights.
3. These are different models. The published test figures therefore **do not describe the
   deployed application**.
4. The practical gap for the clinically important quantity is **−6.25 percentage points of
   cancer sensitivity** (0.6215 → 0.5590) at the deployed threshold, while overall accuracy
   moves by only +0.4 points. A reader comparing headline accuracy would not notice this.
5. The deployed checkpoint is *not* a broken model — it is a later training state that early
   stopping already passed (epoch 13 vs best epoch 7). The defect is the **selection policy**:
   `config.py` points at the last epoch rather than at the best checkpoint.

**Recommended fix (not applied — it changes runtime behaviour and must be an explicit decision):**
point `CONVNEXT_CKPT` at `convnext_tiny_best.pt` (the checkpoint the published metrics and the
0.525 threshold were derived from), **or** re-derive the threshold and all reported metrics
against the epoch-13 weights. Either way, one of the two must be made authoritative before the
metrics are quoted in a paper. This is escalated as **ISSUE-01**.

### 5.7 RESOLUTION — applied and verified 2026-10-20

**Option A was approved by the user and applied.** The first recommendation above is now the
deployed configuration.

```python
# config.py:21-29
CONVNEXT_CKPT = os.path.join(RESULTS_DIR, "convnext_tiny_best.pt")            # primary
CONVNEXT_BEST = os.path.join(RESULTS_DIR, "checkpoints", "stage2_epoch13.pt")  # fallback
```

**Verified after the change (VERIFIED BY EXECUTION, via the real `inference.load_convnext_model()` path):**

| Check | Result |
|---|---|
| Live state-dict SHA-256 | `1af4d17eee76911a1582fbe4d83cf419…` |
| Matches `convnext_tiny_best.pt` (epoch 7) | **True** |
| Matches `stage2_epoch13.pt` (epoch 13) | **False** |
| Startup log | `Loaded checkpoint: val_auc=0.9133, stage=stage2, epoch=7` (now the file's own score) |
| Parameters | 27,820,897 (27.82 M) |
| `GET /health` | 200 — threshold 0.525, `params_m` 27.82, `model_loaded` true |
| `POST /predict` (melanoma) | 200 — p = 0.6874, verdict `affected`, risk `high`, 118,408-byte heatmap |
| `POST /predict_batch` (2 images) | 200 — 0.6874 (`affected`) and 0.0653 (`not_affected`); both match the independently measured epoch-7 predictions |
| `POST /predict` (bad extension) | 400 |
| `pytest tests -q` | **56 passed, 36 warnings in 51.19 s** (exit 0) |
| `python tests/test_regression.py` | **22 passed, 0 failed, 0 errored** (exit 0) |
| `verify_lesion_split.py` / `verify_reported_metrics.py` | exit 0 / exit 0; audit JSON SHA-256 unchanged |
| Threshold / weights / prediction arrays | **unchanged** (original hashes and mtimes) |

**Consequence:** the deployed application and every published ConvNeXt-Tiny metric now describe
the same weights. **No published number, threshold, or prediction array needed to change.** The
measured effect of the swap on the frozen test split is 50 of 1,505 verdicts (3.32 %); fp16-vs-
fp32 arithmetic alone changed 0 verdicts, so the effect is the model change and not numeric
precision. Backups taken before editing: `config.py.bak_20261020`, `README.md.bak_20261020`,
`PROJECT_AUDIT.md.bak_20261020`.

---

## 6. MobileNetV2 metrics

**Status: SUPPORTED BY EXISTING ARTIFACT — NOT independently re-derived**

- Full MobileNetV2 test predictions **do not exist** anywhere in the project.
  `model/model_card.json` stores only **6** single-image predictions; 6 cannot reproduce a
  1,505-image test evaluation.
- The reported MobileNetV2 test figures (accuracy 0.798007, precision 0.481982, recall 0.743056,
  F1 0.584699, ROC-AUC 0.862366 at threshold 0.22) are therefore reported **as stated in
  `model_card.json`**, not as verified numbers.
- The 6 stored single-image predictions do include a genuine inconsistency worth noting:
  `cancer-melanoma.jpg` was scored **0.172773 (not_affected)** by MobileNetV2, i.e. the stored
  evidence does not show that model correctly flagging the melanoma sample.
- **Do not attribute these figures to ConvNeXt-Tiny and vice versa.**

---

## 7. Threshold configuration and metadata discrepancy

**Status: DISCREPANCY FOUND** (runtime code authoritative; JSON metadata outdated)

| Source | MobileNetV2 threshold | ConvNeXt threshold |
|---|---|---|
| `config.py` (runtime, authoritative) | `MOBILENET_THRESHOLD = 0.22` (line 36) | `CONVNEXT_THRESHOLD = 0.525` (line 23) |
| `model/model_card.json` | `threshold.value = 0.22`, but `app_py_current_threshold = 0.5` (line 29) | — |
| `results/convnext_tiny/metrics.json` | — | `app_py_current_threshold = 0.5` (line 95), `app_py_changed = false` |
| `results/convnext_tiny/threshold.json` | — | `app_py_remains = 0.5` |

- `app.py` / `inference.py` read the thresholds from `config.py`; the value `0.5` appearing in
  the JSON artifacts is **stale metadata**, not the runtime behaviour. **VERIFIED BY EXECUTION**
  (`inference.py:213` uses `config.CONVNEXT_THRESHOLD`; `get_model_info()` returns
  `"threshold": 0.525`).
- Runtime thresholds were **not** changed. The outdated JSON fields are documented instead.
- The threshold 0.525 was selected on the **validation split only** (argmax F1 over
  0.05…0.95 step 0.005); the test split was evaluated once at that frozen value
  (`train_convnext_tiny.py:876-907`). `metrics.json` also records that
  sensitivity ≥ 0.95 is **not achievable** on validation (max recall on the grid = 0.898990).

### Risk bands vs. classification thresholds — no conflict
`config.RISK_MEDIUM = 0.3`, `config.RISK_HIGH = 0.6` are **severity labels** applied to the
calibrated probability; the classification thresholds (0.22 / 0.525) decide the
cancer / non-cancer verdict. They are different quantities and are not in conflict.
**Action: documentation only, no code change.**

---

## 8. EfficientNetB2, auxiliary loss, TTA, parameter count

| Item | Finding | Evidence |
|---|---|---|
| EfficientNetB2 models | **NOT VERIFIED for this project.** `skin_cancer_efficientnetb2_final.keras` (142.0 MB) and `skin_cancer_efficientnetb2_accuracy.keras` (83.6 MB) exist at `D:\User\Downloads\`, **outside** the project. No EfficientNetB2 model, prediction array, metric file, or evaluation artifact exists inside the project. | `VERIFIED BY EXECUTION` (directory listing) |
| EfficientNetB2 loss / class weights | **NOT VERIFIED.** The project's ConvNeXt loss is `F.binary_cross_entropy_with_logits` with optional `pos_weight` (`dermai_exp/engine.py:249`). No focal loss is implemented in `dermai_exp/`. The values `c01_aux7 = 1.0`, `c02_pw4 = 4.0935`, `c03_heavy_mixup = 1.0` come from the **experiment plan**, not from a verified EfficientNetB2 training run inside this project. | `INFERRED FROM CODE` |
| ConvNeXt auxiliary-loss weight λ | **CONFIRMED for one experiment config only.** `results/experiments/plan_c.json` → `c01_aux7` has `aux_classes: 7, aux_weight: 0.3, aux_label_smoothing: 0.1, aux_mix: 0.3`. This is an **experiment configuration**, not the production `convnext_tiny_best.pt` configuration. | `VERIFIED BY EXECUTION` (file read) |
| Test-time augmentation | Production metrics were produced with **TTA = 1** (no augmentation) — `scripts/training/train_convnext_tiny.py` has no TTA argument at all and evaluates the plain model. The 8-view dihedral `EvalTTA` exists in `dermai_exp/aug.py` and is used by the **experiment** harness only. Published metrics therefore correspond to a single forward pass. | `INFERRED FROM CODE` + `VERIFIED BY EXECUTION` (grep) |
| Parameter count | **27,820,897** (27.82 M), all trainable, for the ConvNeXt-Tiny + GAP + Dropout(0.30) + Linear(768,1) graph. Verified for the **checkpoint the application actually loads** (`stage2_epoch13.pt`) by instantiating the live model and counting `model.parameters()`. Both checkpoints share the identical 184-tensor architecture, so the count is architecture-level, not checkpoint-specific. Cross-check: `evaluation_log.txt:42` records the same figure during training. **This is a PyTorch model, not a Keras model.** | `VERIFIED BY EXECUTION` |

---

## 9. Test results and warnings

### Full pytest suite — `python -m pytest tests -q`

```
56 passed, 36 warnings in 52.22s
```

Breakdown: `tests/test_quality.py` 18 passed / 3 warnings; `tests/test_regression.py` 38 passed / 33 warnings.
(The two suites are **not additive** with the standalone runner; the standalone runner is a separate harness.)

### Standalone regression suite — `python tests/test_regression.py`

```
convnext_tiny  11 passed, 0 failed/errored
mobilenetv2    11 passed, 0 failed/errored
TOTAL: 22 passed, 0 failed, 0 errored, 0 warnings
REGRESSION SUITE: ALL TESTS PASSED
```

### Warning inventory (documented, not dismissed)

| Count | Warning | Interpretation |
|---|---|---|
| 19 | `Full backward hook is firing when gradients are computed with respect to module outputs since no inputs require gradients` (`inference.py:295`, `test_regression.py:343`) | The Grad-CAM full-backward hook is technically not the recommended API for this graph. The tests that use it pass, but the warning indicates the hook is firing on output-gradient computation rather than input gradients. It does not invalidate the other tests; it does mean Grad-CAM implementation quality deserves a dedicated review. |
| 12 | Keras `The structure of 'inputs' doesn't match the expected structure. Expected: ['input_layer'] Received: inputs=Tensor(shape=(1,224,224,3))` | MobileNetV2 Keras graph is being called with a raw tensor instead of a KerasInput structure. Functionally tolerated by this Keras build, but it is a fragile call pattern. |
| 1 | `heatmap_localization(cancer-melanoma.jpg): activation concentrated near image corners/borders - localization may be spurious (see README limitation)` | **The known unresolved MobileNetV2 Grad-CAM corner-activation defect.** The test suite detects and reports it rather than hiding it. A heatmap being produced does **not** mean it localizes the lesion. |
| 4 | Additional `register_full_backward_hook` notices from the Grad-CAM shape-compatibility tests | Same root cause as row 1. |

### NEW FINDING — the test suite is CWD-dependent

Running `python -m pytest <abs path to tests> -q` from any directory **other than the project
root** produced **22 passed, 34 errors** with `FileNotFoundError: 'test-images/cancer-melanoma.jpg'`.
The same applies to the standalone runner. The cause is that `tests/test_regression.py` resolves
fixture images through the **relative** path `test-images/...` (`test_regression.py:91, 146`).
**Both suites must be invoked with the working directory set to the project root.**
**Action:** either run from the project root (documented in the README/test instructions), or
patch the fixtures to resolve paths relative to `__file__`. No code change was made in this
audit phase because it alters test files.

---

## 10. Reference verification

**Status: PARTIALLY VERIFIED**

- The reference used by the project is **Le et al. 2020, arXiv:2009.05977v1**. Its existence and
  bibliographic identity are established; `Research Paper.pdf` in `MINI3` is this **reference**
  paper, **not** the DermaAI project paper. **SUPPORTED BY EXISTING ARTIFACT**
- The reference paper itself evaluates ResNet50 / VGG16 / MobileNet / EfficientNetB1. It does
  **not** evaluate ConvNeXt-Tiny or EfficientNetB2, so it cannot be cited as direct
  methodological precedent for the project's model choices. **SUPPORTED BY EXISTING ARTIFACT**
- **The DermaAI reference list could not be verified** because the DermaAI paper source does not
  exist (section 12). No DOI, no plagiarism score, and no reference-by-reference check can be
  claimed.

---

## 11. Statistical limitations

**Status: limitations disclosed; no significance testing performed**

- The **test split has been inspected repeatedly** across the project's development. This is
  acknowledged as a limitation and cannot be undone; it inflates the risk of optimistic test
  estimates.
- Bootstrap or other resampling intervals, where any exist, treat **images** as independent.
  HAM10000 contains **7,470 lesions across 10,015 images**, so images from the same lesion are
  correlated. Lesion-clustered resampling would be the correct method.
- **No formal significance test, p-value, or confidence interval was computed** in this audit,
  and none is claimed. Paired comparison between ConvNeXt-Tiny and MobileNetV2 is **not
  possible** because MobileNetV2 per-image predictions do not exist in the project.
- The validation split was used repeatedly for threshold selection and checkpoint selection.
  The test split was evaluated once per checkpoint in `train_convnext_tiny.py`, but the deployed
  checkpoint and the evaluated checkpoint are different (section 5), so the "single test
  evaluation" property applies to the **epoch-7** checkpoint only.

---

## 12. Research-paper DOCX — **BLOCKED, source paper missing**

**No DermaAI research paper source exists.** The audit searched the project root, the `MINI3`
directory, `research/`, and backup folders for `.docx`/`.doc` files and found none.

Files that *are* present in `MINI3` are:
- `Research Paper.pdf` — the **reference** paper (Le et al. 2020), **not** the DermaAI paper
- `research implementation.pdf`
- `Future Work.pdf`
- `Layman's Language.pdf`

Per the audit rules, **no paper was fabricated, no reference PDF was renamed, and no substitute
document was generated.** `DERMAI_RESEARCH_PAPER_FINAL.docx` was **not** created, because doing
so would require inventing the DermaAI method, results, and citations.

**To unblock:** the user must supply the actual editable DermaAI paper source (`.docx`).
Once supplied, the corrections identified by this audit — most importantly the
checkpoint/provenance mismatch in section 5 and the threshold metadata in section 7 — can be
applied to it with evidence-only edits that preserve layout, figures, tables, equations,
captions, references, and the medical disclaimer.

---

## 13. Consolidated status

| Item | Status |
|---|---|
| Lesion-level splitting | **VERIFIED** |
| Zero lesion overlap | **VERIFIED** |
| ConvNeXt published metrics (epoch-7 checkpoint) | **VERIFIED** |
| ConvNeXt checkpoint-loading investigation | **RESOLVED AND FIXED 2026-10-20 — see section 5.7** |
| Deployed ConvNeXt behaviour | **VERIFIED — the app now serves the epoch-7 weights, i.e. exactly the checkpoint the published metrics describe** |
| MobileNetV2 metrics | **SUPPORTED BY EXISTING ARTIFACT — NOT re-derived** |
| Threshold configuration | **DISCREPANCY FOUND** (metadata only; runtime code is correct) |
| EfficientNetB2 claims | **NOT VERIFIED** (artifacts live outside the project) |
| Parameter count | **VERIFIED** (27,820,897) |
| Tests | **pytest: 56 passed, 36 warnings (52.22 s). Standalone: 22 passed, 0 failed.** |
| References | **PARTIALLY VERIFIED** |
| Statistical significance | **NOT VERIFIED — none performed, none claimed** |
| External validation | **NOT ESTABLISHED** — `test-images/*` are external clinical photographs that `model_card.json` proves are *not* HAM10000 images (hash + pixel comparison over all 10,015 images) |
| Confidence calibration | **NOT APPLIED** to the production models — `config.MOBILENET_TEMPERATURE = 1.0`, and `inference.py:212` states `calibrated = prob  # No temperature scaling for ConvNeXt`. The calibration estimate in `model_card.json` is explicitly marked "NOT applied to app.py". |
| Image-quality thresholds | **HEURISTIC** — `config.py:89-94` thresholds were tuned without a human-labelled calibration set. |
| Grad-CAM lesion localization | **NOT ESTABLISHED** — the MobileNetV2 corner-activation defect is unresolved and is actively warned about by the test suite. |
| Research paper DOCX | **BLOCKED — source paper missing** |
| Retraining required | **NO** — the epoch-7 checkpoint already exists and reproduces the published metrics. A configuration change (which checkpoint `config.py` points at) was required; **that change has been applied and verified (section 5.7)**. |

---

## 14. Medical-use disclaimer (restated)

DermaAI is a **research / demonstration decision-support application**. It is **not** a medical
diagnostic device, is not a substitute for professional medical diagnosis, and must not be used
to make clinical decisions. Any skin concern requires assessment by a qualified dermatologist.
`config.DISCLAIMER` carries the in-app equivalent of this statement.

---

## 15. Safety confirmation for this phase

- No training executed; no model retrained.
- No model weights, checkpoints, prediction arrays, dataset files, or notebooks modified.
- No files deleted; no research artifacts removed.
- No commit, no push, no force-push, no history rewrite, no deployment.
- Runtime thresholds and model weights left exactly as found.
- **Files written by the audit:** the audit reports, the merged audit JSON, one machine-readable
  evidence artifact (`scripts/convnext_checkpoint_verification.json`), and — after explicit
  user approval of Option A — `config.py` (checkpoint paths only), `README.md` and
  `PROJECT_AUDIT.md` (documentation only). Backups of all three edited files were taken first
  (`*.bak_20261020`). `inference.py` and `app.py` were **not** modified.
