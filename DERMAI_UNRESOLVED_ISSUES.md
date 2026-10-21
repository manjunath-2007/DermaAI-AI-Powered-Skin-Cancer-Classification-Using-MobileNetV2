# DERMAI — Unresolved Issues

**Project:** DermaAI — AI-Powered Skin Cancer Classification (MobileNetV2 + ConvNeXt-Tiny)
**Project root:** `C:\Users\Manjunath\OneDrive\Desktop\MINI3\skin-cancer-app-simple\skin-cancer-app`
**Compiled:** 2026-10-20
**Status key:** `OPEN` · `BLOCKED` (cannot be resolved without an external input) ·
`DOCUMENTED` (accepted limitation, no action proposed)

Nothing in this list was silently "fixed". Runtime behaviour, thresholds, model weights, and
published metric files were left exactly as found so that every discrepancy stays visible and
auditable.

---

## Severity summary

| Severity | Count | IDs |
|---|---|---|
| **Critical** | 1 | ISSUE-05 |
| **High** | 4 | ISSUE-02, ISSUE-03, ISSUE-04, ISSUE-06 |
| **Medium** | 6 | ISSUE-07, ISSUE-08, ISSUE-09, ISSUE-10, ISSUE-11, ISSUE-12 |
| **Low** | 6 | ISSUE-13 … ISSUE-18 |
| **Medium (new, 2026-10-20)** | 1 | ISSUE-20 |
| **Low (new, 2026-10-20)** | 1 | ISSUE-21 |
| **RESOLVED** | 3 | ISSUE-01 (2026-10-20), ISSUE-17 (2026-10-20), ISSUE-22 (2026-10-20) |

---

## ISSUE-01 — Deployed ConvNeXt checkpoint ≠ checkpoint behind the published metrics

**Severity: CRITICAL (historical)** · **Status: RESOLVED — FIXED AND VERIFIED 2026-10-20**

> **RESOLUTION (2026-10-20).** Option A was approved and applied. `config.CONVNEXT_CKPT` now
> points at `results/convnext_tiny/convnext_tiny_best.pt` (stage-2 epoch 7) and
> `config.CONVNEXT_BEST` is the fallback `results/convnext_tiny/checkpoints/stage2_epoch13.pt`.
> Verified by loading through the real path `inference.load_convnext_model()`: the live
> state-dict SHA-256 (`1af4d17e…`) matches epoch 7 and **not** epoch 13; the startup log now
> reads `Loaded checkpoint: val_auc=0.9133, stage=stage2, epoch=7`.
> `pytest` 56 passed / 36 warnings and the standalone suite 22 passed / 0 failed were re-run
> after the change. No threshold, weight, checkpoint or prediction array was modified, and no
> published number needed to change. The findings below are retained as the historical record
> of the defect and of how it was established.

**Evidence (VERIFIED BY EXECUTION)**

| | |
|---|---|
| Deployed checkpoint | `results/convnext_tiny/checkpoints/stage2_epoch13.pt` (stage2, epoch 13) — selected by `config.py:21` and loaded by `inference.py:134-161` |
| Published-metrics checkpoint | `results/convnext_tiny/convnext_tiny_best.pt` (stage2, epoch 7) — loaded by `train_convnext_tiny.py:756-766` / `:1112-1116`, recorded in `evaluation_log.txt:60-61,137` |
| Weight identity | State-dict SHA-256 differs; per-tensor equality `False`. Live application model fingerprint **matches epoch 13**, does **not** match epoch 7. |
| Pipeline validation | Epoch-7 re-evaluation reproduces the stored `val_proba.npy` / `test_proba.npy` (max \|Δ\| 5.5e-03 / 1.07e-02, fp16-vs-fp32 residual). Epoch-13 does not (max \|Δ\| 0.50 / 0.56). |

**Measured impact** at the frozen threshold 0.525, same split, same preprocessing:

| Test metric (n = 1505) | Published (epoch 7) | Deployed (epoch 13) | Δ |
|---|---|---|---|
| Accuracy | 0.860465 | 0.864452 | +0.003987 |
| Precision | 0.639286 | 0.676471 | +0.037185 |
| **Recall / cancer sensitivity** | 0.621528 | **0.559028** | **−0.062500** |
| Specificity | 0.917009 | 0.936730 | +0.019721 |
| F1 | 0.630282 | 0.612167 | −0.018114 |
| ROC-AUC | 0.892432 | 0.881342 | −0.011090 |

**Impact.** Every ConvNeXt number in the project report and in any paper draft describes a model
the application does not serve. The clinically relevant sensitivity gap is **−6.25 percentage
points**; a reader comparing headline accuracy (+0.4 points) would not notice it. The threshold
0.525 was also selected on epoch-7 validation probabilities, so it is not validated for the
deployed weights.

**Resolution options (not applied — both change runtime behaviour)**

- **A (recommended).** Set `CONVNEXT_CKPT` to `results/convnext_tiny/convnext_tiny_best.pt`.
  The published metrics, the 0.525 threshold, and the deployed model then all describe the same
  weights. No retraining required. → **APPLIED 2026-10-20 (see resolution note above).**
- **B.** Keep `stage2_epoch13.pt` and re-derive the threshold and every reported metric against
  it. The epoch-13 validation arrays are required to do this correctly (see ISSUE-19).
- **C.** Re-run training with the best checkpoint as the deployment artifact.

**Retraining required: NO.** The correct weights already exist on disk. This is a
configuration/policy defect, not a modelling defect.

---

## ISSUE-02 — The load log reports the wrong validation AUC for the deployed checkpoint

**Severity: HIGH** · **Status: PARTIALLY RESOLVED** · **Affects the FALLBACK path only**

`inference.py:157` logs `checkpoint.get("best_auc", -1)`. For `stage2_epoch13.pt` that field is
the **running best across the whole run** (0.9133, achieved at epoch 7), not the epoch-13
model's own score. The actual epoch-13 validation ROC-AUC is **0.908626** (measured;
`history_stage2.csv` records 0.908643 for epoch 13).

On startup the application therefore logged `Loaded checkpoint: val_auc=0.9133, stage=stage2,
epoch=13` — a statement that is internally inconsistent and that overstates the deployed model.

**Status after the 2026-10-20 fix:** the **primary** path now loads `convnext_tiny_best.pt`,
which stores its own score under `val_auc` and takes the `model_state_dict` branch at
`inference.py:147`. The startup log correctly reads
`Loaded checkpoint: val_auc=0.9133, stage=stage2, epoch=7`. The misleading line can therefore
only fire if the application falls back to `stage2_epoch13.pt`.

**Proposed fix (not applied):** log the checkpoint's own score, or log `best_auc` under a label
that says "best during training". Optional hardening to make a fallback load visible — log the
resolved filename — was offered to the user and **not approved** in this phase.

---

## ISSUE-03 — Threshold metadata is stale and contradicts the runtime code

**Severity: HIGH** · **Status: OPEN** (documentation only)

| Source | Claim | Actual runtime value |
|---|---|---|
| `config.py:36` | `MOBILENET_THRESHOLD = 0.22` | 0.22 — authoritative |
| `config.py:23` | `CONVNEXT_THRESHOLD = 0.525` | 0.525 — authoritative |
| `model/model_card.json:29` | `app_py_current_threshold: 0.5` | **wrong** |
| `results/convnext_tiny/metrics.json:95` | `app_py_current_threshold: 0.5`, `app_py_changed: false` | **wrong** |
| `results/convnext_tiny/threshold.json` | `app_py_remains: 0.5` | **wrong** |

`inference.py:213` and `get_model_info()` confirm the runtime uses 0.525 / 0.22.
**No threshold was changed.** Correcting the three JSON fields to 0.22 / 0.525 would remove a
documented contradiction that a reviewer will find; it changes no behaviour.

---

## ISSUE-04 — MobileNetV2 metrics cannot be independently reproduced

**Severity: HIGH** · **Status: OPEN**

`model/model_card.json` reports MobileNetV2 test metrics (accuracy 0.798007, recall 0.743056,
ROC-AUC 0.862366 at threshold 0.22) but **no full prediction array exists** in the project —
only **6** single-image predictions, which cannot reproduce a 1,505-image evaluation. The
figures are therefore *supported by artifact*, never *verified*.

Two further concerns:

1. The stored single-image predictions show MobileNetV2 scoring `cancer-melanoma.jpg` at
   **0.172773 → not_affected**. The archived evidence does not show that model correctly
   flagging the one clearly malignant sample it was given.
2. A paired statistical comparison between MobileNetV2 and ConvNeXt-Tiny is **impossible**
   without MobileNetV2 per-image predictions.

**Resolution:** re-run MobileNetV2 inference over the frozen test split and store
`test_proba.npy` / `test_true.npy` alongside the ConvNeXt arrays. No retraining needed.

---

## ISSUE-05 — The DermaAI research paper source does not exist

**Severity: CRITICAL** · **Status: BLOCKED — SOURCE PAPER MISSING**

No `.docx` / `.doc` DermaAI paper exists in the project root, `MINI3`, `research/`, or the
backup folders. The PDFs present in `MINI3` are:

| File | What it actually is |
|---|---|
| `Research Paper.pdf` | Le et al. 2020, arXiv:2009.05977v1 — a **reference** paper, *not* the DermaAI paper |
| `research implementation.pdf` | Supporting material, not the paper source |
| `Future Work.pdf` | Supporting material |
| `Layman's Language.pdf` | Supporting material |

**No paper was fabricated and no reference PDF was renamed.** `DERMAI_RESEARCH_PAPER_FINAL.docx`
was deliberately **not** produced.

**To unblock:** the user must supply the actual editable DermaAI paper source. The corrections
this audit has already identified and can then be applied with evidence-only edits (preserving
layout, figures, tables, equations, captions, references, and the medical disclaimer):

- the checkpoint/provenance mismatch (ISSUE-01) — the paper must not attribute epoch-7 metrics
  to the epoch-13 deployment;
- the stale threshold metadata (ISSUE-03);
- the calibration status (ISSUE-08);
- the external-validation status (ISSUE-10);
- the Grad-CAM localization status (ISSUE-06);
- the test-set-reuse limitation (ISSUE-14);
- the absence of significance testing (ISSUE-15).

---

## ISSUE-06 — Grad-CAM lesion localization is not established

**Severity: HIGH** · **Status: OPEN / DOCUMENTED**

The MobileNetV2 Grad-CAM output concentrates on image corners/borders for the melanoma sample.
The test suite **detects and warns about this** rather than hiding it:

```
UserWarning: heatmap_localization(cancer-melanoma.jpg): activation concentrated near
image corners/borders - localization may be spurious (see README limitation)
```

A heatmap being produced does **not** mean it localizes the lesion. The ConvNeXt Grad-CAM path
additionally uses `register_full_backward_hook`, which emits a PyTorch warning on every call
(19 occurrences across the suite) because the hook fires on output gradients rather than input
gradients.

**Impact:** no paper or report may claim that Grad-CAM localizes the lesion or that the model
"highlights the cancerous region". It must be described as an interpretability aid only.

---

## ISSUE-07 — The test suite only works from the project root

**Severity: MEDIUM** · **Status: OPEN**

`tests/test_regression.py:91` and `:146` resolve fixture images through the **relative** path
`test-images/...`. Running the suites from any other working directory fails:

```
python -m pytest <abs path>\tests -q        (run from MINI3)
  -> 22 passed, 34 errors
  -> FileNotFoundError: 'test-images/cancer-melanoma.jpg'

python <abs path>\tests\test_regression.py  (run from MINI3)
  -> exit 1, harness error on the same missing file
```

From the project root both suites pass cleanly (56 passed / 22 passed).

**Proposed fix (not applied):** resolve the fixture paths relative to `__file__` in the test
module instead of the process CWD. This touches only test files and would not affect runtime
behaviour — it was left out of this phase because the audit rules for this cycle restrict
changes to documentation and verification tooling.

---

## ISSUE-08 — Confidence calibration is not applied to the production models

**Severity: MEDIUM** · **Status: DOCUMENTED**

- `config.MOBILENET_TEMPERATURE = 1.0` — "no calibration applied".
- `inference.py:212` — `calibrated = prob  # No temperature scaling for ConvNeXt`.
- `model_card.json` contains a temperature-scaling estimate (T = 1.2543) that is explicitly
  marked `"note": "NOT applied to app.py. Provided for review only."`

Raw sigmoid outputs are therefore presented to the user as confidence and are used directly for
the risk bands. **Impact:** reported confidence and risk levels are not calibrated probabilities.

---

## ISSUE-09 — Image-quality thresholds are heuristic

**Severity: MEDIUM** · **Status: DOCUMENTED**

`config.py:89-94` sets `QUALITY_BLUR_THRESHOLD = 100.0`, `QUALITY_BRIGHTNESS_DARK = 0.20`,
`QUALITY_BRIGHTNESS_BRIGHT = 0.95`, `QUALITY_OVEREXPOSURE_RATIO = 0.10`. No human-labelled
calibration dataset was found, so these values are engineering heuristics. The blur threshold is
also explicitly scale-dependent (tuned for the 256×256 analysis size).

The quality module never rejects an image — it only attaches warnings — so this affects user
guidance, not decision correctness.

---

## ISSUE-10 — External validation is not established

**Severity: MEDIUM** · **Status: DOCUMENTED**

`model/model_card.json` records that the four `test-images/*` photographs are **not** HAM10000
images: MD5 + native-resolution pixel comparison + 256-bit dHash/aHash nearest-neighbour search
over all 10,015 HAM10000 images gave distances of 78-130 of 256 bits, i.e. indistinguishable
from random. Their filenames are therefore **not** ground truth. Two WhatsApp photographs in the
project root *are* traceable to HAM10000 (`ISIC_0027190` / mel and `ISIC_0024698` / nv), but
two samples do not constitute external validation.

**No independent external evaluation exists.** No report or paper may claim external validation.

---

## ISSUE-11 — EfficientNetB2 artifacts live outside the project and cannot be verified

**Severity: MEDIUM** · **Status: OPEN**

`skin_cancer_efficientnetb2_final.keras` (142.0 MB) and `skin_cancer_efficientnetb2_accuracy.keras`
(83.6 MB) exist at `D:\User\Downloads\`, **outside** the project. There is **no** EfficientNetB2
model, prediction array, metric file, or evaluation artifact inside the project.

The claimed EfficientNetB2 class weights (`c01_aux7 = 1.0`, `c02_pw4 = 4.0935`,
`c03_heavy_mixup = 1.0`) come from the **experiment plan** (`results/experiments/plan_c.json`),
and the project's loss is `F.binary_cross_entropy_with_logits` with optional `pos_weight` —
**no focal loss is implemented in `dermai_exp/`**. Any EfficientNetB2 performance claim is
therefore **NOT VERIFIED** and must not be presented as a project result.

(The auxiliary-loss weight `aux_weight = 0.3` for experiment `c01_aux7` **is** confirmed, but
only as an experiment configuration — not as the production model's configuration.)

---

## ISSUE-12 — Patient-level separation is not assessable

**Severity: MEDIUM** · **Status: DOCUMENTED**

HAM10000 provides `lesion_id` but no patient identifier, so patient-level leakage cannot be
ruled in or out. Lesion-level separation **is** verified (zero overlap across all three splits,
re-derived in this phase). Any statement about patient-level separation is unsupported.

---

## ISSUE-13 — Checkpoints are deserialized with `weights_only=False`

**Severity: LOW** · **Status: OPEN (hardening recommendation)**

`inference.py:145` calls `torch.load(ckpt_path, map_location=device, weights_only=False)`.
This executes arbitrary pickle code embedded in the checkpoint. The checkpoints are local and
trusted today, so there is no active exploitation path — but the call also pulls a full
optimizer/scheduler/RNG payload (324 MB) into memory that the application never uses, slowing
start-up for no benefit.

**Recommended:** load with `weights_only=True` against a whitelisted structure, or strip the
optimizer state from the checkpoint at save time. Not applied — it touches runtime loading
behaviour.

---

## ISSUE-14 — The test split has been inspected repeatedly

**Severity: LOW** · **Status: DOCUMENTED**

Across the project's development the test split has been re-evaluated multiple times (including
by this audit, which re-evaluated it for both checkpoints). This is a known, unavoidable
limitation of the workflow and must be disclosed in any paper: test-set estimates carry an
optimistic bias that cannot be quantified away.

The "test evaluated exactly once" property in `metrics.json` / `classification_report.txt`
accurately describes **one evaluation run of the epoch-7 checkpoint at the frozen threshold**;
it does not describe the project's history as a whole, and it does not apply to the deployed
epoch-13 checkpoint.

---

## ISSUE-15 — No confidence intervals, significance tests, or lesion-clustered resampling

**Severity: LOW** · **Status: OPEN**

No bootstrap intervals, no p-values, and no lesion-clustered resampling were performed in this
audit, and none are claimed. Images from the same lesion are correlated (7,470 lesions across
10,015 images), so any image-level bootstrap would understate uncertainty.

**Resolution:** paired, lesion-clustered bootstrap between ConvNeXt-Tiny and MobileNetV2 —
possible only after ISSUE-04 is fixed (MobileNetV2 per-image predictions must exist).

---

## ISSUE-16 — Missing-model handling is not verified end-to-end

**Severity: LOW** · **Status: OPEN**

`inference.load_convnext_model()` raises `FileNotFoundError` when neither checkpoint path
exists (`inference.py:139-142`). The path was **not** exercised against a live server in this
phase — doing so requires moving or hiding the checkpoints, which the audit rules forbid
("do not delete or modify model weights"). The code path is therefore
**INFERRED FROM CODE**, not verified by execution.

---

## ISSUE-17 — Browser-based PDF download is not verified

**Severity: LOW** · **Status: RESOLVED — VERIFIED 2026-10-20**

> **RESOLUTION (2026-10-20).** A real headless-Chrome run against the live Flask
> server now exercises the whole user journey and downloads the report: the
> lesion case produces a 476 255-byte PDF containing "LIKELY AFFECTED", the
> embedded heatmap image and the medical disclaimer; the out-of-domain case
> produces a 154 005-byte PDF containing "NOT ANALYSABLE" and
> "No result was produced for this image." and **no** score bars or
> "Scan confidence" line. The downloads were opened and their content checked.
> 25/25 browser checks pass. The original concern below is retained as the
> historical record.

The frontend contains client-side PDF generation code, but no real browser download was
exercised in any audit phase. **Do not claim that browser PDF download was verified.** A
headless-browser test would be required.

---

## ISSUE-18 — MobileNetV2's Grad-CAM Keras call pattern is fragile

**Severity: LOW** · **Status: OPEN**

Keras emits 12 warnings during the suite:
`The structure of 'inputs' doesn't match the expected structure. Expected: ['input_layer']
Received: inputs=Tensor(shape=(1, 224, 224, 3))`. The current Keras build tolerates it, but a
future version may not. Cosmetic today; latent breakage risk.

---

## ISSUE-19 — Epoch-13 validation probabilities are not archived

**Severity: LOW** · **Status: OPEN**

`results/convnext_tiny/` archives `val_proba_best_epoch.npy` / `val_true_best_epoch.npy`
(epoch 7) and `val_proba.npy` / `val_true.npy` (also epoch 7, from the final evaluation), but
**no prediction array exists for the epoch-13 checkpoint**. Any future threshold re-selection
for the deployed weights requires regenerating those arrays (≈5 minutes of CPU inference, as
demonstrated in this phase).

---

## ISSUE-20 — The input-domain gate refuses two genuine lesion photographs

**Severity: MEDIUM** · **Status: RESOLVED 2026-10-20 — rule changed, thresholds unchanged**

The input-domain gate added in the previous phase refuses genuine clinical photographs
of basal cell carcinoma. Measured on 29 genuine lesion photographs (the project's four
clinical test images, two phone photographs of a lesion, 13 clinical lesion photographs
from Wikimedia Commons and 11 photographs of skin):

| File | Domain distance | Decision |
|---|---|---|
| `legit_bcc_04.jpg` (basal cell carcinoma) | 0.8579 | **refused** (cut-off 0.82) |
| `legit_bcc_05.jpg` (basal cell carcinoma) | 0.8772 | **refused** (cut-off 0.82) |
| `legit_bcc_06.jpg`, `legit_bcc_08.jpg` | 0.758 / 0.795 | accepted (second opinion) |
| all other 25 genuine lesion photographs | ≤ 0.72 | accepted |

The calibration record in `results/convnext_tiny/domain_reference_meta.json` and the
comment in `config.py` stated *"real lesion photos max = 0.795"*; that figure was measured
on a 14-image sample and is falsified by these two photographs.

### Investigation (2026-10-20)

A **real** photograph corpus of 100 images was built for this issue (30 genuine
skin-lesion photographs and 70 ordinary photographs — 5 each in poster, document,
screenshot, classroom, building, scenery, furniture, book, electronics, animal, food,
vehicle, selfie and body_photo — downloaded from Wikimedia Commons). 156 synthetic
stand-ins were tried first and none of them exceeded the 0.82 cut-off (max 0.8064) while
real photographs reach 0.929, so synthetic images cannot exercise the region that
matters.

The investigation **falsified the premise of this issue and of its recommended fix
(option A)**:

| Rule | genuine lesions refused | ordinary photographs accepted | ordinary reported as cancer |
|---|---|---|---|
| original (`>0.82`, or band and second opinion `< 0.22`) | 7/30 (23.3%) | 12/70 (17.1%) | 0 |
| **option A** (second opinion may veto above 0.82) | 7/30 (23.3%) | 23/70 (32.9%) | **4** |
| **distance only, band split 0.76** (applied) | **2/30 (6.7%)** | **9/70 (12.9%)** | **0** |

* **Option A is worse on every axis.** It admits 11 more ordinary photographs
  (posters, documents, a screenshot, a classroom scene, a building, two museum interiors,
  a plate of food and three selfies) and raises the number of ordinary photographs
  reported as **cancer** from 0 to 4. It also does not reliably rescue the far lesions:
  the genuine clinical skin-tumour photograph `SkinTumors-P9280849.jpg` (distance 0.8301)
  has a MobileNetV2 probability of 0.066 and is refused under option A as well.
* **The root cause of the false rejections was not the 0.82 cut-off but the second
  opinion.** MobileNetV2 is a *cancer* classifier, not a lesion-presence detector: its
  probability is **lower on genuine lesion photographs (median 0.074) than on ordinary
  photographs (median 0.123)**. Refusing images whose second opinion was below 0.22
  therefore refused genuine lesions preferentially — 6 of the 8 genuine lesions inside
  the ambiguous band, including two images the deployed model itself calls highly
  malignant (probability 0.82 and 0.92).
* The distance to the HAM10000 manifold **is** the separating signal: inside the
  ambiguous band the genuine lesions occupy the lower half (0.70–0.77) and the ordinary
  photographs the upper half (0.72–0.82).

### Fix applied

The gate is now decided by the distance alone: analyse while the score is at most
**0.76**, refuse above it. `DOMAIN_BAND_SPLIT = (DOMAIN_SUSPICIOUS_DISTANCE +
DOMAIN_MAX_DISTANCE) / 2` is **derived** from the two existing thresholds, so no new
tuned number was introduced and **no model, weight, threshold, checkpoint or stored
prediction array was changed**. The MobileNetV2 second opinion is still computed for
images near the boundary and reported as `second_opinion_cancer_prob` for auditing, but
it can no longer refuse or admit an image.

Outcome on the 100 real photographs: ordinary photographs refused **61/70 (87.1%)**,
genuine lesion photographs accepted **28/30 (93.3%)**, **no ordinary photograph reported
as cancer**. The two genuine lesions still refused are the ones that sit furthest from
the manifold (0.8301 and 0.7658); no signal available in this project can rescue them
without re-admitting ordinary photographs.

Rejection message is now **"Invalid Image — Please upload a suitable skin-lesion image"**,
with no verdict, probability, risk level or heatmap in the payload, and the
image-quality warnings remain a separate advisory channel.

**Residual limitation:** 9 of the 70 ordinary photographs (12.9%) are still analysed and
reported as *not affected* — mostly scanned documents, posters, books and screenshots,
which are flat light-field images with dark marks and therefore genuinely close to the
dermoscopic manifold. Separating them needs a text/layout detector, which would be a new
component rather than a threshold change.

---

## ISSUE-21 — A stale development server was serving pre-gate code on port 5000

**Severity: LOW** · **Status: RESOLVED — PROCESS TERMINATED 2026-10-20**

A `python app.py` process started on **2026-10-16 22:08** (project venv) was still
listening on `0.0.0.0:5000` when this phase started. Windows allows a second socket to
bind the same port, and the **older** process kept receiving the requests. The first
browser test run therefore exercised the *pre-gate* application: the classroom
photograph was reported as "Not Affected" (the original false-positive defect) and the
response contained no `analysable` / `domain_*` fields at all.

The stale process was terminated and the port was verified to have a single listener
before the browser tests were repeated. **No code defect** — but any test run against
`localhost:5000` must first confirm which process owns the port
(`netstat -ano | findstr :5000`).

**→ RECURRED 2026-10-20 (later in the same day).** A `Start-Process` launch that failed
with a tool error left an orphaned `python app.py` running since 18:06:24, holding
`0.0.0.0:5000`. A second, correct server was then started at 18:29:08 — and **Windows
let it bind the same port while the older process kept receiving the requests**, so the
served page was the pre-edit code (the old "This image cannot be analysed" message and
the old gate). The symptom is exactly the original one: the page looks unchanged and
stale results are returned. **Always run `netstat -ano | findstr LISTENING | findstr :5000`
and compare the owning PIDs against the processes you started** before trusting any
result from `localhost:5000`.

---

## ISSUE-22 — EXIF orientation ignored and Grad-CAM not thread-safe

**Severity: MEDIUM (both) at the time of discovery** · **Status: RESOLVED — FIXED AND VERIFIED 2026-10-20**

Both were reported as secondary findings in `DERMAI_ODD_FALSE_POSITIVE_REPORT.md` §7 and
were left unchanged in that phase. Both are genuine defects and both are fixed now.

**1. EXIF orientation was ignored.** `Image.open()` does not apply the EXIF rotation
tag, so a portrait phone photograph was analysed rotated by 90° while the browser
showed the user the upright picture. The model is strongly orientation sensitive, so
the verdict changed: the melanoma test photograph scored **0.6885** upright and
**0.1194** stored sideways (orientation 6) — *Affected* became *Not Affected* — and the
Grad-CAM overlay was drawn on a rotated copy of the photo on screen.

*Fix.* `inference.apply_exif_orientation()` applies the tag; it is called when the
upload is decoded (`app.py`), in `inference.preprocess_image()`, in
`app.mobilenet_preprocess()` and in `domain.domain_scores()`, so quality screening, the
domain gate, both models and the heatmap all see the same orientation as the browser.
Images without a tag are returned unchanged (same object, identical pixels).

*Verification.* A rotated-and-tagged upload now returns the same probability as the
upright one (0.687 / 0.6987 / 0.6851 for orientations 6 / 3 / 8 against the baseline
0.6885) and the same verdict; in a real browser the heatmap of the tagged upload and
of the upright upload differ by a mean of **0.59 grey levels** (JPEG re-encoding
noise). `tests/test_exif_gradcam.py` locks this in (4 tests).

**2. Grad-CAM was not thread-safe.** `inference.generate_gradcam()` registered a
forward and a full-backward hook on the **shared** `model.features` module, called
`model.zero_grad()` and `target.backward()`, and read whatever the hooks had captured.
With `threaded=True` two concurrent requests therefore captured *each other's*
activations and gradients. A deterministic reproduction (two threads, one shared
module) shows both requests receiving the identical activation tensor. One request's
`zero_grad()` could also wipe the gradients the other was using, and 182 parameter
tensors were left holding gradients (Σ|grad| = 322 856) after every call.

*Fix.* The heatmap gradient is now taken with `torch.autograd.grad(target, features)`
on the feature map the forward pass returned. No hook is registered on the shared model
and no parameter `.grad` is written.

*Verification.* The CAM arrays are **bit-identical** to the hook implementation
(max |difference| = 0.000e+00 for 4 images × 2 target classes), the parameter
gradients stay at 0, and concurrent heatmaps equal the sequential ones. The PyTorch
"Full backward hook is firing…" warning is gone from the server log. 3 tests in
`tests/test_exif_gradcam.py` lock this in.

**Not changed:** model weights, the 0.525 / 0.22 decision thresholds and every stored
prediction array. SHA-256 and original modification times of all `.pt`, `.npy` and
`.h5` files are unchanged.

---

## Issues explicitly checked and found clean

| Check | Result |
|---|---|
| Lesion overlap between splits | **0** — re-verified by execution in this phase |
| Duplicate image IDs / (image_id, lesion_id) pairs | **0** / **0** |
| Split counts vs. the reference split | Match exactly for all three splits |
| ConvNeXt published test/val metrics vs. `metrics.json` | Match exactly |
| ConvNeXt parameter count | 27,820,897, confirmed on the live loaded model |
| Application startup, `/health`, single prediction, batch prediction, invalid-image handling, error responses | Exercised by the regression suites — 22/22 pass, 56/56 pytest pass |
| Model files intact | All checkpoints and prediction arrays byte-identical; nothing written to `results/convnext_tiny/` |
| Comprehensive audit JSON protected from the verifier scripts | SHA-256 identical before/after both verifier runs |
| No secrets / credentials in the audit artifacts | None found |
| No large files added to the working tree | Only audit artifacts added; all model files remain excluded |
