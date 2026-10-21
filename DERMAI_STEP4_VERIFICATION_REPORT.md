# DermaAI — Step 4 Verification Report

**Scope:** EfficientNetB2 training details and MobileNetV2 full-test metrics (the items left open by the previous audit phase)
**Date:** 2026-10-20
**Paper under audit:** `C:\Users\Manjunath\OneDrive\Desktop\MINI3\DermaAI_Research_Paper_Revised.FINAL(1).docx`
**Method:** read-only inspection of the original training notebook, the deployed project, and every surviving model/threshold artifact, plus independent re-evaluation of the checkpoints on the reproduced 1,505-image test split and re-derivation of the paper's bootstrap intervals.
**No modifications were made** to the paper, model artifacts, thresholds, prediction arrays, the app code, or git history. All scratch scripts and outputs were written to `C:\Users\MANJUN~1\AppData\Local\Temp\kilo\`.

---

## 0. Verdict summary

| ID | Paper claim | Verdict |
|----|-------------|---------|
| A1 | EfficientNetB2 input 260×260, batch 16, 8.14 M params | **VERIFIED** |
| A2 | EfficientNetB2 optimiser Adam, LR 1e-3 then 1e-5 | **VERIFIED** (initial LR; `ReduceLROnPlateau` decays it further — undocumented) |
| A3 | EfficientNetB2 binary focal loss, γ = 2, α = 0.25 | **VERIFIED** |
| A4 | Two-stage training, last 80 backbone layers unfrozen, BN kept frozen | **VERIFIED** |
| A5 | EfficientNetB2 max epochs 20 + 35, checkpoint on validation AUC | **VERIFIED** |
| A6 | Balanced class weights for the two baselines | **VERIFIED** |
| A7 | EfficientNetB2 augmentation "flip H/V; rot 0.15; zoom 0.15" | **VERIFIED but incomplete** (translation ±0.10 and contrast 0.20 also applied; prose mentions them, Table III does not) |
| A8 | EfficientNetB2 test metrics 82.33 / 52.84 / 71.18 / 60.65, AUC 0.8908, τ = 0.33 | **VERIFIED against notebook output**; **not independently recomputable** (final checkpoint corrupt) |
| A9 | EfficientNetB2 FP 183 / FN 83, specificity 84.96 % | **VERIFIED** |
| A10 | EfficientNetB2 sweep rows 0.40 / 0.46 / 0.50 (Table VII) | **VERIFIED** |
| A11 | Accuracy-focused EfficientNetB2 variant 84.32 % / AUC 0.8688 / τ = 0.50 / sens. 46.88 % | **VERIFIED as a notebook claim; UNVERIFIABLE independently** (preserved variant checkpoint is degenerate) |
| B1 | Notebook MobileNetV2 test metrics 76.01 / 42.01 / 66.67 / 51.54, AUC 0.8091, τ = 0.58 | **VERIFIED against notebook output**; **not reproducible from the surviving checkpoint** (different run) |
| B2 | Notebook MobileNetV2 confusion 952 / 265 / 96 / 192, specificity 78.23 % | **VERIFIED** |
| B3 | Table VII MobileNetV2 row τ = 0.50 → 73.55 / 39.85 / 75.00 / 52.05 | **VERIFIED** |
| B4 | Notebook MobileNetV2 head: GAP, BN, dropout 0.35, 128-unit ReLU, dropout 0.25, sigmoid; 2.43 M params | **VERIFIED** |
| C1 | Deployed MobileNetV2 (Table VIII) 79.80 / 48.20 / 74.31 / 58.47, AUC 0.8624, τ = 0.22, TN/FP/FN/TP 987/230/74/214 | **VERIFIED — independently reproduced exactly** |
| C2 | Deployed MobileNetV2 at τ = 0.50: 84.9 % accuracy, 49.0 % sensitivity | **VERIFIED — independently reproduced** (84.92 % / 48.96 %) |
| C3 | Deployed MobileNetV2 size ≈ 21.8 MB | **VERIFIED** (21,825,960 bytes) |
| C4 | Deployed MobileNetV2 head "global-average-pooling, a 256-unit ReLU layer, dropout and a sigmoid output" | **INCORRECT** (actual head: GAP → Dropout(0.5) → Dense(1, sigmoid); no 256-unit ReLU, no BN) |
| C5 | "only six single-image predictions are stored for MobileNetV2, so its test metrics cannot be reproduced independently" | **INCORRECT** (six single-image predictions exist, but the checkpoint and model card make the metrics fully reproducible — done here) |
| D1 | Table V bootstrap intervals for the three notebook models | **Point estimates VERIFIED exactly; intervals reproduce to within 0.1–0.4 pp, not bit-identical** |
| D2 | Table VIII bootstrap intervals for the deployed MobileNetV2 | **VERIFIED to within 0.1–0.3 pp** |
| E1 | Prior audit: "no focal loss is implemented" (as the basis for distrusting EfficientNetB2) | **INCORRECT as applied** — focal loss exists in the training notebook; it is genuinely absent from `dermai_exp/`, which is the ConvNeXt framework, not the EfficientNetB2 pipeline |
| E2 | Prior audit: EfficientNetB2 class weights taken from `plan_c.json` (1.0 / 4.0935 / 1.0) | **INCORRECT** — those entries are `arch: convnext_tiny`; the EfficientNetB2 weights are {0: 0.6213, 1: 2.5603} |

---

## 1. Primary sources located

| Artifact | Path | SHA-256 (first 16) | Size |
|---|---|---|---|
| Training notebook (canonical) | `C:\Users\Manjunath\OneDrive\Desktop\MINI3\skin_cancer1.ipynb` | `3C9753C6E036C193` | 693,096 B |
| Notebook copy | `D:\User\Downloads\skin_cancer1.ipynb` | `3C9753C6E036C193` | identical |
| Notebook copy | `D:\User\Downloads\skin_cancer1 (1).ipynb` | `3C9753C6E036C193` | identical |
| Colab export | `D:\User\Downloads\skin_cancer1 (1).ipynb.html` | — | code only, **no outputs** |
| EfficientNetB2 final checkpoint | `D:\User\Downloads\skin_cancer_efficientnetb2_final.keras` | `E85683C8858456B3` | 142,023,826 B |
| EfficientNetB2 accuracy variant | `D:\User\Downloads\skin_cancer_efficientnetb2_accuracy.keras` | `908379E942D37EB0` | 83,572,467 B |
| Notebook MobileNetV2 checkpoint | `D:\User\Downloads\skin_cancer_mobilenetv2_final.keras` | `96F6F67633541490` | 24,970,441 B |
| EfficientNet threshold file | `D:\User\Downloads\efficientnet_best_threshold.txt` | `E7461D011A69B16A` | 18 B |
| MobileNet threshold file | `D:\User\Downloads\best_threshold.txt` | `58B606EDFFE795F7` | 18 B |
| Deployed MobileNetV2 | `…\skin-cancer-app\model\skin_cancer_mobilenetv2.h5` | `2FC7E52B8F5955DA` | 21,825,960 B |
| Deployed threshold | `…\skin-cancer-app\model\threshold.txt` | — | 0.22 |

Notebook cell numbers below refer to the nbformat cell order in `skin_cancer1.ipynb` (full dump: `C:\Users\MANJUN~1\AppData\Local\Temp\kilo\nbfull\sc1_MINI3.txt`, 3,949 lines).

The notebook contains the recorded outputs, so every training detail and every metric quoted by the paper for the notebook benchmark is directly checkable. ConvNeXt-Tiny material (`dermai_convnext_tiny_colab.ipynb`, `plan_c.json`, `D:\User\Downloads\FINAL_TEST_RESULTS_VERIFIED_SPLIT.json`, the figure zips) belongs to a separate framework and was excluded except where the prior audit misattributed it (see §7).

---

## 2. EfficientNetB2 training details — VERIFIED

Paper source: §III-B/§III-C/§III-D and Table III.

| Claim | Notebook evidence | Cell |
|---|---|---|
| Input 260×260, batch 16 | `IMG_SIZE_EFF = 260`, `BATCH_SIZE_EFF = 16` | 59 |
| Parameters 8.14 M | `Total params: 8,136,186 (31.04 MB)`; trainable 364,289 / non-trainable 7,771,897 in stage 1 | 66 |
| Head: GAP → BN → Dropout 0.40 → Dense(256, relu) → BN → Dropout 0.30 → Dense(1, sigmoid) | `eff_model.summary()` matches the paper's "256-unit ReLU with two dropout layers (0.40 and 0.30) and two batch-normalisation layers" | 66 |
| Adam, LR 1e-3 (stage 1) / 1e-5 (stage 2) | `Adam(learning_rate=1e-3)` then `Adam(learning_rate=1e-5)` | 67, 71 |
| Binary focal loss, γ = 2.0, α = 0.25 | `def binary_focal_loss(gamma=2.0, alpha=0.25)` and `loss=binary_focal_loss(gamma=2.0, alpha=0.25)` | 65, 67, 71 |
| Two stages, backbone frozen first | `base_eff.trainable = False`; `eff_model.fit(..., epochs=20, class_weight=eff_class_weights)` | 66, 69 |
| Last 80 layers unfrozen, BN frozen | `for layer in base_eff.layers[:-80]: layer.trainable = False`; all `BatchNormalization` frozen → `Trainable EfficientNet layers: 63` | 70 |
| Stage 2 35 epochs | `eff_model.fit(..., epochs=35, ...)` → early stopped at epoch 29, best epoch 21 | 73 |
| Checkpoint on validation AUC | `ModelCheckpoint(monitor="val_auc", ...)` → `efficientnet_stage1.keras`, `efficientnet_best.keras` | 68, 72 |
| Balanced class weights | `compute_class_weight("balanced", …)` → `{0: 0.6213437333806062, 1: 2.560262965668371}` | 64 |
| Augmentation | `RandomFlip("horizontal_and_vertical")`, `RandomRotation(0.15)`, `RandomZoom(±0.15)`, `RandomTranslation(0.10)`, `RandomContrast(0.20)` | 60 |

**Refinements the paper does not state (documentation gaps, not errors):**

1. **Learning rate.** Table III lists "1×10⁻³, then 1×10⁻⁵". These are the *initial* learning rates. `ReduceLROnPlateau` was active in both stages: stage 1 decayed to 9×10⁻⁵ then 2.7×10⁻⁵ (cell 69), stage 2 to 6.25×10⁻⁷ (cell 73). The paper's ConvNeXt row already says "3×10⁻⁴ initial"; the two baseline rows should be worded the same way.
2. **Early stopping.** Both stages stopped early (stage 1: 17 of 20 epochs, best epoch 12; stage 2: 29 of 35, best epoch 21), so "Max epochs 20 + 35" is a ceiling, not the realised schedule.
3. **Augmentation.** Table III lists only flip/rotation/zoom for EfficientNetB2, while the pipeline also applies translation (±0.10) and contrast (0.20). §III-B does mention "translation and contrast changes" in prose, so the gap is limited to Table III (the MobileNetV2 cell has the same omission).

---

## 3. EfficientNetB2 test results — VERIFIED against the notebook, not independently recomputable

Paper source: Table IV, Table V, §IV-A, §IV-C, §IV-E, Table VII.

| Claim | Notebook evidence | Cell |
|---|---|---|
| τ = 0.33 chosen on validation F1 | grid `np.arange(0.10, 0.91, 0.01)`, best F1 → `Best validation threshold: 0.33`, `Validation F1: 0.6433` | 76 |
| Accuracy 82.33 %, Precision 52.84 %, Recall 71.18 %, F1 60.65 %, AUC 0.8908, τ = 0.33 | printed FINAL RESULTS block matches digit for digit | 78 |
| Test set = 1,505 images | `Test Images : 1505` | 54 (model), 80 |
| Non-cancer precision 0.9257 / recall 0.8496; cancer precision 0.5284 / recall 0.7118 | `classification_report` output | 80 |
| FP 183, FN 83; specificity 84.96 % | implied exactly by cell 80: TN = 1217×0.8496 = 1034, FP = 1217−1034 = 183, FN = 288×0.2882 = 83, TP = 288−83 = 205 → accuracy (1034+205)/1505 = 82.33 % ✓, precision 205/388 = 52.84 % ✓, recall 205/288 = 71.18 % ✓ | 80 |
| Sweep 0.40 → 85.18 / 61.73 / 59.38 / 60.53 | printed sweep | 88 |
| Sweep 0.46 → 86.31 / 71.81 / 46.88 / 56.72 | printed sweep | 88 |
| Sweep 0.50 → 85.85 / 75.51 / 38.54 / 51.03 | printed sweep | 88 |
| Accuracy-focused variant 84.32 %, AUC 0.8688, τ = 0.50, sens. 46.88 % | separate 7,949,050-parameter model (cell 90), BCE at LR 5e-6 for 30 epochs (95/96), results printed in cell 100 | 90, 95–101 |

**Why this cannot be independently recomputed:** the checkpoint that produced these numbers is `D:\User\Downloads\skin_cancer_efficientnetb2_final.keras`, and that file is damaged —

* 142,023,826 bytes, of which **31,054,096 U+FFFD replacement sequences** span from byte offset 14 to the end (≈ 65.6 % of the file is corrupt);
* `zipfile` refuses it: `zipfile.BadZipFile: Bad magic number for central directory`;
* `tf.keras.models.load_model` → **LOAD FAILED**; the evaluation harness reports `LOAD FAILED` for this model.

The accuracy-focused variant `skin_cancer_efficientnetb2_accuracy.keras` loads (valid zip, input 260×260, 7,949,050 params) but is **degenerate**: outputs are confined to [0.019558, 0.028404], validation AUC 0.4017, test AUC 0.4332, and every prediction is negative for any τ ≥ 0.05. It therefore cannot reproduce the notebook's 84.32 % / AUC 0.8688 either.

**Provenance conflict on the threshold file.** Notebook cell 82 writes `str(best_val_threshold)` → `0.32999999999999985`, which is what the paper quotes (τ = 0.33). The file actually sitting in `D:\User\Downloads\efficientnet_best_threshold.txt` contains `0.2799999999999999` — a value this notebook never produced (its grid minimum is 0.10 and its result is 0.33). The downloaded file therefore comes from a different run, and `efficientnet_best_threshold (1).txt` is 0 bytes.

**Consequence:** the EfficientNetB2 numbers in the paper are *verifiable as faithful transcriptions of the notebook's recorded outputs* (and internally self-consistent to the second decimal), but they are **not reproducible from any artifact that survives on this machine**. Provenance of the run that produced them cannot be established beyond the notebook file itself.

---

## 4. Notebook MobileNetV2 full-test metrics — VERIFIED against the notebook, not reproducible from the checkpoint

| Claim | Notebook evidence | Cell |
|---|---|---|
| τ = 0.58 from validation F1 | grid `np.arange(0.05, 0.96, 0.01)` → `Best validation threshold: 0.58`, `Validation F1: 0.5185` | 33 |
| Accuracy 76.01 %, Precision 42.01 %, Recall 66.67 %, F1 51.54 %, AUC 0.8091, τ = 0.58 | FINAL MODEL RESULTS block (0.7601 / 0.4201 / 0.6667 / 0.5154 / 0.8091 / 0.58) | 50, 54 |
| TN 952, FP 265, FN 96, TP 192 (total 1,505) | printed confusion matrix | 55 |
| Specificity 78.23 % | 952/1217 = 78.23 % ✓; false-alarm rate 265/1217 = 21.8 % ✓; miss rate 96/288 = 33.3 % ✓ | 55 |
| Table VII row τ = 0.50 → 73.55 / 39.85 / 75.00 / 52.05 | sweep row `0.50 | 0.7355 | 0.3985 | 0.7500 | 0.5205` | 57 |
| Head: GAP, BN, dropout 0.35, 128-unit ReLU, dropout 0.25, sigmoid; 2.43 M params | `Total params: 2,427,201` | 22 |
| Stage 2 unfreezes last 40 layers, BN frozen | `base_model.layers[:-40]` frozen, `Trainable layers: 26` | 26 |
| Max epochs 15 + 25, checkpoint on validation AUC | stage fits + `ModelCheckpoint(monitor="val_auc")` | 17–18, 24–25 |

**The surviving notebook checkpoint does not reproduce these numbers.** `D:\User\Downloads\skin_cancer_mobilenetv2_final.keras` (24,970,441 bytes, valid zip, 2,427,201 parameters, input 224×224) was re-scored on the reproduced split under both plausible preprocessing paths:

| Preprocessing | Val-best-F1 τ | Val F1 / acc / AUC | Test @ that τ | Test accuracy @ τ = 0.50 |
|---|---|---|---|---|
| `tf.image.resize` bilinear + `mobilenet.preprocess_input` (x/127.5 − 1) | **0.46** | 0.5270 / 0.7427 / 0.8000 | acc 0.7136, prec 0.3765, rec 0.7569, F1 0.5029, AUC 0.8056 (856/361/70/218) | 0.7309 |
| PIL resize 224 + x/255 | **0.44** | — | acc 0.7189, AUC 0.8099 | 0.7482 |

The notebook's own threshold file `D:\User\Downloads\best_threshold.txt` contains `0.4600000000000001` — exactly the threshold re-derived from this checkpoint. **Conclusion:** the downloaded MobileNetV2 checkpoint and threshold pair belong to a *different training run* than the one whose outputs are recorded in the notebook and quoted in the paper. The paper's 76.01 % / 0.8091 figures are faithful to the notebook but are not backed by any surviving model file.

---

## 5. Deployed MobileNetV2 (Table VIII) — independently reproduced, VERIFIED

The deployed checkpoint `…\skin-cancer-app\model\skin_cancer_mobilenetv2.h5` was re-evaluated on the reproduced split (1,505 test images, lesion-level stratified 70/15/15, `random_state=42`; split reconstructed from `D:\User\Downloads\archive\HAM10000_metadata.csv` + image folders and confirmed image-by-image against the notebook's test set):

| Paper claim (Table VIII) | Recomputed | Verdict |
|---|---|---|
| τ = 0.22 | 0.22 (argmax validation F1; validation F1 0.6451, accuracy 0.8320, precision 0.5545, recall 0.7710, AUC 0.8947; alternative best-accuracy τ = 0.44) | VERIFIED |
| Accuracy 79.80 % | 0.7980066445 → 79.80 % | VERIFIED |
| Precision 48.20 % | 0.48198 → 48.20 % | VERIFIED |
| Sensitivity 74.31 % | 0.74306 → 74.31 % | VERIFIED |
| Specificity 81.10 % | 987/1217 = 81.10 % | VERIFIED |
| F1 58.47 % | 0.58470 → 58.47 % | VERIFIED |
| ROC-AUC 0.8624 | 0.8623664749 → 0.8624 | VERIFIED |
| TN / FP / FN / TP = 987 / 230 / 74 / 214 | 987 / 230 / 74 / 214 | VERIFIED |
| At τ = 0.50: 84.9 % accuracy, 49.0 % sensitivity | 0.8491694352 → 84.92 %, recall 0.4895833 → 48.96 % | VERIFIED |
| Size ≈ 21.8 MB | 21,825,960 B = 21.83 MB | VERIFIED |

Every digit of Table VIII's MobileNetV2 column reproduces exactly from the deployed checkpoint.

**Two statements in the paper are INCORRECT:**

1. **Deployed architecture.** §IV-F: *"The deployed MobileNetV2 is a Keras model of about 21.8 MB with a global-average-pooling layer, a 256-unit ReLU layer, dropout and a sigmoid output."*
   The saved HDF5 has 157 layers and **2,259,265 parameters** (MobileNetV2 backbone 2,257,984 + 1,281 trainable). The actual head is `global_average_pooling2d → dropout(0.5) → dense(1, sigmoid)`. There is **no 256-unit ReLU layer, no batch-normalisation and only one dropout**. The same incorrect description is stored in `model/model_card.json` under `model_architecture`
   (`"MobileNetV2 (ImageNet) + GAP + Dense(256,relu) + Dropout + Dense(1,sigmoid)"`).
   Note this is *not* the notebook architecture either — the notebook head (cell 22) is GAP → BN → Dropout 0.35 → Dense(128) → Dropout 0.25 → Dense(1), which the paper describes correctly in §III-C. The deployed model is a third, smaller variant, and the paper describes it by borrowing the EfficientNetB2 head description.
2. **"only six single-image predictions are stored for MobileNetV2, so its test metrics cannot be reproduced independently"** (§IV-F).
   The six single-image predictions do exist (`model_card.json → single_image_predictions`, length 6), but the conclusion is false: the deployed checkpoint is present and was fully re-evaluable, and `model/model_card.json` additionally stores the complete test metric set, the full 332-point ROC curve, the threshold-selection record and per-diagnosis statistics. The deployed MobileNetV2 figures are the **only** model results in the paper that are independently reproducible today — they were reproduced here to the last decimal.

---

## 6. Bootstrap intervals (Tables V and VIII)

Point estimates reproduce exactly from the confusion counts. Intervals were recomputed with 2,000 percentile bootstraps at seed 42 under two implementations — multinomial resampling of the four cell counts, and resampling of 1,505 individual outcomes — and match the paper to within 0.1–0.4 percentage points:

| Model | Metric | Point (recomputed) | Paper point | Recomputed CI | Paper CI |
|---|---|---|---|---|---|
| MobileNetV2 (notebook) | Accuracy | 76.01 | 76.01 | 73.89–78.07 / 73.95–78.21 | 73.82–78.27 |
| | Precision | 42.01 | 42.01 | 37.47–46.46 / 37.53–46.74 | 37.78–46.67 |
| | Sensitivity | 66.67 | 66.67 | 61.05–71.67 / 61.11–72.00 | 61.32–72.02 |
| | Specificity | 78.23 | 78.23 | 75.89–80.54 / 75.91–80.52 | 75.80–80.57 |
| | F1 | 51.54 | 51.54 | 46.87–55.61 / 47.23–55.88 | 47.28–55.99 |
| EfficientNetB2 (notebook) | Accuracy | 82.33 | 82.33 | 80.40–84.19 / 80.40–84.25 | 80.47–84.32 |
| | Precision | 52.84 | 52.84 | 47.48–57.75 / 47.92–58.11 | 47.81–57.96 |
| | Sensitivity | 71.18 | 71.18 | 65.73–76.08 / 66.06–76.30 | 65.83–76.17 |
| | Specificity | 84.96 | 84.96 | 82.95–86.92 / 82.91–86.86 | 82.96–87.08 |
| | F1 | 60.65 | 60.65 | 55.71–64.73 / 56.20–64.97 | 56.17–64.95 |
| MobileNetV2 (deployed) | Accuracy | 79.79 / 79.82 | 79.80 | 77.81–81.73 / 77.81–81.79 | 77.81–81.86 |
| | Precision | 48.15 / 48.23 | 48.20 | 43.56–52.76 / 43.49–52.94 | 43.57–53.04 |
| | Sensitivity | 74.23 / 74.38 | 74.31 | 68.85–78.99 / 69.13–79.36 | 69.00–79.04 |
| | Specificity | 81.10 / 81.11 | 81.10 | 78.88–83.29 / 78.99–83.32 | 78.80–83.35 |
| | F1 | 58.38 / 58.48 | 58.47 | 54.11–62.45 / 54.10–62.64 | 54.13–62.81 |

**Verdict:** the interval *procedure* described in the paper is confirmed (2,000 percentile bootstraps, seed 42, validation-only thresholds) and the published intervals are consistent with it, but they are not bit-reproducible from the published confusion counts — the exact RNG call sequence is not recorded, and the intervals move by a few tenths of a point between equally valid implementations. This is a normal reproducibility limitation for bootstrap CIs, not an error. If the project wants bit-exact intervals, the bootstrap script and RNG seeding call must be archived.

---

## 7. Corrections to the previous audit phase

1. **"No focal loss is implemented" → INCORRECT as applied to EfficientNetB2.**
   `dermai_exp/engine.py:249-260` does implement only `binary_cross_entropy_with_logits` (+ optional auxiliary cross-entropy) — the prior statement about that codebase is correct. But the EfficientNetB2 in the paper was **not** trained by `dermai_exp/`; it was trained in `skin_cancer1.ipynb`, whose cell 65 defines `binary_focal_loss(gamma=2.0, alpha=0.25)` and whose cells 67 and 71 compile with it. The focal-loss claim in §III-D is therefore **verified**, and the earlier "focal loss not implemented" finding should be restated as "focal loss is absent from the ConvNeXt experiment framework, which is a different pipeline".

2. **EfficientNetB2 class weights taken from `results/experiments/plan_c.json` → INCORRECT.**
   `plan_c.json` contains five ConvNeXt-family experiments: `c01_aux7`, `c02_pw4`, `c03_heavy_mixup` and `c04_cnxts` all declare `"arch": "convnext_tiny"`, and `c05_effv2s` declares `"arch": "efficientnet_v2_s"` — a different architecture from EfficientNetB2. The `pos_weight` values 1.0 / 4.0935 / 1.0 therefore have **no relationship** to the paper's EfficientNetB2, whose actual weights are `{0: 0.6213437333806062, 1: 2.560262965668371}` (notebook cell 64, `compute_class_weight("balanced")`).

3. **"Deployed MobileNetV2 metrics cannot be reproduced independently" → INCORRECT** (see §5). This was based on the presence of only six single-image predictions in the model card and ignored both the checkpoint and the complete metric/ROC records in the same file.

4. **Preprocessing mismatch.** The notebook pipelines use `keras.applications.*.preprocess_input` (verified numerically identical to `x/127.5 − 1.0` for both MobileNetV2 and EfficientNetB2), whereas `app.py` feeds `x/255.0 ∈ [0,1]` to the deployed model. This is consistent with the paper's §IV deployment description ("MobileNetV2 receives pixels scaled to [0, 1]") and is not an error, but it must be respected when re-evaluating either checkpoint — the earlier `tools/verify_models.py` candidate preprocessors (`raw255`, `255`) for EfficientNetB2 do not match the notebook's `x/127.5 − 1.0`.

---

## 8. Artifact integrity register

| Artifact | State | Impact |
|---|---|---|
| `skin_cancer_efficientnetb2_final.keras` | **CORRUPT** — 31,054,096 U+FFFD sequences from offset 14 to EOF (≈ 65.6 % of bytes); `BadZipFile: Bad magic number for central directory`; `load_model` fails | The paper's headline EfficientNetB2 result cannot be independently recomputed |
| `skin_cancer_efficientnetb2_accuracy.keras` | Valid archive, **degenerate weights** — outputs ∈ [0.0196, 0.0284], val AUC 0.4017, test AUC 0.4332 | The 84.32 % / AUC 0.8688 accuracy-variant claim cannot be independently confirmed |
| `skin_cancer_mobilenetv2_final.keras` | Valid archive, healthy weights, but **belongs to a different run** (re-derived τ = 0.46, test acc ≈ 71–72 % at that τ) | The paper's notebook MobileNetV2 row (76.01 %, τ = 0.58) has no surviving model file |
| `efficientnet_best_threshold.txt` | Contains `0.2799999999999999`; notebook wrote `0.32999999999999985` | Threshold file does not belong to the reported run |
| `efficientnet_best_threshold (1).txt` | 0 bytes | Empty download |
| `best_threshold.txt` | Contains `0.4600000000000001`; notebook wrote `0.58` | Same run-mismatch as the checkpoint above |
| `model/skin_cancer_mobilenetv2.h5` | Valid, self-consistent with `model_card.json` | Supports the deployed Table VIII column |

---

## 9. Recommended paper amendments (not applied — no edits were made)

1. **§IV-F, deployed MobileNetV2 architecture** — replace "with a global-average-pooling layer, a 256-unit ReLU layer, dropout and a sigmoid output" with the actual head (global average pooling, dropout 0.5, sigmoid output; 2.26 M parameters). The same correction is needed in `model/model_card.json → model_architecture`.
2. **§IV-F, reproducibility sentence** — delete "whereas only six single-image predictions are stored for MobileNetV2, so its test metrics cannot be reproduced independently" and replace with the fact that the deployed MobileNetV2 test metrics *were* independently recomputed from the deployed checkpoint (79.80 % accuracy, AUC 0.8624, τ = 0.22).
3. **§III-D / Table III** — mark the baseline learning rates as *initial* (both stages additionally used `ReduceLROnPlateau`), note that early stopping ended stage 1 at epoch 17 (best 12) and stage 2 at epoch 29 (best 21), and add translation ±0.10 and contrast 0.20 to the two baseline augmentation cells.
4. **Limitations section** — add that the notebook EfficientNetB2 and MobileNetV2 checkpoints for the reported runs could not be preserved intact (the EfficientNetB2 file is corrupt, the MobileNetV2 file is from a different run), so only their notebook-recorded outputs can be audited, while the deployed MobileNetV2 remains fully reproducible.
5. **Provenance note** — the paper quotes τ = 0.33 and 0.58 for the notebook baselines, whereas the threshold files that survived on disk contain 0.28 and 0.46; stating the source of the thresholds (the notebook run) would remove ambiguity.
6. **Archive the bootstrap script** — publish the exact RNG/seeding sequence used for Tables V and VIII so the intervals become bit-reproducible.

---

## 10. Reproduction package

| Item | Path |
|---|---|
| Notebook cell dump used for every citation | `C:\Users\MANJUN~1\AppData\Local\Temp\kilo\nbfull\sc1_MINI3.txt` |
| Paper text + table extraction | `C:\Users\MANJUN~1\AppData\Local\Temp\kilo\paper_tables.txt` |
| Independent re-evaluation harness | `C:\Users\MANJUN~1\AppData\Local\Temp\kilo\step4_eval.py` |
| Re-evaluation log | `C:\Users\MANJUN~1\AppData\Local\Temp\kilo\step4_eval.log` |
| Machine-readable re-evaluation results | `C:\Users\MANJUN~1\AppData\Local\Temp\kilo\step4\step4_results.json` |
| Saved test probabilities | `C:\Users\MANJUN~1\AppData\Local\Temp\kilo\step4\*_testproba.npy` |
| Bootstrap recomputation (notebook models) | `C:\Users\MANJUN~1\AppData\Local\Temp\kilo\bootstrap_check.py` |
| Bootstrap recomputation (deployed model) | `C:\Users\MANJUN~1\AppData\Local\Temp\kilo\bootstrap_deployed.py` |
| Corruption scan / zip checks | `C:\Users\MANJUN~1\AppData\Local\Temp\kilo\check_zip.py`, `corrupt_all.py`, `scan_corrupt.py` |
| Keras/HDF5 structure readers | `C:\Users\MANJUN~1\AppData\Local\Temp\kilo\keras_meta.py`, `h5_arch.py` |

Environment: `…\skin-cancer-app\venv\Scripts\python.exe` (TensorFlow 2.21.0; the notebook was executed with TF 2.20.0), Python 3.13, scikit-learn for the bootstrap.

---

*End of Step 4. No paper, model, threshold, prediction array, application source, or git state was modified during this phase.*
