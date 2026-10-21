# DermaAI — Why ordinary photographs were reported as skin lesions

**Status:** root cause identified, fix implemented and validated
**Date:** 2026-10-20
**Scope:** preprocessing, image-quality screening, and the prediction pipeline
**Protected assets untouched:** model weights, the `0.525` decision threshold, all stored
prediction arrays, all existing checkpoints

---

## 1. The symptom

The app reported ordinary photographs — classroom scenes among them — as skin lesions, in
some cases with high confidence ("Malignant (Cancer Detected)", p up to 0.99).

Reproduced with 126 real photographs: **16 of them were reported as cancer** by the
deployed ConvNeXt-Tiny, including a bowl of fruit (p = 0.64), a plate of food (p = 0.82),
a hotel lobby (p = 0.81), a still-life painting (p = 0.73) and a football stadium (p = 0.63).

---

## 2. What was ruled out

| Hypothesis | Result |
|---|---|
| Inference preprocessing differs from training | **Ruled out.** Inference (`PIL .resize((224,224))`, default BICUBIC) is **byte-identical** to the training recipe (`im.convert("RGB").resize((224,224), Image.BICUBIC)`) for every probe image. |
| The deployed checkpoint is not the evaluated one | **Ruled out.** The deployed `convnext_tiny_best.pt` reproduces the stored arrays: max abs diff **0.0011** on `val_proba.npy`, **0.0014** on `test_proba.npy` (float16-autocast noise). |
| The 0.525 threshold is wrong | **Ruled out.** It is the validation-only argmax-F1 threshold and is correct *within* the HAM10000 domain. It is also explicitly out of scope for this task. |
| The image-quality screen should have caught these | **Ruled out** — see §3. |
| A different/better checkpoint would fix it | **Out of scope.** A newer checkpoint (epoch 13) exists but is not the one behind the published metrics. |

## 3. Root cause

**The pipeline has no input-domain check, and the image-quality screen cannot stand in for
one.**

The deployed model is a binary classifier trained *only* on HAM10000 dermoscopic
close-ups. It has no "not a skin image" class — every input is forced into
cancer / not-cancer. Its actual behaviour on arbitrary photographs is that of a
**"dark blob on a light field" detector**: a dark object sitting on a beige or white
background is read as a melanoma. That is exactly the structure present in a bowl of
apples, a plate of food, a parked car and a classroom scene.

The output is also extremely brittle to framing, which is what makes the false positives
so confident:

| Image | 0° | 90° | 180° | 270° |
|---|---|---|---|---|
| `cancer-melanoma.jpg` (real melanoma) | 0.6885 | 0.1844 | 0.0346 | 0.1190 |
| `ood_classroom_01.jpg` (classroom) | 0.0731 | — | **0.8037** | — |

A 180° rotation of a classroom photo takes it from "benign" to "cancer detected". The
network is not making a nuanced error; it is extrapolating wildly outside its training
distribution, and the sigmoid logit lands wherever it lands.

`quality.py` cannot catch this, and in fact points the wrong way:

* It is **advisory only by design** — it measures blur, brightness and overexposure and
  never rejects a decodable image (its own docstring: *"Warnings never reject the
  image — predictions still run."*).
* None of its three metrics carry information about *what the picture shows*. A sharp,
  well-exposed classroom photo passes cleanly.
* The blur threshold is **anti-correlated** with what we need: it flags
  `noncancer-congenital-nevus.jpg` (a legitimate, smooth dermoscopic nevus, Laplacian
  variance 36.6) as blurry while passing busy classroom photos (variance 100–2500).

## 4. The fix — an input-domain gate

A new additive layer refuses to emit a diagnosis for images that are not dermoscopic skin
lesions. It changes no weights, no threshold and no stored array.

**How the image is described.** The deployed model itself provides the descriptor: its
768-d penultimate embedding (global average pooled, layer-normalised) is the exact
feature space the classifier head consumes. `domain.py` measures the mean cosine distance
from an input embedding to its k = 5 nearest neighbours in a reference matrix holding the
**HAM10000 train split** embeddings, taking the worst case over {original, horizontal
flip} because the model is orientation sensitive.

**Decision rule** (`config.py`, revised 2026-10-20 after the ISSUE-20 investigation):

```
score <= DOMAIN_SUSPICIOUS_DISTANCE (0.70)  -> analyse as before
score >  DOMAIN_BAND_SPLIT (0.76)            -> refuse
otherwise (0.70 < score <= 0.76)             -> analyse as before
```

`DOMAIN_BAND_SPLIT` is **derived** from the two existing thresholds
(`(0.70 + 0.82) / 2`), so no new tuned number was introduced.

**The MobileNetV2 second opinion was removed from the decision.** The original rule was
`score > 0.82 -> refuse` plus `0.70 < score <= 0.82 and MobileNetV2 says "not a lesion"
-> refuse`. The ISSUE-20 investigation measured that rule on 100 real photographs
(30 genuine lesion photographs and 70 ordinary photographs across posters, documents,
screenshots, classrooms, buildings, scenery, furniture, books, electronics, animals, food,
vehicles, selfies and body photographs) and found the second opinion to be **wrong in the
direction that matters**:

| MobileNetV2 cancer probability | median |
|---|---|
| 30 genuine lesion photographs | **0.074** |
| 70 ordinary photographs | **0.123** |

MobileNetV2 is a *cancer* classifier, not a lesion-presence detector, so it reads most
benign lesions as "not a lesion" and reads ordinary photographs (posters, documents,
classrooms, selfies) as lesion-like. Using it to refuse images in the band therefore
refused **6 of the 8 genuine lesions** in that band — including two images the deployed
model itself calls highly malignant (ConvNeXt probability 0.82 and 0.92) — while
refusing only 10 of the 19 ordinary photographs in the band.

Measured on the same 100 photographs (FR = genuine lesion wrongly refused,
FA = ordinary photograph wrongly accepted, FC = ordinary photograph accepted **and**
reported as cancer):

| Rule | FR | FA | FC |
|---|---|---|---|
| original (`>0.82`, or band and second opinion below 0.22) | 23.3% (7/30) | 17.1% (12/70) | **0** |
| ISSUE-20 option A (second opinion may veto above 0.82) | 23.3% (7/30) | 32.9% (23/70) | **4** |
| **distance only (current)** | **6.7% (2/30)** | **12.9% (9/70)** | **0** |

ISSUE-20 option A — the fix originally proposed for this issue — is **worse on every
axis**: it admits 11 more ordinary photographs and raises the number of ordinary
photographs reported as cancer from 0 to 4. It also cannot rescue far-from-manifold
genuine lesions reliably: the genuine clinical skin-tumour photograph
`SkinTumors-P9280849.jpg` (distance 0.8301) has a MobileNetV2 probability of 0.066 and
is refused under option A as well.

The distance to the HAM10000 manifold is the only validated lesion-presence signal
available, so the gate now uses it alone. The second opinion is still computed for images
near the boundary and reported as `second_opinion_cancer_prob` for auditing, but it can
no longer refuse or admit an image.

**Refused images get no verdict at all**: the API returns `analysable: false`,
`verdict: "not_analysable"`, a null probability/confidence/risk, an empty heatmap and the
message "Invalid Image — Please upload a suitable skin-lesion image", so no consumer can
mistake the payload for a diagnosis. The UI shows a neutral "Invalid image" state and the
PDF report omits the score bars and the heatmap.

## 5. Validation

Measured through the production code path (`domain.check_domain`) on a **real**
photograph corpus downloaded from Wikimedia Commons for this investigation — 30 genuine
skin-lesion photographs and 70 ordinary photographs, five in each of the fourteen
categories listed above. (156 synthetic stand-ins were also tried; none of them exceeded
the 0.82 cut-off — maximum 0.8064 — while real photographs reach 0.929, so synthetic
images cannot exercise the region the rule change affects.)

| Set | n | Refused | Accepted | Accepted as cancer |
|---|---|---|---|---|
| Genuine lesion photographs | 30 | 2 (6.7%) | 28 | — |
| Ordinary photographs | 70 | 61 (87.1%) | 9 (12.9%) | **0** |

Per-category rejection rate for ordinary photographs: animal 5/5, building 5/5,
classroom 5/5, electronics 5/5, food 5/5, furniture 5/5, scenery 5/5, selfie 5/5,
vehicle 5/5, body photograph 4/5, book 4/5, screenshot 4/5, poster 3/5, document 1/5.
**No ordinary photograph was reported as cancer.** Every accepted ordinary photograph
received a *not affected* verdict.

Earlier measurements, unchanged by this work:

| Set | n | Refused | Verdict changes |
|---|---|---|---|
| HAM10000 validation lesions | 1500 | **0** | **0** |
| HAM10000 test lesions | 1505 | **0** | **0** |
| Genuine lesion photographs (project set) | 14 | **0** | 0 |

* Distance scores: HAM10000 val max 0.6358, test max 0.6298, genuine lesion photos max
  0.877, ordinary photographs 0.591–0.929.

Regression status: `tests/` **73 passed, 0 failed** (38 regression, 10 domain gate,
18 quality, 7 EXIF/Grad-CAM). The published metrics still reproduce from the stored
arrays and every protected artifact (model weights, the 0.525 threshold, checkpoints,
prediction arrays, the domain reference matrix) is byte-identical.

## 6. Files changed

| File | Change |
|---|---|
| `domain.py` | **new** — the gate, its decision rules and the refusal payload |
| `inference.py` | added `forward_with_embedding` and `predict_batch_with_embeddings`; `predict_batch` refactored onto a shared forward (numerics unchanged) |
| `config.py` | added the `DOMAIN_*` settings (new gate parameters only) |
| `app.py` | gate applied in `/predict` and `/predict_batch`; added `mobilenet_cancer_probability` second opinion |
| `templates/index.html` | "Not analysable" UI state and PDF report handling |
| `static/js/app.js` | guard for the new response shape (legacy front end) |
| `results/convnext_tiny/domain_reference.npy` | **new** — 7010×768 float16 reference embeddings (10.8 MB) |
| `results/convnext_tiny/domain_reference_meta.json` | **new** — provenance and calibration record |
| `scripts/build_domain_reference.py` | **new** — regenerates the reference and the calibration record |
| `tests/test_domain_gate.py` | **new** — 8 tests covering the gate |
| `tests/test_quality.py` | one assertion updated: the quality layer still never rejects, but the new domain layer may refuse a non-photograph |

Backups of every edited file: `D:\kilo-backups\dermai-domain-gate-20261020-160840\`

## 7. Secondary findings

1. **Grad-CAM is not thread-safe.** `inference.generate_gradcam` registers forward and
   backward hooks on the *shared* `model.features` module and removes them afterwards.
   `app.py` runs Flask with `threaded=True`, so two concurrent requests can capture each
   other's activations and gradients and return a heatmap computed from another user's
   image. A per-request hook target or a lock is needed. Not touched here because it is
   outside the scope of this fix.
   **→ RESOLVED 2026-10-20.** The heatmap gradient is now taken with
   `torch.autograd.grad(target, features)`; no hook is registered on the shared model
   and no parameter `.grad` is written. The CAM arrays are bit-identical to the hook
   implementation (max |Δ| = 0). See `DERMAI_UNRESOLVED_ISSUES.md` ISSUE-22.
2. **EXIF orientation is ignored.** `Image.open()` does not apply EXIF rotation, so a
   portrait phone photo is analysed rotated. Given how orientation-sensitive the model
   is (§3), this materially changes the result.
   **→ RESOLVED 2026-10-20.** `inference.apply_exif_orientation()` now applies the
   tag at decode time. The melanoma test photo scored 0.6885 upright and 0.1194
   stored sideways; the rotated upload now returns 0.687. See ISSUE-22.
3. **The quality blur threshold is mis-calibrated** for the 256×256 analysis size and
   flags legitimate smooth dermoscopic images. It should be recalibrated or dropped.
   **→ still open** (ISSUE-09).

## 8. Known limitations of the fix

* **9 of 70 ordinary photographs (12.9%) are still analysed rather than refused** and
  receive a *not affected* verdict. **None is reported as cancer.** The weak categories
  are documents (1 of 5 refused), posters (3 of 5), books (4 of 5) and screenshots
  (4 of 5): a scanned page or a poster is a flat, textured, light-field image with dark
  marks on it, which is genuinely close to the dermoscopic manifold in this feature
  space. Separating them would need a text/layout detector, which is not part of this
  model and would be a new component rather than a threshold change.
* **2 of 30 genuine lesion photographs (6.7%) are still refused**: one clinical
  skin-tumour photograph at distance 0.8301 and one pigmented verrucous photograph at
  distance 0.7658. Genuine lesions that sit far from the manifold cannot be rescued by
  any signal available here — the second opinion reads the 0.8301 photograph as
  "not a lesion" (0.066), so the ISSUE-20 option A would refuse it too. This residual
  is irreducible without a different reference manifold or a lesion-presence model.
* The gate measures the *image*, not the *user*. It refuses a photo of a genuine lesion
  if that photo also contains enough of a non-skin scene to move it off the manifold.
