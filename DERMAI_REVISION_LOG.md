# DERMAI — Revision Log

**Project:** DermaAI — AI-Powered Skin Cancer Classification (MobileNetV2 + ConvNeXt-Tiny)
**Project root:** `C:\Users\Manjunath\OneDrive\Desktop\MINI3\skin-cancer-app-simple\skin-cancer-app`
**Log scope:** Every file created, modified, backed up, or restored during the DermaAI audit
cycle from 2026-10-17 through 2026-10-20 (this phase).

**Standing constraints observed throughout the whole cycle:**
no training, no retraining, no modification of model weights / checkpoints / prediction arrays /
dataset files, no deletion of research artifacts, no commit, no push, no force-push, no history
rewrite, no deployment, no fabricated paper or metrics.

---

## Legend

| Tag | Meaning |
|---|---|
| `CREATED` | New file written during the audit |
| `MODIFIED` | Existing file edited, with the change described |
| `BACKED UP` | Original preserved as a `.bak_*` copy before editing |
| `RESTORED` | File returned to a prior good state after accidental damage |
| `REVERTED` | An incorrect change was undone |
| `UNCHANGED` | Inspected and deliberately left alone |

---

## 2026-10-17 — Phase A: consolidated project report

| File | Action | Detail |
|---|---|---|
| `DERMAI_FINAL_PROJECT_REPORT.md` | `CREATED` | 38,563 bytes. Consolidated technical report (34 required sections) with an explicit evidence-label scheme (`VERIFIED BY EXECUTION` / `SUPPORTED BY EXISTING ARTIFACTS` / `INFERRED FROM CODE` / `NOT VERIFIED` / `BLOCKED`). |
| `PROJECT_COMPLETION_REPORT.md` | `MODIFIED` | The report claimed "18 passed, 3 warnings" for the pytest suite. Actual measured result at that time was **56 passed, 36 warnings**. The figure was corrected to the measured value. |
| `README.md` | `MODIFIED` | Corrected test path, Grad-CAM target-layer description, and test instructions. |

### Test results recorded at that point
- `pytest tests/ -v` → **56 passed, 36 warnings**
- `python tests/test_regression.py` → **22 passed, 0 failed, 0 errored**

---

## 2026-10-17/18 — Phase B: verification tooling

| File | Action | Detail |
|---|---|---|
| `scripts/verify_lesion_split.py` | `CREATED` | Recomputes the lesion-level stratified 70/15/15 split (seed 42) from `HAM10000_metadata.csv`, asserts zero lesion overlap across splits, zero duplicate image IDs, and that image/lesion/class counts match the reference split. Exits nonzero on integrity failure. |
| `scripts/verify_reported_metrics.py` | `CREATED` | Recalculates ConvNeXt-Tiny test/validation metrics from the saved `.npy` prediction arrays and reports MobileNetV2 figures as *supported-by-artifact* because no full prediction array exists for it. Does not fabricate predictions. |
| `scripts/split_verification_results.json` | `CREATED` | Machine-readable split-verification output. |
| `scripts/split_verification_results.json.bak_20261017` | `BACKED UP` | Preserved copy of the earlier split-verification output. |
| `DERMAI_VERIFICATION_RESULTS.json.bak_20261017` | `BACKED UP` | Snapshot of the comprehensive multi-phase audit JSON, taken before any tooling change could touch it. This snapshot later proved to be the recovery point for the comprehensive report. |

---

## 2026-10-20 — Phase C: tooling defect fixes

### Defect 1 — the metrics verifier overwrote the comprehensive audit JSON

`scripts/verify_reported_metrics.py` originally wrote its machine-readable output to
`DERMAI_VERIFICATION_RESULTS.json` — i.e. running the Phase-3 verifier **destroyed the
multi-phase audit report** that the audit was supposed to be accumulating.

- `scripts/verify_reported_metrics.py.bak_20261020` — `BACKED UP` (8,747 bytes) before editing.
- `scripts/verify_reported_metrics.py` — `MODIFIED`:
  - output path changed from `DERMAI_VERIFICATION_RESULTS.json` to
    `scripts/metrics_verification_results.json` (line 201), so the comprehensive audit JSON can
    never be clobbered by this script again;
  - docstring updated to state the new output location (lines 22-23).
- `scripts/metrics_verification_results.json` — `CREATED` as the verifier's new, dedicated
  output file.

### Defect 2 — the verifier always exited 0

The verifier reported integrity failures (missing or length-mismatched ConvNeXt `.npy` pairs) as
text and still returned a success exit code, so a broken artifact set could never fail CI.

- `scripts/verify_reported_metrics.py` — `MODIFIED`:
  - a `failures` list is now accumulated and `sys.exit(1)` is raised when a **required**
    ConvNeXt prediction pair is missing or length-mismatched (lines 215-229);
  - MobileNetV2's absent predictions remain a non-fatal, explicitly-reported condition.

### Defect 3 — the comprehensive audit JSON had been damaged and could not be parsed

Running the unfixed verifier had already overwritten `DERMAI_VERIFICATION_RESULTS.json`. The
recovery copy `DERMAI_VERIFICATION_RESULTS.json.bak_20261017` itself contained a JSON syntax
error (a numeric value followed by an inline English parenthetical comment), so a naive restore
would have produced an unparseable file.

- `DERMAI_VERIFICATION_RESULTS.json` — `RESTORED` from the backup, then repaired and rewritten
  as strict JSON, preserving **all** phases 1-7 (`phase1_inventory`,
  `phase2_lesion_split_verification`, `phase3_metric_verification`,
  `phase4_technical_inconsistencies`, `phase5_statistical_validity`, `phase6_references`,
  `phase7_paper_changes`, `overall_status`).
- Verified after repair by parsing the file with a strict JSON parser.

### Defect 4 — the metrics verifier's own results file was clobbered by a later run

The verifier had written its output over the comprehensive JSON, so its own dedicated output
file did not exist. It was regenerated by the fixed script.

- `scripts/metrics_verification_results.json` — `CREATED` / regenerated (2,338 bytes).

### Phase C verification runs (2026-10-20)

| Command | Result |
|---|---|
| `python scripts/verify_lesion_split.py` | `STATUS: PASSED`, exit **0** |
| `python scripts/verify_reported_metrics.py` | ConvNeXt test + val metrics reproduced, exit **0** |
| SHA-256 of `DERMAI_VERIFICATION_RESULTS.json` before/after both runs | **identical** (`E7FE79AE…DEC518`) → the comprehensive audit JSON is demonstrably protected |

---

## 2026-10-20 — Phase D: ConvNeXt checkpoint-loading investigation (NEW)

This phase resolved the outstanding ConvNeXt checkpoint question. Nothing in the project was
modified to achieve it; the investigation was read-only apart from one new evidence artifact.

| File | Action | Detail |
|---|---|---|
| `scripts/convnext_checkpoint_verification.json` | `CREATED` | Machine-readable record of the dual-checkpoint evaluation: per-checkpoint metadata, the pipeline-validation deltas against the stored prediction arrays, and the full metric sets for both checkpoints. |

### What was established

1. `config.py:21` points `CONVNEXT_CKPT` at `results/convnext_tiny/checkpoints/stage2_epoch13.pt`;
   that file exists, so the fallback `convnext_tiny_best.pt` is never used.
2. `scripts/training/train_convnext_tiny.py:1112-1116` (and `evaluation_log.txt:60-61,137`)
   show that the published evaluation loaded `convnext_tiny_best.pt` (stage2, epoch 7).
3. The two checkpoints hold **different weights** (different SHA-256 state-dict fingerprints,
   per-tensor equality `False`).
4. The live application model fingerprint matches `stage2_epoch13.pt` exactly and does **not**
   match `convnext_tiny_best.pt` → the deployed weights are the epoch-13 ones.
5. Evaluating both on the same frozen val/test splits produced a measured gap at threshold 0.525:
   **test recall 0.621528 → 0.559028 (−0.0625)**; test F1 0.630282 → 0.612167;
   test ROC-AUC 0.892432 → 0.881342.

**Conclusion:** the published ConvNeXt metrics describe a different checkpoint from the one the
application serves. Escalated as ISSUE-01 in `DERMAI_UNRESOLVED_ISSUES.md`. **No runtime change
was applied** — choosing the authoritative checkpoint is an explicit product decision, not an
audit action.

---

## 2026-10-20 — Phase E: audit deliverables

| File | Action | Detail |
|---|---|---|
| `DERMAI_FINAL_VERIFICATION_REPORT.md` | `CREATED` | Consolidated verification report: files inspected, commands executed, split verification, metric verification, the resolved checkpoint investigation, threshold discrepancy, risk-band interpretation, EfficientNetB2 / auxiliary-loss / TTA findings, parameter count, test results and warning inventory, reference status, statistical limitations, the research-paper blocker, and a consolidated status table. |
| `DERMAI_REVISION_LOG.md` | `CREATED` | This file. |
| `DERMAI_UNRESOLVED_ISSUES.md` | `CREATED` | Open issues with severity, evidence, impact, and proposed resolution. |
| `DERMAI_VERIFICATION_RESULTS.json.bak_20261020_pre_phase8` | `BACKED UP` | Snapshot of the comprehensive audit JSON taken immediately before the phase-8/phase-9 merge. |
| `DERMAI_VERIFICATION_RESULTS.json` | `MODIFIED` | **Merged, not replaced.** All phases 1-7 preserved verbatim; a dated `phase8_fresh_reverification_2026_10_20` section, a `phase9_convnext_checkpoint_deployment_investigation` section, and a reconciled `overall_status` (21 status entries) were added. The file was re-parsed with a strict JSON parser after writing, and the merge script asserted that every original top-level key and the original phase-2/3/7 values were still present and unchanged. |

---

## 2026-10-20 — Phase F: ISSUE-01 fix applied (approved Option A)

The user reviewed the checkpoint investigation and explicitly approved **Option A**: make
`convnext_tiny_best.pt` (stage-2 epoch 7) the primary ConvNeXt checkpoint and
`checkpoints/stage2_epoch13.pt` (epoch 13) the fallback. Every file below was backed up
**before** editing.

| File | Action | Detail |
|---|---|---|
| `config.py.bak_20261020` | `BACKED UP` | Pre-edit copy, SHA-256 verified against the source. |
| `config.py` | `MODIFIED` | `CONVNEXT_CKPT` → `RESULTS_DIR/convnext_tiny_best.pt` (was `checkpoints/stage2_epoch13.pt`); `CONVNEXT_BEST` → `RESULTS_DIR/checkpoints/stage2_epoch13.pt` (was `convnext_tiny_best.pt`). Explanatory comments added. **Threshold, input size, device, risk bands and every other setting untouched.** |
| `README.md.bak_20261020` | `BACKED UP` | Pre-edit copy, SHA-256 verified. |
| `README.md` | `MODIFIED` | Lines 56-57 (directory tree), 161 (model-information table), 176-177 (checkpoint-requirements table), 179 (load-order sentence) now describe `convnext_tiny_best.pt` as primary and `stage2_epoch13.pt` as fallback, and state that the primary produced every published metric and the 0.525 threshold. |
| `PROJECT_AUDIT.md.bak_20261020` | `BACKED UP` | Pre-edit copy, SHA-256 verified. |
| `PROJECT_AUDIT.md` | `MODIFIED` | Lines 49-50 (directory tree), 137-138 (critical-files table), 349-350 ("must not be moved" risk table), 404-405 (critical-path dependency block) updated to match `config.py`. Re-applied at byte level so the file's pre-existing mixed CRLF/LF distribution (46 / 485) is preserved exactly — the diff contains only the 8 intended content changes. |
| `DERMAI_VERIFICATION_RESULTS.json.bak_20261020_pre_phase10` | `BACKED UP` | Snapshot taken immediately before the phase-10 merge. |
| `DERMAI_VERIFICATION_RESULTS.json` | `MODIFIED` | Added `phase10_checkpoint_mismatch_fix_2026_10_20`; phases 1-9 preserved verbatim; `overall_status` updated for `deployed_model_metrics`, `convnext_checkpoint_loading`, `tests`, `files_modified` and `git_state_at_verification`. Re-parsed with a strict JSON parser; all original top-level keys asserted present. |
| `DERMAI_FINAL_VERIFICATION_REPORT.md` | `MODIFIED` | Added section 5.7 (resolution + post-fix verification table); updated the consolidated status rows and the safety confirmation. |
| `DERMAI_UNRESOLVED_ISSUES.md` | `MODIFIED` | ISSUE-01 marked **RESOLVED — FIXED AND VERIFIED 2026-10-20** with the resolution note; ISSUE-02 marked **PARTIALLY RESOLVED** (fallback path only); severity table updated. |

### Files deliberately NOT changed in Phase F

| File | Reason |
|---|---|
| `inference.py` | The resolution order at `inference.py:134-142` already prefers the primary and falls back correctly. No code change needed. |
| `app.py` | No defect implicated. |
| `config.py` (thresholds) | `CONVNEXT_THRESHOLD = 0.525`, `MOBILENET_THRESHOLD = 0.22`, `RISK_MEDIUM = 0.3`, `RISK_HIGH = 0.6` all unchanged. The 0.525 threshold was selected on the epoch-7 validation predictions, so it is now exactly correct for the deployed model. |
| `results/convnext_tiny/*.pt` | No checkpoint or weight file modified. SHA-256 and original mtimes (2026-10-04) confirmed unchanged for `convnext_tiny_best.pt`, `stage2_epoch12.pt` and `stage2_epoch13.pt`. |
| `results/convnext_tiny/*.npy` | All four prediction arrays unmodified; original mtimes (2026-10-04 01:25:59) confirmed. |
| `model/skin_cancer_mobilenetv2.h5` | Unmodified. |

### Phase F verification results

| Check | Result |
|---|---|
| `inference.load_convnext_model()` (real path) | Live state-dict SHA-256 `1af4d17e…` **matches epoch 7**, does **not** match epoch 13 |
| Startup log | `Loaded checkpoint: val_auc=0.9133, stage=stage2, epoch=7` |
| `GET /health` | 200 — threshold 0.525, params 27.82 M, `model_loaded` true |
| `POST /predict` (melanoma) | 200 — p = 0.6874, `affected`, risk high, heatmap 118,408 B |
| `POST /predict_batch` (2 images) | 200 — 0.6874 / 0.0653, matching the independently measured epoch-7 predictions |
| `POST /predict` (bad extension) | 400 |
| `pytest tests -q` | **56 passed, 36 warnings in 51.19 s**, exit 0 |
| `python tests/test_regression.py` | **22 passed, 0 failed, 0 errored**, exit 0 |
| `verify_lesion_split.py` / `verify_reported_metrics.py` | exit 0 / exit 0; audit JSON SHA-256 unchanged across both runs |
| Git | 3 tracked files modified, 17 untracked audit artifacts. **No commit, no push, no deployment.** |

### Net effect

The deployed application and every published ConvNeXt-Tiny metric now describe the same
weights. No published number, threshold, or prediction array needed to change.

---

## 2026-10-20 — Phase G: end-to-end application test, EXIF orientation and Grad-CAM thread safety

This phase executed the application end to end against a real-photograph corpus (125
photographs: 29 genuine lesion photos, 12 classroom scenes, 20 tables, 8 books, 36
scenery photographs, 20 defocused photographs), plus corrupted uploads and invalid file
types; verified the frontend and the PDF export in a real headless browser; and fixed the
two defects that this exposed. Every file below was backed up **before** editing
(`D:\kilo-backups\dermai-step3-20261020-174500\`).

| File | Action | Detail |
|---|---|---|
| `inference.py` | `MODIFIED` | Added `apply_exif_orientation()` (honours the EXIF orientation tag, no-op when there is none) and applied it in `preprocess_image()`. Replaced the hook-based Grad-CAM with `torch.autograd.grad(target, features)`: no hook is registered on the shared model and no parameter `.grad` is written. The CAM arrays are bit-identical to the previous implementation (max \|Δ\| = 0.000e+00 over 4 images × 2 target classes). |
| `app.py` | `MODIFIED` | Applies the EXIF orientation to every decoded upload (single and batch), so quality screening, the domain gate, both models and the heatmap all use the orientation the browser displays. The input-domain gate now runs **before** the prediction, so no Grad-CAM backward pass is spent on an image that is going to be refused (the API response is byte-for-byte the same). Batch items now always carry the `heatmap` key so every item of a batch has the same shape as the single-image response. |
| `domain.py` | `MODIFIED` | `domain_scores()` analyses the EXIF-oriented image, so the gate, the flip view and the model all agree. |
| `tests/test_exif_gradcam.py` | `CREATED` | 7 tests: EXIF no-op, EXIF idempotency, the API honouring the EXIF orientation, no hooks left on the shared model, no parameter gradients left behind, concurrent heatmaps equal to the sequential ones, and Grad-CAM determinism. |
| `README.md` | `MODIFIED` | Documented the EXIF handling (with the measured 0.6885 → 0.1194 → 0.687 numbers), the thread-safe Grad-CAM, the new test file, the current test results and the new false-reject limitation. |
| `DERMAI_UNRESOLVED_ISSUES.md` | `MODIFIED` | ISSUE-17 resolved (browser PDF verified), ISSUE-20 opened (the gate refuses two genuine lesion photographs), ISSUE-21 added (stale server on port 5000) and resolved, ISSUE-22 added and resolved (EXIF + Grad-CAM thread safety). |
| `DERMAI_ODD_FALSE_POSITIVE_REPORT.md` | `MODIFIED` | §7 secondary findings 1 and 2 marked resolved; §8 limitation corrected — the "0.025 margin over the worst genuine lesion photograph" does not hold. |
| `DERMAI_REVISION_LOG.md` | `MODIFIED` | This entry. |
| `DERMAI_VERIFICATION_RESULTS.json` | `MODIFIED` | **Merged, not replaced.** Phases 1-10 preserved verbatim; a dated `phase11_end_to_end_test_exif_gradcam` section added. Re-parsed with a strict JSON parser; every original top-level key asserted present. |

### Phase G test results (all actually executed)

| Check | Result |
|---|---|
| API matrix, 125 real photographs | 125/125 HTTP 200, schema consistent, 0 errors; 69 refused by the domain gate, 0 ordinary photograph reported as cancer |
| Corrupted uploads (8 variants) | 8/8 → **400** with a readable error; a valid PNG renamed `.jpg` is accepted (200), as it should be |
| Invalid file types (14 variants) | 11/11 rejected with **400**; `uppercase.JPEG`, `mixedcase.JpG` and `spaces .jpg` correctly accepted |
| Batch endpoint | mixed batch 200, corrupt member 400, invalid member 400, empty list 400, missing field 400, 65 files 400 |
| `GET /health`, `GET /`, unknown route | 200 / 200 / 404 |
| Headless-Chrome frontend run | **25/25 checks pass** — result panel, heatmap, "Not analysable" state, both PDF variants downloaded and content-checked, client-side type and size validation, corrupted upload error, EXIF alignment, no page exceptions, no unexpected console errors |
| `pytest tests -q` | **71 passed, 32 warnings in 80.36 s**, exit 0 (was 56 before this phase) |
| `python tests/test_regression.py` | **22 passed, 0 failed, 0 errored**, exit 0 |
| `scripts/verify_reported_metrics.py` | exit 0 — the published ConvNeXt test and val metrics still reproduce from the stored arrays |
| Audit JSON across the verifier run | SHA-256 identical before and after |
| Model weights / prediction arrays | SHA-256 and original mtimes of every `.pt`, `.npy` and `.h5` unchanged (2026-09-28 / 2026-10-04) |
| Git | 6 tracked files modified, 1 new test file, no commit, no push |

### Two genuine bugs found and fixed

1. **EXIF orientation ignored** (ISSUE-22). The melanoma test photograph scored 0.6885
   upright and **0.1194** when stored sideways with an orientation-6 tag — *Affected*
   became *Not Affected*, and the heatmap overlay was rotated relative to the photo the
   user sees. Fixed by applying the EXIF orientation on the server; the rotated upload
   now returns 0.687 / 0.6987 / 0.6851 for orientations 6 / 3 / 8 against the 0.6885
   baseline, and the browser heatmaps differ by a mean of 0.59 grey levels.
2. **Grad-CAM not thread-safe** (ISSUE-22). The hook-based implementation let two
   concurrent requests capture each other's activations (demonstrated deterministically:
   both requests received the identical activation tensor) and left 182 parameter tensors
   holding gradients (Σ\|grad\| = 322 856) after every call. Replaced with
   `torch.autograd.grad`; the heatmaps are bit-identical and no shared state is touched.

### One defect found and deliberately NOT changed

**ISSUE-20 — the domain gate refuses two genuine lesion photographs** (clinical
basal-cell-carcinoma photos, distance 0.858 and 0.877 against the 0.82 cut-off). The
overlap with the ordinary photographs (0.79–0.94) means no cut-off separates them, and
lowering it re-admits the classroom photographs that the gate exists to remove. The
thresholds were left exactly as they are and the finding is documented with three
options in `DERMAI_UNRESOLVED_ISSUES.md`.

---

## Files deliberately NOT modified

| File | Reason |
|---|---|
| `config.py` | Threshold values (0.22 / 0.525) and checkpoint paths are authoritative runtime configuration. Changing them would change application behaviour; the discrepancies are documented instead. |
| `inference.py` | Contains the `weights_only=False` pickle load and the misleading `best_auc` log line — both recorded as issues, neither silently patched. |
| `app.py`, `quality.py` | No defect found that justified modification during the audit. |
| `model/skin_cancer_mobilenetv2.h5` | Original model weights — preserved untouched. |
| `results/convnext_tiny/*.pt` | All ConvNeXt checkpoints, including `stage2_epoch13.pt` and `convnext_tiny_best.pt` — preserved untouched. |
| `results/convnext_tiny/*.npy` | All saved prediction arrays — preserved untouched. |
| `results/convnext_tiny/metrics.json`, `model/model_card.json` | Contain the stale `app_py_current_threshold = 0.5` fields. Documented as outdated metadata rather than edited, so that the discrepancy stays visible and auditable. |
| `tests/test_regression.py`, `tests/test_quality.py` | The CWD-dependent relative-path fixture issue is documented (ISSUE-07); the test files were not edited during the audit. |
| `D:\User\Downloads\archive\*` | HAM10000 dataset and metadata — read only. |
| `D:\dermai_cache\*.npy` | Decoded image caches — read only, via read-only memmap. |
| `D:\User\Downloads\*.keras` | EfficientNetB2 / MobileNetV2 models outside the project — not touched, not copied in. |
| `Research Paper.pdf` and the other PDFs in `MINI3` | `Research Paper.pdf` is a third-party **reference** paper, not the DermaAI paper. It was **not** renamed, edited, or converted. |

---

## Net file inventory change produced by the audit cycle

**New files created (9)**

```
DERMAI_FINAL_VERIFICATION_REPORT.md
DERMAI_REVISION_LOG.md
DERMAI_UNRESOLVED_ISSUES.md
scripts/verify_lesion_split.py
scripts/verify_reported_metrics.py
scripts/split_verification_results.json
scripts/split_verification_results_fresh_run_20261020.json
scripts/metrics_verification_results.json
scripts/convnext_checkpoint_verification.json
```

**Backups preserved (9)**

```
DERMAI_VERIFICATION_RESULTS.json.bak_20261017
DERMAI_VERIFICATION_RESULTS.json.bak_20261020_pre_phase8
DERMAI_VERIFICATION_RESULTS.json.bak_20261020_pre_phase10
scripts/split_verification_results.json.bak_20261017
scripts/verify_reported_metrics.py.bak_20261020
config.py.bak_20261020
README.md.bak_20261020
PROJECT_AUDIT.md.bak_20261020
```

**Existing files modified (7)**

```
DERMAI_VERIFICATION_RESULTS.json          (merged, phases 1-9 preserved; phase 10 added)
scripts/verify_reported_metrics.py         (output-path + exit-code defects fixed)
PROJECT_COMPLETION_REPORT.md               (pytest figure corrected 18 -> 56)
README.md                                  (test path / Grad-CAM description corrected;
                                            checkpoint primary/fallback corrected in Phase F)
PROJECT_AUDIT.md                           (checkpoint primary/fallback corrected in Phase F)
config.py                                  (ConvNeXt checkpoint paths: epoch 7 primary,
                                            epoch 13 fallback - approved Option A)
DERMAI_FINAL_VERIFICATION_REPORT.md        (section 5.7 + consolidated status updated)
DERMAI_UNRESOLVED_ISSUES.md                (ISSUE-01 resolved, ISSUE-02 partially resolved)
```

**Model weights, checkpoints, prediction arrays, dataset files, and notebooks modified: 0.**

**Thresholds modified: 0** (`CONVNEXT_THRESHOLD = 0.525`, `MOBILENET_THRESHOLD = 0.22`,
`RISK_MEDIUM = 0.3`, `RISK_HIGH = 0.6` all unchanged.)

**Commits / pushes / deployments performed: 0.**
