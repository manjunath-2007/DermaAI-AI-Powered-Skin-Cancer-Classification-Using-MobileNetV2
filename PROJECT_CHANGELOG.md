# PROJECT_CHANGELOG.md

## DermaAI - Skin Cancer Detection Project Changelog

---

## [2026-10-15] Image-Quality Warnings (Blur / Dark / Overexposed)

### Summary
Lightweight, advisory image-quality screening added to the backend and
surfaced in the active frontend. Warnings never block a prediction: an
image that is merely blurry, too dark, or overexposed still gets scanned
and a prediction is always returned. Only un-decodable files are
rejected, via the pre-existing error handling.

### Files Added
- `quality.py` - `check_image_quality(img)` runs on the **already-decoded**
  PIL Image (no second decode). Downscales to `QUALITY_ANALYSIS_SIZE`
  (256x256) for a single resize + grayscale conversion, then computes:
  - `blur_score` = variance of the Laplacian (lower = blurrier)
  - `mean_brightness` = mean grayscale intensity normalised to [0, 1]
  - `overexposure_ratio` = fraction of pixels above 240/255
  Returns `{"warnings": [...], "is_valid": bool, "metrics": {...}}` with
  user-facing messages for `blur`, `dark`, and `bright` issues.

### Files Modified
- `config.py` - new "Image quality checks" section with configurable
  thresholds: `QUALITY_ENABLED`, `QUALITY_ANALYSIS_SIZE`,
  `QUALITY_BLUR_THRESHOLD` (100.0), `QUALITY_BRIGHTNESS_DARK` (0.20),
  `QUALITY_BRIGHTNESS_BRIGHT` (0.95),
  `QUALITY_OVEREXPOSURE_RATIO` (0.10).
- `app.py` - `/predict` and `/predict_batch` now call the quality check
  after decoding and add `quality_warnings`, `quality_metrics`, and
  `quality_is_valid` to successful responses. All existing response
  fields, status codes, and error handling are unchanged.
- `templates/index.html` (active frontend) - warning panel below the
  image preview and above the result, `.quality-warnings` styling, and
  `showQuality()` rendering from `d.quality_warnings`. Panels clear on
  new upload / reset / re-analyse. The Future Work card is now marked
  "Implemented".
- `static/js/app.js`, `static/css/style.css` - the alternative frontend
  (not linked by `index.html`) received the same compatible
  `showQualityWarnings()` handling and `.quality-warnings` styles so the
  two stay in sync.
- `tests/test_quality.py` (new) - unit tests for blurry, dark,
  overexposed, acceptable, non-RGB-mode, disabled-config, and
  threshold-configuration cases, plus API compatibility tests for
  `/predict` and `/predict_batch`.

### Test Results

```
tests/test_quality.py (pytest): 15 passed, 0 failed
tests/test_regression.py:      22 passed, 0 failed, 0 errored (1 non-fatal WARN)
```

The single regression WARN is the pre-existing MobileNetV2 melanoma
corner-activation diagnostic, unrelated to this change.

### Known Limitations
- Thresholds (especially `QUALITY_BLUR_THRESHOLD`) were tuned on a small
  set of synthetic and sample images; real-world skin photos may need
  retuning. Metrics are rounded to 2-4 decimals in the response.
- The overexposure check is a blunt near-white pixel fraction; skin with
  large specular highlights can trigger it.
- Warnings are advisory only - no retake is enforced and no prediction
  is blocked.
- `blur_score` is not numerically identical across input resolutions
  (the Laplacian variance of fine texture changes after downscaling);
  classification is stable, the number is not.
- The quality check runs on CPU in the request path (single 256x256
  resize + one Laplacian, typically <10 ms), but is not benchmarked in
  the regression suite.

---

## [2026-10-15] Grad-CAM Investigation & Fixes

### Investigation Summary
Comprehensive investigation of Grad-CAM localization for both ConvNeXt-Tiny and MobileNetV2 models using 4 external clinical test images.

### ConvNeXt-Tiny (PyTorch) - Primary Model

**Changes Made:**
1. **Fixed target layer** in `inference.py`:
   - Before: `features.7.2.block.0` (depthwise 7×7 conv inside last CNBlock)
   - After: `model.features` (final spatial feature map before global average pooling)
   - Shape: `[1, 768, 7, 7]` → correct Grad-CAM target per paper methodology

2. **Removed GaussianBlur** from Grad-CAM pipeline:
   - Before: `cv2.GaussianBlur(cam_np, (3, 3), 0)` shifted hotspot by ~0.2 in normalized coords
   - After: Unblurred CAM for faithful visualization

3. **Added detailed logging** for debugging:
   - Target layer, activation/gradient shapes
   - Activation/gradient absolute means
   - CAM range, mean, hotspot location

**Grad-CAM Results (ConvNeXt-Tiny):**

| Image | Verdict | Prob | Target Class | Hotspot (224×224) | CAM Mean | Occlusion Max Drop |
|-------|---------|------|--------------|-------------------|----------|-------------------|
| melanoma | affected | 0.735 | affected | (80, 80) | 0.157 | -6.61 |
| nevus | not_affected | 0.038 | not_affected | (79, 112) | 0.097 | -1.54 |
| congenital_nevus | not_affected | 0.022 | not_affected | (112, 111) | 0.086 | -1.73 |
| seborrheic_keratosis | affected | 0.947 | affected | (144, 111) | 0.097 | -8.37 |

**Key Observations:**
- All hotspots cluster near image center (center_dist ≤ 0.05 in regression test)
- No corner-activation warning for any ConvNeXt-Tiny image
- Occlusion sensitivity confirms center regions drive cancer predictions (large negative drops when occluded)
- Without ground-truth lesion masks, cannot confirm hotspots align with actual lesions
- Gradients are weak (grad_abs_mean ~0.006–0.010) suggesting diffuse attention

### MobileNetV2 (TensorFlow/Keras) - Legacy Model

**Current State (unchanged):**
- Still uses last Conv2D layer as Grad-CAM target
- Still applies GaussianBlur (3×3) before resize
- Still uses legacy implementation in `app.py`

**Grad-CAM Results (MobileNetV2):**

| Image | Verdict | Prob | Target Class | Hotspot (224×224) | CAM Mean |
|-------|---------|------|--------------|-------------------|----------|
| melanoma | not_affected | 0.173 | not_affected | (208, 208) | 0.230 |
| nevus | affected | 0.511 | affected | (112, 111) | 0.456 |
| congenital_nevus | not_affected | 0.098 | not_affected | (0, 0) | 0.000 |
| seborrheic_keratosis | affected | 0.327 | affected | (80, 175) | 0.245 |

**Key Observations:**
- **Melanoma**: Hotspot at bottom-right corner (208, 208) - corner-frame share 0.68 → regression test WARN
- **Congenital nevus**: Empty CAM (all zeros) - gradient mean near zero
- **Seborrheic keratosis**: Hotspot at left-bottom (80, 175)
- GaussianBlur shifts hotspots vs. unblurred version
- Regression test correctly flags melanoma corner activation as WARN

### Frontend Display Issue (Documented, Not Fixed)

The frontend (`templates/index.html`) displays the **blended overlay** (0.55×original + 0.45×JET heatmap) with label "Where the scan looked" alongside the resized original. This makes the raw heatmap difficult to interpret. The backend returns only the blended overlay, not the raw heatmap.

**Per user instruction:** Frontend design kept unchanged. This is a known limitation.

---

## Files Modified

- `skin-cancer-app/inference.py` - ConvNeXt-Tiny Grad-CAM target layer, GaussianBlur removal, logging
- `skin-cancer-app/gradcam_investigation/` - Generated comparison images and statistics (new)
- `skin-cancer-app/gradcam_outputs_new/` - Previous corrected Grad-CAM outputs (new)
- `skin-cancer-app/gradcam_generation.log` - Generation log (new)
- `skin-cancer-app/gradcam_investigation.log` - Investigation log (new)

---

## Test Results

```
convnext_tiny  11 passed,  0 failed/errored
mobilenetv2    11 passed,  0 failed/errored

TOTAL: 22 passed, 0 failed, 0 errored, 1 warnings
REGRESSION SUITE: ALL TESTS PASSED
```

### Warnings (Non-fatal)
- `[WARN] mobilenetv2 :: heatmap_localization(cancer-melanoma.jpg)` - activation concentrated near image corners/borders

---

## Remaining Limitations

1. **No ground-truth lesion masks** - Cannot quantitatively verify heatmap-lesion alignment
2. **ConvNeXt-Tiny center bias** - Hotspots cluster near center; centrality ≠ lesion localization
3. **MobileNetV2 corner activation** - Melanoma heatmap focuses on bottom-right corner; model may respond to artefacts
4. **MobileNetV2 empty CAM** - Congenital nevus produces zero-gradient CAM
5. **GaussianBlur on MobileNetV2** - Still shifts hotspots; not removed per scope
6. **Frontend shows blended overlay** - Raw heatmap not directly visible to user
7. **Test images are external clinical photos** - Not validated ground truth for localization

---

## Next Steps (Not in Scope)

- [ ] Add ground-truth lesion segmentation for localization validation
- [ ] Fix MobileNetV2 Grad-CAM (target layer, remove blur, handle empty gradients)
- [ ] Return both raw heatmap and overlay from backend
- [ ] Frontend: show raw heatmap alongside or instead of blended overlay
- [ ] Occlusion sensitivity as quantitative localization metric
- [ ] Multi-class Grad-CAM for lesion-type-specific attention