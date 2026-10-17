#!/usr/bin/env python3
"""
Generate comprehensive Grad-CAM visualizations with raw heatmaps, overlays,
and numerical statistics for both models.
"""

import os
import io
import base64
import numpy as np
from PIL import Image
import cv2
import json

# Test images
TEST_IMAGES = [
    ("test-images/cancer-melanoma.jpg", "melanoma"),
    ("test-images/noncancer-nevus.jpg", "nevus"),
    ("test-images/noncancer-congenital-nevus.jpg", "congenital_nevus"),
    ("test-images/noncancer-seborrheic-keratosis.jpg", "seborrheic_keratosis"),
]

OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_image(img_array, path):
    """Save image array to file."""
    if img_array.dtype != np.uint8:
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    Image.fromarray(img_array).save(path)


def decode_heatmap(heatmap_b64):
    """Decode base64 heatmap to numpy array (RGB)."""
    raw = base64.b64decode(heatmap_b64)
    arr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def extract_raw_cam_from_overlay(overlay, original):
    """Extract approximate raw CAM from overlay.
    
    Overlay = 0.55 * original + 0.45 * JET(cam)
    We can't perfectly reverse JET, but we can compute a heat proxy.
    """
    # Compute heat proxy: R - B channel difference (JET is blue->red)
    b, g, r = overlay[:, :, 0].astype(np.float64), overlay[:, :, 1].astype(np.float64), overlay[:, :, 2].astype(np.float64)
    heat = np.maximum(r - b, 0.0)
    return heat


def compute_statistics(overlay, original, label, model_name, target_class):
    """Compute numerical statistics for the heatmap."""
    # Extract heat proxy
    heat = extract_raw_cam_from_overlay(overlay, original)
    
    stats = {
        "model": model_name,
        "label": label,
        "target_class": target_class,
        "overlay_shape": list(overlay.shape),
        "heat_min": float(heat.min()),
        "heat_max": float(heat.max()),
        "heat_mean": float(heat.mean()),
        "heat_std": float(heat.std()),
    }
    
    # Centroid and corner concentration
    flat = heat.flatten()
    if flat.max() > 0:
        h, w = heat.shape
        thr = np.percentile(flat, 90)
        hot = heat >= thr
        ys, xs = np.nonzero(hot)
        if len(xs) > 0:
            cy, cx = ys.mean() / h, xs.mean() / w
            center_dist = float(np.hypot(cx - 0.5, cy - 0.5))
            
            margin = 0.125
            m_h, m_w = int(h * margin), int(w * margin)
            frame = np.zeros((h, w), dtype=bool)
            frame[:m_h, :] = True
            frame[h - m_h:, :] = True
            frame[:, :m_w] = True
            frame[:, w - m_w:] = True
            corner_share = float((hot & frame).sum()) / float(max(hot.sum(), 1))
            
            stats.update({
                "centroid_x": cx,
                "centroid_y": cy,
                "center_distance": center_dist,
                "corner_frame_share": corner_share,
                "hot_pixel_count": int(hot.sum()),
                "total_pixels": h * w,
            })
        else:
            stats.update({
                "centroid_x": 0.5,
                "centroid_y": 0.5,
                "center_distance": 0.0,
                "corner_frame_share": 0.0,
                "hot_pixel_count": 0,
                "total_pixels": h * w,
            })
    else:
        stats.update({
            "centroid_x": 0.5,
            "centroid_y": 0.5,
            "center_distance": 0.0,
            "corner_frame_share": 0.0,
            "hot_pixel_count": 0,
            "total_pixels": heat.size,
        })
    
    return stats, heat


def test_convnext_visualizations():
    """Generate ConvNeXt-Tiny visualizations with statistics."""
    print("\n=== Generating ConvNeXt-Tiny Visualizations ===")
    import inference
    
    model = inference.load_convnext_model()
    device = inference.get_device()
    
    all_stats = []
    
    for path, label in TEST_IMAGES:
        print(f"\nProcessing {label}...")
        img = Image.open(path).convert("RGB")
        
        # Get prediction
        result = inference.predict_single(img)
        print(f"  Verdict: {result['verdict']}, Prob: {result['cancer_probability']:.4f}")
        
        # Save original (resized to 224x224)
        orig = np.array(img.resize((224, 224)))
        orig_path = os.path.join(OUTPUT_DIR, f"convnext_{label}_original.png")
        save_image(orig, orig_path)
        
        if result['heatmap']:
            # Decode heatmap overlay
            overlay = decode_heatmap(result['heatmap'])
            overlay_path = os.path.join(OUTPUT_DIR, f"convnext_{label}_overlay.png")
            save_image(overlay, overlay_path)
            
            # Extract and save raw heat proxy
            target = result['verdict']  # affected or not_affected
            stats, heat = compute_statistics(overlay, orig, label, "convnext_tiny", target)
            stats["predicted_class"] = result['verdict']
            stats["cancer_probability"] = result['cancer_probability']
            stats["raw_logit"] = None  # Would need to extract from model
            
            # Save raw heat proxy as grayscale
            heat_norm = np.uint8(255 * heat / (heat.max() + 1e-7)) if heat.max() > 0 else np.zeros_like(heat, dtype=np.uint8)
            heat_path = os.path.join(OUTPUT_DIR, f"convnext_{label}_raw_heat.png")
            save_image(heat_norm, heat_path)
            
            # Also save JET colormap of raw heat
            heat_jet = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
            heat_jet = cv2.cvtColor(heat_jet, cv2.COLOR_BGR2RGB)
            heat_jet_path = os.path.join(OUTPUT_DIR, f"convnext_{label}_raw_heat_jet.png")
            save_image(heat_jet, heat_jet_path)
            
            all_stats.append(stats)
            print(f"  Stats: centroid=({stats['centroid_x']:.2f},{stats['centroid_y']:.2f}) "
                  f"center_dist={stats['center_distance']:.2f} corner_share={stats['corner_frame_share']:.2f}")
        else:
            print("  No heatmap generated")
    
    return all_stats


def test_mobilenet_visualizations():
    """Generate MobileNetV2 visualizations with statistics."""
    print("\n=== Generating MobileNetV2 Visualizations ===")
    import tensorflow as tf
    from app import mobilenet_grad_cam, mobilenet_preprocess, load_mobilenet_model
    
    model = load_mobilenet_model()
    
    all_stats = []
    
    for path, label in TEST_IMAGES:
        print(f"\nProcessing {label}...")
        img = Image.open(path).convert("RGB")
        
        # Preprocess
        arr = mobilenet_preprocess(img)
        
        # Get prediction
        raw_prob = float(model.predict(arr, verbose=0)[0][0])
        print(f"  Raw prob: {raw_prob:.4f}")
        
        # Save original (resized to 224x224)
        orig = np.array(img.resize((224, 224)))
        orig_path = os.path.join(OUTPUT_DIR, f"mobilenet_{label}_original.png")
        save_image(orig, orig_path)
        
        # Generate heatmaps for both target classes
        for target in ["affected", "not_affected"]:
            heatmap_b64 = mobilenet_grad_cam(arr, target_class=target)
            if heatmap_b64:
                overlay = decode_heatmap(heatmap_b64)
                overlay_path = os.path.join(OUTPUT_DIR, f"mobilenet_{label}_{target}_overlay.png")
                save_image(overlay, overlay_path)
                
                # Compute statistics
                stats, heat = compute_statistics(overlay, orig, label, "mobilenetv2", target)
                stats["predicted_class"] = "affected" if raw_prob >= 0.22 else "not_affected"  # using threshold
                stats["raw_probability"] = raw_prob
                
                # Save raw heat proxy
                heat_norm = np.uint8(255 * heat / (heat.max() + 1e-7)) if heat.max() > 0 else np.zeros_like(heat, dtype=np.uint8)
                heat_path = os.path.join(OUTPUT_DIR, f"mobilenet_{label}_{target}_raw_heat.png")
                save_image(heat_norm, heat_path)
                
                heat_jet = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
                heat_jet = cv2.cvtColor(heat_jet, cv2.COLOR_BGR2RGB)
                heat_jet_path = os.path.join(OUTPUT_DIR, f"mobilenet_{label}_{target}_raw_heat_jet.png")
                save_image(heat_jet, heat_jet_path)
                
                all_stats.append(stats)
                print(f"  {target}: centroid=({stats['centroid_x']:.2f},{stats['centroid_y']:.2f}) "
                      f"center_dist={stats['center_distance']:.2f} corner_share={stats['corner_frame_share']:.2f}")
            else:
                print(f"  {target}: No heatmap generated")
    
    return all_stats


def main():
    print("Generating comprehensive Grad-CAM visualizations...")
    print(f"Output directory: {OUTPUT_DIR}/")
    
    convnext_stats = test_convnext_visualizations()
    mobilenet_stats = test_mobilenet_visualizations()
    
    # Save all statistics to JSON
    all_stats = convnext_stats + mobilenet_stats
    stats_path = os.path.join(OUTPUT_DIR, "visualization_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\nAll visualizations saved to {OUTPUT_DIR}/")
    print(f"Statistics saved to {stats_path}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    for s in all_stats:
        print(f"{s['model']:14s} {s['label']:20s} target={s['target_class']:12s} "
              f"centroid=({s['centroid_x']:.2f},{s['centroid_y']:.2f}) "
              f"center_dist={s['center_distance']:.2f} corner_share={s['corner_frame_share']:.2f}")


if __name__ == "__main__":
    main()