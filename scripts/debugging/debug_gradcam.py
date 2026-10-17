#!/usr/bin/env python3
"""
Debug script to visualize Grad-CAM heatmaps for both models.
Saves original, raw heatmap, and overlay for each test image.
"""

import os
import io
import base64
import numpy as np
from PIL import Image
import cv2

# Test images
TEST_IMAGES = [
    ("test-images/cancer-melanoma.jpg", "melanoma"),
    ("test-images/noncancer-nevus.jpg", "nevus"),
    ("test-images/noncancer-congenital-nevus.jpg", "congenital_nevus"),
    ("test-images/noncancer-seborrheic-keratosis.jpg", "seborrheic_keratosis"),
]

OUTPUT_DIR = "debug_gradcam"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_image(img_array, path):
    """Save image array to file."""
    if img_array.dtype != np.uint8:
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    Image.fromarray(img_array).save(path)


def decode_heatmap(heatmap_b64):
    """Decode base64 heatmap to numpy array."""
    raw = base64.b64decode(heatmap_b64)
    arr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def test_convnext_gradcam():
    """Test ConvNeXt-Tiny Grad-CAM."""
    print("\n=== Testing ConvNeXt-Tiny Grad-CAM ===")
    import inference
    
    model = inference.load_convnext_model()
    device = inference.get_device()
    
    for path, label in TEST_IMAGES:
        print(f"\nProcessing {label}...")
        img = Image.open(path).convert("RGB")
        
        # Get prediction
        result = inference.predict_single(img)
        print(f"  Verdict: {result['verdict']}, Prob: {result['cancer_probability']:.4f}")
        
        if result['heatmap']:
            # Decode heatmap
            heatmap = decode_heatmap(result['heatmap'])
            
            # Save original (resized to 224x224)
            orig = np.array(img.resize((224, 224)))
            save_image(orig, os.path.join(OUTPUT_DIR, f"convnext_{label}_original.png"))
            
            # Save heatmap overlay
            save_image(heatmap, os.path.join(OUTPUT_DIR, f"convnext_{label}_overlay.png"))
            
            # Also extract and save raw CAM (before color map)
            # The heatmap is 0.55*orig + 0.45*JET(cam)
            # We can't easily reverse this, but we can save what we have
            
            print(f"  Heatmap shape: {heatmap.shape}, range: [{heatmap.min()}, {heatmap.max()}]")
        else:
            print("  No heatmap generated")


def test_mobilenet_gradcam():
    """Test MobileNetV2 Grad-CAM."""
    print("\n=== Testing MobileNetV2 Grad-CAM ===")
    import tensorflow as tf
    from app import mobilenet_grad_cam, mobilenet_preprocess, load_mobilenet_model
    
    model = load_mobilenet_model()
    
    for path, label in TEST_IMAGES:
        print(f"\nProcessing {label}...")
        img = Image.open(path).convert("RGB")
        
        # Preprocess
        arr = mobilenet_preprocess(img)
        
        # Get prediction
        raw_prob = float(model.predict(arr, verbose=0)[0][0])
        print(f"  Raw prob: {raw_prob:.4f}")
        
        # Generate heatmap for both classes
        for target in ["affected", "not_affected"]:
            heatmap_b64 = mobilenet_grad_cam(arr, target_class=target)
            if heatmap_b64:
                heatmap = decode_heatmap(heatmap_b64)
                orig = np.array(img.resize((224, 224)))
                save_image(orig, os.path.join(OUTPUT_DIR, f"mobilenet_{label}_original.png"))
                save_image(heatmap, os.path.join(OUTPUT_DIR, f"mobilenet_{label}_{target}_overlay.png"))
                print(f"  {target}: Heatmap shape: {heatmap.shape}, range: [{heatmap.min()}, {heatmap.max()}]")
            else:
                print(f"  {target}: No heatmap generated")


if __name__ == "__main__":
    test_convnext_gradcam()
    test_mobilenet_gradcam()
    print(f"\nDebug images saved to {OUTPUT_DIR}/")