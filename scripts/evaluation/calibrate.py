"""
Temperature-scaling calibration script.
Run this in Colab (or locally) AFTER training to find:
  - TEMPERATURE (T) : paste into app.py
  - THRESHOLD       : paste into app.py

Usage:
  python calibrate.py --model model/skin_cancer_mobilenetv2.h5 \
                      --val_dir path/to/validation_images

Validation directory layout expected by ImageDataGenerator:
  val_dir/
    benign/   *.jpg
    malignant/ *.jpg
"""

import argparse
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

IMG_SIZE = (224, 224)
BATCH    = 32
TARGET_SENSITIVITY = 0.95   # catch at least 95 % of true positives


def load_val_data(val_dir):
    gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    ds  = gen.flow_from_directory(
        val_dir, target_size=IMG_SIZE, batch_size=BATCH,
        class_mode="binary", shuffle=False,
    )
    return ds


def get_raw_probs(model, ds):
    probs  = model.predict(ds, verbose=1).flatten()
    labels = ds.classes.astype("float32")
    return probs, labels


def temperature_nll(T, logits, labels):
    scaled = 1.0 / (1.0 + np.exp(-logits / T))
    scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
    return log_loss(labels, scaled)


def find_threshold(cal_probs, labels, target_sens=TARGET_SENSITIVITY):
    """
    Find the lowest threshold that still catches at least `target_sens` of
    the true positives. Anything at or above it is reported as affected.
    """
    pos_mask = labels == 1
    best_t   = 0.5

    for t in np.arange(0.05, 0.95, 0.01):
        if np.mean(cal_probs[pos_mask] >= t) >= target_sens:
            best_t = round(float(t), 2)
            break

    return best_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True)
    parser.add_argument("--val_dir", required=True)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    ds    = load_val_data(args.val_dir)

    print("Running inference on validation set …")
    raw_probs, labels = get_raw_probs(model, ds)

    # Convert probabilities to logits for temperature scaling
    raw_probs_c = np.clip(raw_probs, 1e-7, 1 - 1e-7)
    logits      = np.log(raw_probs_c / (1 - raw_probs_c))

    result = minimize_scalar(
        temperature_nll, bounds=(0.1, 10.0), method="bounded",
        args=(logits, labels),
    )
    T = round(float(result.x), 4)
    cal_probs = 1.0 / (1.0 + np.exp(-logits / T))

    threshold = find_threshold(cal_probs, labels)

    print("\n=== Calibration results ===")
    print(f"  TEMPERATURE = {T}")
    print(f"  THRESHOLD   = {threshold}")
    print("\nPaste these two values into app.py.")


if __name__ == "__main__":
    main()
