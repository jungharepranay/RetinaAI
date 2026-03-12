"""
predict.py
----------
Run single-image prediction using the trained retinal disease model.

Usage::

    python -m src.predict path/to/image.jpg
"""

import os
import sys
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_and_preprocess
from src.data_loader import DISEASE_COLUMNS, DISEASE_NAMES

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "retinal_model.keras")
THRESHOLD = 0.5


def predict_image(image_path: str, model=None, threshold: float = THRESHOLD):
    """
    Predict diseases for a single retinal fundus image.

    Parameters
    ----------
    image_path : str
        Path to the image.
    model : tf.keras.Model or None
        Pre-loaded model. If None, loads from ``MODEL_PATH``.
    threshold : float
        Classification threshold.

    Returns
    -------
    detected : list[str]
        Human-readable disease names exceeding the threshold.
    probabilities : dict[str, float]
        Per-class probabilities.
    """
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)

    img = load_and_preprocess(image_path)
    img_batch = np.expand_dims(img, axis=0)

    preds = model.predict(img_batch, verbose=0)[0]

    probabilities = {
        DISEASE_NAMES[col]: float(round(preds[i], 4))
        for i, col in enumerate(DISEASE_COLUMNS)
    }

    detected = [
        DISEASE_NAMES[col]
        for i, col in enumerate(DISEASE_COLUMNS)
        if preds[i] >= threshold
    ]

    return detected, probabilities


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"Error: file not found — {image_path}")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH} …")
    model = tf.keras.models.load_model(MODEL_PATH)

    detected, probs = predict_image(image_path, model)

    print("\n===== Prediction Results =====")
    print(f"Image: {image_path}\n")

    print("Probabilities:")
    for disease, prob in probs.items():
        marker = " ◀" if prob >= THRESHOLD else ""
        print(f"  {disease:40s}  {prob:.4f}{marker}")

    print("\nDetected Diseases:")
    if detected:
        for d in detected:
            print(f"  • {d}")
    else:
        print("  (none above threshold)")


if __name__ == "__main__":
    main()
