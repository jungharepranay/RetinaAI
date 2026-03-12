"""
preprocessing.py
-----------------
Image preprocessing utilities for retinal fundus images.

* Resize to 224×224
* Normalise pixel values to [0, 1]
* Optional CLAHE contrast enhancement
* Optional brightness normalisation
"""

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 224


# ---------- OpenCV-based helpers (for single-image use) ---------- #

def load_and_preprocess(image_path: str, target_size: int = IMG_SIZE) -> np.ndarray:
    """
    Read an image from disk, resize, and normalise to [0, 1].

    Parameters
    ----------
    image_path : str
        Path to the image file.
    target_size : int
        Target height and width.

    Returns
    -------
    np.ndarray  (target_size, target_size, 3) float32 in [0, 1]
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size))
    img = img.astype(np.float32) / 255.0
    return img


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation)
    on the L-channel of the LAB colour space.
    """
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a, b))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return result.astype(np.float32) / 255.0


def normalise_brightness(image: np.ndarray) -> np.ndarray:
    """Adjust brightness via histogram equalisation on the V channel."""
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv = cv2.merge((h, s, v))
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return result.astype(np.float32) / 255.0


# --------- TensorFlow-native helpers (for tf.data pipeline) --------- #

def tf_decode_and_resize(image_path: tf.Tensor, target_size: int = IMG_SIZE):
    """TF-native: read, decode, resize and normalise an image."""
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [target_size, target_size])
    img = tf.cast(img, tf.float32) / 255.0
    return img
