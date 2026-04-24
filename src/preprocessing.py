"""
preprocessing.py
-----------------
Image preprocessing utilities for retinal fundus images.

* Resize to 224×224
* CLAHE contrast enhancement (LAB luminance)
* ImageNet normalisation for EfficientNet-B3
* Optional brightness normalisation
"""

import cv2
import numpy as np

IMG_SIZE = 224

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------- CLAHE preprocessing ---------- #

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE on the L-channel of the LAB colour space.

    Parameters
    ----------
    image : np.ndarray
        RGB image, uint8 or float32 [0,1].

    Returns
    -------
    np.ndarray  float32, [0, 1]
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


# ---------- Core preprocessing ---------- #

def load_and_preprocess(image_path: str, target_size: int = IMG_SIZE,
                        use_clahe: bool = True,
                        normalize_imagenet: bool = True) -> np.ndarray:
    """
    Read an image from disk, apply CLAHE, resize, and normalise.

    Parameters
    ----------
    image_path : str
    target_size : int
    use_clahe : bool
        Apply CLAHE preprocessing (should match training).
    normalize_imagenet : bool
        If True, apply ImageNet mean/std normalisation.
        If False, just scale to [0, 1].

    Returns
    -------
    np.ndarray  (target_size, target_size, 3) float32
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # CLAHE on uint8 image
    if use_clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_ch)
        lab_out = cv2.merge([l_enhanced, a_ch, b_ch])
        img = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)

    img = cv2.resize(img, (target_size, target_size))
    img = img.astype(np.float32) / 255.0

    if normalize_imagenet:
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

    return img


# ---------- Brightness normalisation ---------- #

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
