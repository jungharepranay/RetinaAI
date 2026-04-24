"""
quality_check.py
----------------
Image quality validation for retinal fundus images.

Rejects images that are too blurry, too dark/bright, have
insufficient contrast, or lack a detectable optic disc —
preventing unreliable downstream predictions.

An image failing any critical check is marked UNGRADABLE.
"""

import cv2
import numpy as np


# -------------------- Thresholds -------------------- #
BLUR_THRESHOLD = 100.0        # Variance of Laplacian below this → blurry
BRIGHTNESS_LOW = 40.0         # Mean pixel value below this → too dark
BRIGHTNESS_HIGH = 220.0       # Mean pixel value above this → overexposed
CONTRAST_THRESHOLD = 30.0     # Std-dev below this → low contrast
MIN_DISC_AREA_FRAC = 0.001   # Optic disc must occupy ≥ 0.1% of image area


def _detect_optic_disc(gray: np.ndarray) -> bool:
    """
    Attempt to detect the optic disc as the large bright region.

    Uses brightness thresholding at the 92nd percentile and checks
    whether a large-enough contour exists.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image, uint8.

    Returns
    -------
    bool
        True if an optic-disc-like bright region is detected.
    """
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh_val = np.percentile(blurred, 92)
    _, mask = cv2.threshold(blurred, int(thresh_val), 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    largest = max(contours, key=cv2.contourArea)
    image_area = gray.shape[0] * gray.shape[1]
    return cv2.contourArea(largest) >= (image_area * MIN_DISC_AREA_FRAC)


def check_image_quality(image: np.ndarray) -> dict:
    """
    Assess whether a retinal fundus image meets minimum quality
    standards for reliable analysis.

    An image is marked UNGRADABLE if:
      - Blur is too high (Laplacian variance < threshold)
      - Brightness is too low (mean < threshold)
      - Optic disc cannot be detected

    Parameters
    ----------
    image : np.ndarray
        RGB image as uint8 (0-255) or float32 (0-1).

    Returns
    -------
    dict
        ``is_valid``           : bool — True if image passes all checks.
        ``reason``             : str  — '' if valid, else failure description.
        ``blur_score``         : float — Laplacian variance (higher = sharper).
        ``brightness_mean``    : float — Mean pixel intensity.
        ``contrast_std``       : float — Pixel standard deviation.
        ``optic_disc_detected``: bool — Whether the optic disc was found.
    """
    # Ensure uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            img = (image * 255).astype(np.uint8)
        else:
            img = image.astype(np.uint8)
    else:
        img = image.copy()

    # Convert to grayscale for metric computation
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ----- Blur detection (Laplacian variance) ----- #
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # ----- Brightness ----- #
    brightness_mean = float(np.mean(gray))

    # ----- Contrast (standard deviation) ----- #
    contrast_std = float(np.std(gray))

    # ----- Optic Disc Detection ----- #
    optic_disc_detected = _detect_optic_disc(gray)

    # ----- Decide ----- #
    reasons = []

    if blur_score < BLUR_THRESHOLD:
        reasons.append("blur")
    if brightness_mean < BRIGHTNESS_LOW:
        reasons.append("low_light")
    if brightness_mean > BRIGHTNESS_HIGH:
        reasons.append("overexposed")
    if contrast_std < CONTRAST_THRESHOLD:
        reasons.append("low_contrast")
    if not optic_disc_detected:
        reasons.append("optic_disc_not_detected")

    # Mark as UNGRADABLE if blur OR low brightness OR no optic disc
    ungradable_conditions = {"blur", "low_light", "optic_disc_not_detected"}
    is_ungradable = bool(ungradable_conditions & set(reasons))

    is_valid = len(reasons) == 0

    # Build reason string
    if is_ungradable:
        reason_str = "UNGRADABLE"
    elif reasons:
        reason_str = ", ".join(reasons)
    else:
        reason_str = ""

    return {
        "is_valid": is_valid,
        "reason": reason_str,
        "blur_score": round(blur_score, 2),
        "brightness_mean": round(brightness_mean, 2),
        "contrast_std": round(contrast_std, 2),
        "optic_disc_detected": optic_disc_detected,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.quality_check <image_path>")
        sys.exit(1)
    img = cv2.imread(sys.argv[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = check_image_quality(img)
    print(result)
