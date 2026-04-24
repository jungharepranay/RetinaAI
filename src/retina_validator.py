"""
retina_validator.py
-------------------
Heuristic retinal fundus image validator.

Uses three complementary signals to determine whether an uploaded
image is a retinal fundus photograph:
  1. Color distribution (red channel dominance)
  2. Circular structure detection (fundus field-of-view)
  3. Aspect ratio & minimum size

██  NO LLM / NO ML MODEL  ██
This is a lightweight heuristic-only validator.

An image must pass at least 2 of 3 checks to be considered valid.
"""

import cv2
import numpy as np


# ─── Thresholds ─── #
MIN_RED_MEAN = 55                # Fundus images have warm-toned illumination
MIN_SIZE = 50                    # Minimum dimension in pixels
MAX_ASPECT_RATIO = 3.0           # Reject extremely elongated images
MIN_CIRCLE_AREA_FRAC = 0.10     # Circle must occupy ≥10% of image area
DARK_BORDER_THRESHOLD = 30       # Pixel value considered "dark border"
MIN_DARK_BORDER_FRAC = 0.05     # At least 5% dark border expected around fundus


def _check_color_distribution(img_rgb: np.ndarray) -> dict:
    """
    Check whether the image has color properties consistent with
    a retinal fundus photograph.

    Retinal images have:
      - High red channel intensity (from fundus camera illumination)
      - Red > Green in most of the non-black region
      - Warm color tones overall

    Returns dict with 'passed' (bool) and 'reason' (str).
    """
    # Mask out very dark pixels (background)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > 20  # non-background pixels

    if mask.sum() < 100:
        return {"passed": False, "reason": "Image is mostly dark/empty"}

    r = img_rgb[:, :, 0][mask].astype(float)
    g = img_rgb[:, :, 1][mask].astype(float)
    b = img_rgb[:, :, 2][mask].astype(float)

    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)

    # Fundus images typically have red-dominant illumination
    red_dominant = r_mean > g_mean
    red_sufficient = r_mean > MIN_RED_MEAN

    # Check that it's not a grayscale image disguised as RGB
    color_spread = max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)
    has_color = color_spread > 8

    passed = red_dominant and red_sufficient and has_color

    return {
        "passed": passed,
        "reason": "" if passed else "Color distribution not consistent with retinal image",
        "r_mean": round(r_mean, 1),
        "g_mean": round(g_mean, 1),
        "b_mean": round(b_mean, 1),
    }


def _check_circular_structure(img_rgb: np.ndarray) -> dict:
    """
    Detect the circular field-of-view characteristic of fundus images.

    Fundus photographs have a circular bright region against a dark
    background, produced by the fundus camera's optical system.

    Uses a combination of:
      - Dark border detection (fundus images have black corners)
      - Contour-based circle detection
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    total_pixels = h * w

    # Check for dark borders (fundus images have black corners)
    corner_size = min(h, w) // 8
    if corner_size < 5:
        return {"passed": False, "reason": "Image too small for structure analysis"}

    # Sample corner regions
    corners = [
        gray[:corner_size, :corner_size],          # top-left
        gray[:corner_size, -corner_size:],          # top-right
        gray[-corner_size:, :corner_size],          # bottom-left
        gray[-corner_size:, -corner_size:],         # bottom-right
    ]

    dark_corner_count = sum(
        1 for c in corners if np.mean(c) < DARK_BORDER_THRESHOLD
    )

    # Fundus images typically have dark corners (circular FOV)
    has_dark_corners = dark_corner_count >= 2

    # Try to find a large bright contour (the fundus circle)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, binary = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    has_large_circular = False
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)

        # Check if contour is roughly circular (circularity metric)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            area_frac = area / total_pixels

            # Circularity > 0.5 and area > 10% → likely a fundus circle
            has_large_circular = (circularity > 0.4 and
                                  area_frac > MIN_CIRCLE_AREA_FRAC)

    passed = has_dark_corners or has_large_circular

    return {
        "passed": passed,
        "reason": "" if passed else "No circular fundus structure detected",
        "dark_corners": dark_corner_count,
        "has_circular_region": has_large_circular,
    }


def _check_aspect_ratio(img_rgb: np.ndarray) -> dict:
    """
    Check image dimensions are reasonable for a retinal photograph.

    Fundus images are typically:
      - Near-square (1:1) or slightly rectangular (4:3)
      - At least 50×50 pixels
      - Not extremely elongated (rules out screenshots, banners)
    """
    h, w = img_rgb.shape[:2]

    if h < MIN_SIZE or w < MIN_SIZE:
        return {
            "passed": False,
            "reason": f"Image too small ({w}×{h}px, minimum {MIN_SIZE}×{MIN_SIZE})"
        }

    ratio = max(h, w) / max(min(h, w), 1)

    if ratio > MAX_ASPECT_RATIO:
        return {
            "passed": False,
            "reason": f"Extreme aspect ratio ({ratio:.1f}:1) — not consistent with retinal image"
        }

    return {
        "passed": True,
        "reason": "",
        "dimensions": f"{w}×{h}",
        "aspect_ratio": round(ratio, 2),
    }


def validate_retinal_image(image_path: str = None,
                           image_array: np.ndarray = None) -> dict:
    """
    Validate whether an image is a retinal fundus photograph.

    Must pass at least 2 of 3 heuristic checks:
      1. Color distribution (red channel dominance)
      2. Circular structure (fundus field-of-view)
      3. Aspect ratio & size

    Parameters
    ----------
    image_path : str or None
        Path to image file. Used if image_array is None.
    image_array : np.ndarray or None
        RGB image array. If provided, image_path is ignored.

    Returns
    -------
    dict
        is_retinal : bool — True if image passes validation.
        reason : str — '' if valid, else failure description.
        checks : dict — Individual check results.
    """
    # Load image
    if image_array is not None:
        img_rgb = image_array.copy()
        # Ensure uint8
        if img_rgb.dtype != np.uint8:
            if img_rgb.max() <= 1.0:
                img_rgb = (img_rgb * 255).astype(np.uint8)
            else:
                img_rgb = img_rgb.astype(np.uint8)
    elif image_path is not None:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {
                "is_retinal": False,
                "reason": f"Cannot read image: {image_path}",
                "checks": {},
            }
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        return {
            "is_retinal": False,
            "reason": "No image provided",
            "checks": {},
        }

    # Run checks
    color_result = _check_color_distribution(img_rgb)
    structure_result = _check_circular_structure(img_rgb)
    aspect_result = _check_aspect_ratio(img_rgb)

    checks = {
        "color_distribution": color_result,
        "circular_structure": structure_result,
        "aspect_ratio": aspect_result,
    }

    # Must pass at least 2 of 3
    passed_count = sum(1 for c in checks.values() if c["passed"])
    is_retinal = passed_count >= 2

    if not is_retinal:
        failed = [name for name, c in checks.items() if not c["passed"]]
        reasons = [checks[f]["reason"] for f in failed if checks[f]["reason"]]
        reason = (
            "Please upload a valid retinal fundus image. "
            "The uploaded image does not appear to be a retinal photograph. "
            f"({'; '.join(reasons)})"
        )
    else:
        reason = ""

    return {
        "is_retinal": is_retinal,
        "reason": reason,
        "checks": checks,
        "passed_count": passed_count,
    }


# ─── CLI test ─── #
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m src.retina_validator <image_path>")
        sys.exit(1)

    result = validate_retinal_image(image_path=sys.argv[1])
    print(json.dumps(result, indent=2))
    print(f"\n→ {'VALID retinal image ✓' if result['is_retinal'] else 'NOT a retinal image ✗'}")
