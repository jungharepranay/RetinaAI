"""
clinical_features.py
--------------------
Extract clinically meaningful features from retinal fundus images
using classical computer vision (OpenCV + NumPy).

Covers ALL 8 ODIR disease classes with literature-backed methods.

Disease Coverage
----------------
  - Diabetic Retinopathy (DR): exudates, hemorrhages, microaneurysms
    → ETDRS Study, AAO PPP for DR
  - Glaucoma: cup-to-disc ratio (CDR)
    → Jonas et al. (1999) ISNT rule, AAO PPP for Glaucoma
  - Myopia: brightness, texture variance, vessel visibility, edge density
    → META-PM / Ohno-Matsui (2014)
  - Cataract: blur score, contrast ratio, haze index
    → Chylack et al. (1993) LOCS III, WHO Blindness Survey
  - AMD: drusen-like deposits, macular irregularity
    → AREDS Report No. 8, Ferris et al. (2013) Beckman classification
  - Hypertension: arteriole-venule ratio proxy, vessel tortuosity
    → Wong & Mitchell (2004) Lancet, Keith-Wagener-Barker classification
  - Normal: overall image clarity, absence-of-anomalies flag
  - Other: anomaly score (texture deviation from expected patterns)
"""

import cv2
import numpy as np


# ================================================================== #
#                       DIABETIC RETINOPATHY                          #
# ================================================================== #

def _detect_exudates(image_uint8: np.ndarray) -> tuple:
    """
    Detect bright yellow exudates via HSV colour filtering + contour
    detection.

    Literature: AAO PPP for DR — hard exudates indicate clinically
    significant macular edema (CSME).

    Returns
    -------
    count : int
        Number of exudate-like contours.
    area  : float
        Total area of exudate regions as fraction of image area.
    """
    hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV)

    # Exudates appear as bright yellow-white patches
    # HSV range: Hue ~15-40, Sat ~40-255, Val ~180-255
    lower = np.array([15, 40, 180], dtype=np.uint8)
    upper = np.array([40, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Filter tiny noise contours (< 30 px area)
    contours = [c for c in contours if cv2.contourArea(c) > 30]
    total_area = sum(cv2.contourArea(c) for c in contours)
    image_area = image_uint8.shape[0] * image_uint8.shape[1]

    return len(contours), total_area / image_area if image_area > 0 else 0.0


def _detect_hemorrhages(image_uint8: np.ndarray) -> tuple:
    """
    Detect dark red hemorrhages via HSV colour filtering + contour
    detection.

    Literature: ETDRS — hemorrhages and microaneurysms are primary
    markers of DR progression.

    Returns
    -------
    count : int
        Number of hemorrhage-like contours.
    area  : float
        Total area as fraction of image area.
    """
    hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV)

    # Hemorrhages appear as dark red spots
    # Low-hue reds (wraps around 0)
    lower1 = np.array([0, 50, 30], dtype=np.uint8)
    upper1 = np.array([10, 255, 150], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)

    # High-hue reds
    lower2 = np.array([160, 50, 30], dtype=np.uint8)
    upper2 = np.array([180, 255, 150], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 30]
    total_area = sum(cv2.contourArea(c) for c in contours)
    image_area = image_uint8.shape[0] * image_uint8.shape[1]

    return len(contours), total_area / image_area if image_area > 0 else 0.0


def _detect_microaneurysms(image_uint8: np.ndarray) -> int:
    """
    Detect microaneurysms — the earliest clinical sign of Diabetic
    Retinopathy (ETDRS).

    Microaneurysms appear as small dark circular spots (3-15 px radius)
    in the retinal fundus. Uses SimpleBlobDetector with parameters tuned
    for small, dark, circular blobs on the green channel (most vessel
    contrast).

    Literature: ETDRS Report No. 10 — microaneurysms as the defining
    feature of mild NPDR.

    Returns
    -------
    count : int
        Number of microaneurysm-like blobs detected.
    """
    # Green channel provides best vessel/lesion contrast
    green = image_uint8[:, :, 1]

    # CLAHE to enhance local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    # Invert so dark spots become bright (blob detector finds bright blobs)
    inverted = cv2.bitwise_not(enhanced)

    # Configure blob detector for small dark circular spots
    params = cv2.SimpleBlobDetector_Params()

    # Filter by area (radius 3-15px → area ~28 - ~706 px²)
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 750

    # Filter by circularity (microaneurysms are roughly circular)
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # Filter by convexity
    params.filterByConvexity = True
    params.minConvexity = 0.6

    # Filter by inertia (roundness)
    params.filterByInertia = True
    params.minInertiaRatio = 0.4

    # Filter by colour (we inverted, so looking for bright = originally dark)
    params.filterByColor = True
    params.blobColor = 255

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(inverted)

    return len(keypoints)


def extract_dr_features(image_uint8: np.ndarray) -> dict:
    """Extract Diabetic Retinopathy features."""
    exudate_count, exudate_area = _detect_exudates(image_uint8)
    hemorrhage_count, hemorrhage_area = _detect_hemorrhages(image_uint8)
    microaneurysm_count = _detect_microaneurysms(image_uint8)

    return {
        "exudate_count": exudate_count,
        "hemorrhage_count": hemorrhage_count,
        "microaneurysm_count": microaneurysm_count,
        "lesion_area": round(exudate_area + hemorrhage_area, 6),
    }


# ================================================================== #
#                            GLAUCOMA                                  #
# ================================================================== #

def _estimate_cdr(image_uint8: np.ndarray) -> float:
    """
    Estimate the Cup-to-Disc Ratio (CDR) from a retinal fundus image.

    Literature: AAO PPP for Glaucoma — CDR > 0.6 is clinically
    suspicious; Jonas et al. (1999) ISNT rule.

    Approach:
      1. Convert to grayscale and threshold for the bright optic disc.
      2. Fit minimum enclosing circle → disc radius.
      3. Threshold a brighter inner region → cup radius.
      4. CDR = cup_radius / disc_radius.

    Returns 0.0 if the optic disc cannot be reliably detected.
    """
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # ----- Detect optic disc (brightest region) ----- #
    # Threshold at 90th percentile brightness
    disc_thresh_val = np.percentile(blurred, 90)
    _, disc_mask = cv2.threshold(blurred, int(disc_thresh_val), 255,
                                 cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    disc_mask = cv2.morphologyEx(disc_mask, cv2.MORPH_CLOSE, kernel,
                                 iterations=3)
    disc_mask = cv2.morphologyEx(disc_mask, cv2.MORPH_OPEN, kernel,
                                 iterations=2)

    contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    # Largest contour → optic disc
    disc_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(disc_contour) < 100:
        return 0.0

    (_, _), disc_radius = cv2.minEnclosingCircle(disc_contour)

    # ----- Detect optic cup (brighter inner region) ----- #
    cup_thresh_val = np.percentile(blurred, 95)
    _, cup_mask = cv2.threshold(blurred, int(cup_thresh_val), 255,
                                cv2.THRESH_BINARY)

    cup_mask = cv2.morphologyEx(cup_mask, cv2.MORPH_CLOSE, kernel,
                                iterations=2)

    cup_contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    if not cup_contours:
        return 0.0

    cup_contour = max(cup_contours, key=cv2.contourArea)
    (_, _), cup_radius = cv2.minEnclosingCircle(cup_contour)

    if disc_radius < 1.0:
        return 0.0

    cdr = cup_radius / disc_radius
    return min(cdr, 1.0)  # Clamp to [0, 1]


def extract_glaucoma_features(image_uint8: np.ndarray) -> dict:
    """Extract Glaucoma features (CDR only — per design constraint)."""
    cdr = _estimate_cdr(image_uint8)
    return {
        "cdr": round(cdr, 4),
    }


# ================================================================== #
#                             MYOPIA                                   #
# ================================================================== #

def extract_myopia_features(image_uint8: np.ndarray) -> dict:
    """
    Extract Myopia-related features.

    Literature: META-PM / Ohno-Matsui (2014) — pathological myopia
    classification based on fundus appearance.

    - brightness : mean intensity of the green channel (most informative
      for retinal vessels).
    - texture_variance : variance of pixel intensities — low variance
      may indicate pathological myopic degeneration.
    - vessel_visibility : Laplacian-based proxy for vessel sharpness.
    - edge_density : ratio of Canny edge pixels to total pixels —
      low edge density may indicate myopic fundus degeneration
      (META-PM classification, Ohno-Matsui).
    """
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    green = image_uint8[:, :, 1]  # Green channel

    brightness = float(np.mean(green))
    texture_variance = float(np.var(gray))

    # Vessel visibility proxy: mean absolute Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    vessel_visibility = float(np.mean(np.abs(laplacian)))

    # Edge density: Canny edge pixel ratio
    edges = cv2.Canny(gray, 50, 150)
    total_pixels = gray.shape[0] * gray.shape[1]
    edge_density = float(np.sum(edges > 0) / total_pixels) if total_pixels > 0 else 0.0

    return {
        "brightness": round(brightness, 2),
        "texture_variance": round(texture_variance, 2),
        "vessel_visibility": round(vessel_visibility, 2),
        "edge_density": round(edge_density, 4),
    }


# ================================================================== #
#                            CATARACT                                  #
# ================================================================== #

def extract_cataract_features(image_uint8: np.ndarray) -> dict:
    """
    Extract Cataract-related features from fundus images.

    Literature:
    - Chylack et al. (1993) LOCS III: Lens opacities cause progressive
      image degradation detectable on fundus photography.
    - WHO Blindness Survey Protocol: Cataract causes overall image
      haziness and reduced contrast.

    Features:
    - blur_score : Laplacian variance — lower indicates blurriness
      consistent with cataract-induced lens opacity.
    - contrast_ratio : Standard deviation of pixel intensities —
      cataracts reduce image contrast.
    - haze_index : Ratio of mean to std brightness — higher = more
      uniform (hazy) image typical of cataract.
    """
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)

    # Blur score: Laplacian variance (lower = blurrier)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Contrast ratio: pixel intensity standard deviation
    contrast_ratio = float(np.std(gray))

    # Haze index: mean / std — higher means more uniform (hazy)
    mean_val = float(np.mean(gray))
    haze_index = mean_val / contrast_ratio if contrast_ratio > 1.0 else 10.0

    return {
        "blur_score": round(blur_score, 2),
        "contrast_ratio": round(contrast_ratio, 2),
        "haze_index": round(haze_index, 4),
    }


# ================================================================== #
#                              AMD                                     #
# ================================================================== #

def extract_amd_features(image_uint8: np.ndarray) -> dict:
    """
    Extract Age-related Macular Degeneration features.

    Literature:
    - AREDS Report No. 8: Large drusen (>125 μm) as primary risk factor.
    - Ferris et al. (2013) Beckman classification: Drusen size and
      pigmentary changes define AMD staging.
    - AREDS2: Macular pigment changes associated with progression.

    Features:
    - drusen_count : Number of bright spot-like deposits in the macular
      region (central 1/3 of the image).
    - macular_irregularity : Texture variance in the macular zone —
      irregularity suggests RPE changes.
    """
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Focus on macular region — central third of the image
    # (approximation; true macula is temporal to disc)
    y1, y2 = h // 3, 2 * h // 3
    x1, x2 = w // 3, 2 * w // 3
    macula_roi = gray[y1:y2, x1:x2]

    # Drusen detection: bright spots in the macular region
    # Drusen appear as yellow-white deposits, bright on grayscale
    blurred_roi = cv2.GaussianBlur(macula_roi, (5, 5), 0)
    thresh_val = np.percentile(blurred_roi, 92)
    _, drusen_mask = cv2.threshold(blurred_roi, int(thresh_val), 255,
                                   cv2.THRESH_BINARY)

    # Morphological cleanup to isolate individual drusen
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    drusen_mask = cv2.morphologyEx(drusen_mask, cv2.MORPH_OPEN,
                                    kernel_small, iterations=1)

    contours, _ = cv2.findContours(drusen_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # Filter: drusen are small (10-500 px) and roughly circular
    drusen_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if 10 < area < 500:
            perimeter = cv2.arcLength(c, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.3:
                    drusen_contours.append(c)

    drusen_count = len(drusen_contours)

    # Macular irregularity: local texture variance in macular zone
    # Using Laplacian as proxy for texture complexity
    lap = cv2.Laplacian(macula_roi, cv2.CV_64F)
    macular_irregularity = float(np.std(np.abs(lap)))

    # Normalise to [0, 1] range (empirical)
    macular_irregularity = min(macular_irregularity / 30.0, 1.0)

    return {
        "drusen_count": drusen_count,
        "macular_irregularity": round(macular_irregularity, 4),
    }


# ================================================================== #
#                          HYPERTENSION                                 #
# ================================================================== #

def extract_hypertension_features(image_uint8: np.ndarray) -> dict:
    """
    Extract Hypertensive Retinopathy features.

    Literature:
    - Wong & Mitchell (2004), Lancet: AV ratio (arteriole-to-venule)
      is a hallmark of hypertensive retinopathy.
    - Keith-Wagener-Barker classification: Grading based on vessel
      changes, hemorrhages, and papilledema.
    - Hubbard et al. (1999): AVR (arteriolar-venular ratio) methodology.

    Features:
    - av_ratio : Proxy for arteriole-venule width ratio. Normal ~0.67-0.75.
      Measured via vessel thickness analysis on green channel.
    - vessel_tortuosity : Proxy for vessel path irregularity.
      Hypertensive vessels show increased tortuosity.
    """
    green = image_uint8[:, :, 1]  # Green channel — best vessel contrast

    # CLAHE enhancement for vessel visibility
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    # Vessel segmentation via adaptive thresholding
    # Vessels are dark on green channel
    inverted = cv2.bitwise_not(enhanced)
    vessel_mask = cv2.adaptiveThreshold(
        inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, -2
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_CLOSE,
                                    kernel, iterations=1)
    vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN,
                                    kernel, iterations=1)

    # AV Ratio proxy: use vessel thickness distribution
    # Thicker vessels = veins, thinner = arteries
    # Compute distance transform to estimate vessel widths
    dist_transform = cv2.distanceTransform(vessel_mask, cv2.DIST_L2, 5)
    vessel_widths = dist_transform[dist_transform > 0]

    if len(vessel_widths) > 10:
        # Arteries: thinner vessels (lower quartile)
        # Veins: thicker vessels (upper quartile)
        q25 = np.percentile(vessel_widths, 25)
        q75 = np.percentile(vessel_widths, 75)
        av_ratio = q25 / q75 if q75 > 0 else 0.0
    else:
        av_ratio = 0.0

    # Vessel tortuosity: curvature proxy using contour analysis
    contours, _ = cv2.findContours(vessel_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    tortuosity_values = []
    for c in contours:
        if cv2.arcLength(c, False) > 30:  # Only significant vessels
            arc_len = cv2.arcLength(c, False)
            # Chord length (start to end)
            if len(c) >= 2:
                start = c[0][0]
                end = c[-1][0]
                chord = np.sqrt((start[0] - end[0])**2 +
                                (start[1] - end[1])**2)
                if chord > 5:
                    tortuosity_values.append(arc_len / chord)

    vessel_tortuosity = float(np.mean(tortuosity_values)) if tortuosity_values else 1.0
    # Normalise: tortuosity = 1.0 is straight, higher = more tortuous
    vessel_tortuosity = max(0, vessel_tortuosity - 1.0)  # deviation from straight

    return {
        "av_ratio": round(av_ratio, 4),
        "vessel_tortuosity": round(vessel_tortuosity, 4),
    }


# ================================================================== #
#                     NORMAL / OTHER                                    #
# ================================================================== #

def extract_normal_features(image_uint8: np.ndarray) -> dict:
    """
    Extract features indicative of a Normal retinal image.

    Literature:
    - AAO Comprehensive Eye Exam Guidelines: A normal fundus has
      clear optic disc margins, regular vessel caliber, and
      no pathological lesions.

    Features:
    - overall_clarity : Laplacian variance — higher = sharper/clearer.
    - color_uniformity : Low std across channels suggests healthy fundus.
    - anomaly_free : True if no significant anomalies detected.
    """
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)

    # Overall clarity: Laplacian variance (higher = clearer image)
    clarity = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Color uniformity: check std across RGB channels
    channel_stds = [float(np.std(image_uint8[:, :, c])) for c in range(3)]
    color_uniformity = float(np.mean(channel_stds))

    return {
        "overall_clarity": round(clarity, 2),
        "color_uniformity": round(color_uniformity, 2),
    }


def extract_other_features(image_uint8: np.ndarray) -> dict:
    """
    Extract features for "Other Abnormalities" — anomaly detection.

    Uses texture deviation from expected healthy retinal patterns as
    an anomaly proxy.

    Features:
    - anomaly_score : Higher values indicate more deviation from
      expected healthy patterns.
    - irregularity_density : Fraction of image with irregular patterns.
    """
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Anomaly detection via Gabor filtering
    # Healthy retinas have regular vessel patterns
    # Anomalous textures deviate from these patterns
    gabor_responses = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        kern = cv2.getGaborKernel(
            ksize=(21, 21), sigma=4.0, theta=theta,
            lambd=10.0, gamma=0.5, psi=0
        )
        filtered = cv2.filter2D(gray, cv2.CV_64F, kern)
        gabor_responses.append(np.abs(filtered))

    # Combined response
    combined = np.mean(gabor_responses, axis=0)

    # Anomaly score: deviation from median response
    median_response = np.median(combined)
    anomaly_map = np.abs(combined - median_response)
    anomaly_score = float(np.mean(anomaly_map))

    # Irregularity density: fraction of high-deviation pixels
    threshold = median_response + 2 * np.std(combined)
    irregular_pixels = np.sum(combined > threshold)
    irregularity_density = irregular_pixels / (h * w) if h * w > 0 else 0.0

    return {
        "anomaly_score": round(anomaly_score, 2),
        "irregularity_density": round(irregularity_density, 4),
    }


# ================================================================== #
#                     COMBINED EXTRACTION                              #
# ================================================================== #

def extract_clinical_features(image: np.ndarray) -> dict:
    """
    Run all clinical feature extractors on a retinal fundus image.

    Covers all 8 ODIR disease classes.

    Parameters
    ----------
    image : np.ndarray
        RGB image, uint8 (0-255) or float32 (0-1).

    Returns
    -------
    dict with keys: ``dr``, ``glaucoma``, ``myopia``, ``cataract``,
                    ``amd``, ``hypertension``, ``normal``, ``other``.
    """
    # Ensure uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            img = (image * 255).astype(np.uint8)
        else:
            img = image.astype(np.uint8)
    else:
        img = image.copy()

    return {
        "dr": extract_dr_features(img),
        "glaucoma": extract_glaucoma_features(img),
        "myopia": extract_myopia_features(img),
        "cataract": extract_cataract_features(img),
        "amd": extract_amd_features(img),
        "hypertension": extract_hypertension_features(img),
        "normal": extract_normal_features(img),
        "other": extract_other_features(img),
    }


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python -m src.clinical_features <image_path>")
        sys.exit(1)
    img = cv2.imread(sys.argv[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    features = extract_clinical_features(img)
    print(json.dumps(features, indent=2))
