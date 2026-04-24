"""
predict.py
----------
Run single-image prediction using the trained EfficientNet-B3 retinal
disease model (PyTorch).

Supports:
  - Simple prediction (``predict_image``)
  - Full clinical pipeline (``predict_initial``)
  - Context-aware assessment (``predict_with_context``)

Pipeline::

    Image → Quality Check → Clinical Features → CNN → Validation Flags
    → Clinical Reasoning → Risk Stratification → Result

██  SAFETY  ██  Model predictions are NEVER modified after inference.
The clinical reasoning layer adds interpretation, not score changes.
"""

import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import DISEASE_COLUMNS, DISEASE_NAMES
from src.efficientnet_model import (
    load_efficientnet_model,
    preprocess_image,
    preprocess_image_from_array,
    predict_single,
    DEVICE,
)

# ─── Ensemble support ─── #
try:
    from src.ensemble_model import load_ensemble, predict_ensemble
    HAS_ENSEMBLE_MODULE = True
except ImportError:
    HAS_ENSEMBLE_MODULE = False

# ─── Model paths ─── #
# Prefer full checkpoint (contains optimal thresholds), fall back to weights-only
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "efficientnet_odir_final_checkpoint.pth")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "efficientnet_odir_final.pth")
ENSEMBLE_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "ensemble_config.json")

if os.path.isfile(CHECKPOINT_PATH):
    MODEL_PATH = CHECKPOINT_PATH
else:
    MODEL_PATH = WEIGHTS_PATH

THRESHOLD = 0.5  # fallback default; overridden by checkpoint thresholds

# ─── Ensemble availability check ─── #
USE_ENSEMBLE = HAS_ENSEMBLE_MODULE and os.path.isfile(ENSEMBLE_CONFIG_PATH)


# ─── Cached model + thresholds ─── #
_model = None
_thresholds = None
_ensemble = None


def _get_ensemble():
    """Lazy-load the ensemble model (3 models combined)."""
    global _ensemble
    if _ensemble is None and USE_ENSEMBLE:
        try:
            _ensemble = load_ensemble(ENSEMBLE_CONFIG_PATH, DEVICE)
            print(f"[predict] Ensemble loaded: {len(_ensemble.models)} models")
        except Exception as e:
            print(f"[predict] Ensemble loading failed: {e}")
            print(f"[predict] Falling back to single EfficientNet model.")
    return _ensemble


def _get_model_and_thresholds():
    """Lazy-load the EfficientNet-B3 model and per-class thresholds."""
    global _model, _thresholds
    if _model is None:
        _model, _thresholds = load_efficientnet_model(MODEL_PATH, DEVICE)
    return _model, _thresholds


# ================================================================ #
#  Simple prediction                                                #
# ================================================================ #

def predict_image(image_path: str, model=None, threshold: float = None,
                  use_ensemble: bool = True):
    """
    Predict diseases for a single retinal fundus image (simple mode).

    Will automatically use ensemble if available and use_ensemble=True.

    Parameters
    ----------
    image_path : str
    model : nn.Module or None
    threshold : float or None
        If given, overrides per-class optimised thresholds.
    use_ensemble : bool
        If True and ensemble is available, use ensemble prediction.

    Returns
    -------
    detected : list[str]
    probabilities : dict[str, float]
    """
    # Try ensemble first (regardless of whether model was passed)
    if use_ensemble and threshold is None:
        ensemble = _get_ensemble()
        if ensemble is not None:
            return predict_ensemble(image_path, ensemble, DEVICE)

    # Fallback to single model
    if model is None:
        model, thresholds = _get_model_and_thresholds()
    else:
        _, thresholds = _get_model_and_thresholds()

    # If caller supplies a flat threshold, build per-class dict
    if threshold is not None:
        thresholds = {c: threshold for c in DISEASE_COLUMNS}

    return predict_single(image_path, model, thresholds, DEVICE)


# ================================================================ #
#  HYBRID CLINICAL PIPELINE                                         #
# ================================================================ #

# ------------------------------------------------------------------ #
#  PHASE 1: Initial Prediction + Clinical Assessment                   #
# ------------------------------------------------------------------ #

def predict_initial(image_path: str, model=None,
                    threshold: float = THRESHOLD,
                    output_dir: str = None,
                    patient_info: dict = None):
    """
    Full clinical pipeline — initial prediction with clinical assessment.

    Pipeline:
      1. Load & preprocess image
      2. Image quality check
      3. Clinical feature extraction (ALL 8 diseases)
      4. Ensemble prediction (EfficientNet-B3 + DenseNet-121 + ConvNeXt-Tiny)
         Falls back to single EfficientNet-B3 if ensemble unavailable.
      5. Clinical validation flags (no score modification)
      6. Patient context interpretation
      7. Clinical reasoning + risk stratification
      8. Grad-CAM explainability

    Parameters
    ----------
    patient_info : dict or None
        Keys: age (int), diabetic (str "yes"/"no"),
        hypertension (str), vision_issues (str).

    Returns
    -------
    dict with full clinical assessment.
    """
    import cv2
    from src.quality_check import check_image_quality
    from src.clinical_features import extract_clinical_features
    from src.validation_rules import apply_validation_rules
    from src.explainability import generate_gradcam_for_all
    from src.llm_questionnaire import interpret_patient_context
    from src.clinical_reasoning import clinical_reasoning

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "reports", "gradcam")

    if model is None:
        model, thresholds = _get_model_and_thresholds()
    else:
        _, thresholds = _get_model_and_thresholds()

    if patient_info is None:
        patient_info = {}

    # ---- Step 1: Load raw image ---- #
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    # ---- Step 2: Quality Check ---- #
    quality = check_image_quality(raw_rgb)
    if not quality["is_valid"]:
        return {
            "predictions": {},
            "confidence": {},
            "clinical_features": {},
            "quality_check": quality,
            "gradcam_paths": {},
            "gradcam_url": None,
            "clinical_assessment": {
                "key_findings": [],
                "all_findings": [],
                "risk_levels": {},
                "recommendations": [
                    f"Image rejected — quality issue: {quality['reason']}. "
                    f"Please upload a clearer retinal fundus image."
                ],
                "urgency": "routine",
                "uncertain": True,
                "explanation": f"Image quality check failed: {quality['reason']}. "
                               f"A clear retinal fundus image is required for "
                               f"reliable AI screening.",
            },
            "final_decision": f"Image rejected — quality issue: "
                              f"{quality['reason']}. Please upload a "
                              f"clearer image.",
        }

    # ---- Step 3: Clinical Feature Extraction (ALL 8 diseases) ---- #
    clinical_features = extract_clinical_features(raw_rgb)

    # ---- Step 4: Ensemble Prediction ---- #
    # Always try ensemble first, regardless of whether single model was passed.
    # Single model is still needed for Grad-CAM (Step 8).
    ensemble = _get_ensemble()
    if ensemble is not None:
        detected, probabilities = predict_ensemble(image_path, ensemble, DEVICE)
        pred_flags = {name: (name in detected) for name in probabilities}
        # Use ensemble thresholds for downstream modules
        thresholds = {col: ensemble.thresholds.get(col, 0.5)
                      for col in DISEASE_COLUMNS}
        print(f"[predict] Using ensemble ({len(ensemble.models)} models)")
    else:
        # Fallback to single model
        print(f"[predict] Ensemble unavailable, using single model")
        img_tensor = preprocess_image(image_path).to(DEVICE)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        probabilities = {
            DISEASE_NAMES[col]: float(round(probs[i], 4))
            for i, col in enumerate(DISEASE_COLUMNS)
        }

        detected = [
            DISEASE_NAMES[col]
            for i, col in enumerate(DISEASE_COLUMNS)
            if probs[i] >= thresholds.get(col, threshold)
        ]

        pred_flags = {name: (name in detected) for name in probabilities}

    # ---- Step 5: Clinical Validation Flags (NO score modification) ---- #
    unchanged_conf, clinical_flags = apply_validation_rules(
        pred_flags, probabilities, clinical_features
    )

    # ---- Step 6: Patient Context Interpretation ---- #
    patient_context = interpret_patient_context(
        pred_flags, probabilities, patient_info
    )

    # ---- Step 7: Clinical Reasoning + Risk Stratification ---- #
    assessment = clinical_reasoning(
        predictions=pred_flags,
        confidences=probabilities,  # RAW, UNCHANGED
        clinical_features=clinical_features,
        clinical_flags=clinical_flags,
        patient_context=patient_context,
        thresholds=thresholds,
    )

    # ---- Step 8: Grad-CAM ---- #
    gradcam_paths = {}
    try:
        gradcam_paths = generate_gradcam_for_all(
            raw_rgb, model, probabilities, output_dir,
            threshold=threshold,
        )
    except Exception as e:
        print(f"[predict] Grad-CAM generation failed: {e}")

    # Build gradcam_url (first available)
    gradcam_url = None
    if gradcam_paths:
        first_path = next(iter(gradcam_paths.values()), None)
        gradcam_url = first_path

    # ---- Build final decision ---- #
    final_decision = _build_final_decision(assessment)

    return {
        "predictions": pred_flags,
        "confidence": probabilities,
        "ensemble_confidence": probabilities,  # alias for PDF report compat
        "clinical_features": clinical_features,
        "quality_check": quality,
        "gradcam": (list(gradcam_paths.values())[0]
                    if gradcam_paths else None),
        "gradcam_paths": gradcam_paths,
        "gradcam_url": gradcam_url,
        "clinical_flags": clinical_flags,
        "clinical_assessment": assessment,
        "patient_context": patient_context,
        "final_decision": final_decision,
    }


def _build_final_decision(assessment: dict) -> str:
    """Build a human-readable final decision from clinical assessment."""
    key_findings = assessment.get("key_findings", [])
    urgency = assessment.get("urgency", "routine")

    if assessment.get("uncertain"):
        return ("Uncertain result — the AI model could not produce a "
                "confident prediction. Clinical evaluation recommended.")

    if not key_findings:
        return "No significant retinal abnormalities detected in this screening."

    parts = []
    for f in key_findings:
        parts.append(f"{f['disease']} ({f['risk_level']})")

    decision = "Findings: " + ", ".join(parts) + "."
    if urgency == "urgent":
        decision += " Urgent referral recommended."
    elif urgency == "semi-urgent":
        decision += " Clinical follow-up recommended."

    return decision


# ------------------------------------------------------------------ #
#  PHASE 2: Context-Aware Assessment                                   #
# ------------------------------------------------------------------ #

def predict_with_context(image_path: str, patient_info: dict,
                         model=None, threshold: float = THRESHOLD,
                         output_dir: str = None):
    """
    Full pipeline with patient context for interpretation.

    ██  SAFETY  ██  Patient info is used ONLY for interpretation
    and recommendations — NEVER for modifying model predictions.
    """
    return predict_initial(
        image_path, model, threshold, output_dir,
        patient_info=patient_info,
    )


# ------------------------------------------------------------------ #
#  BACKWARDS COMPAT                                                    #
# ------------------------------------------------------------------ #

def predict_image_full(image_path: str, model=None,
                       threshold: float = THRESHOLD,
                       output_dir: str = None,
                       patient_context: dict = None):
    """Backwards-compatible full pipeline (delegates to predict_initial)."""
    return predict_initial(image_path, model, threshold, output_dir)


def predict_with_answers(image_path: str, answers: dict,
                         model=None, threshold: float = THRESHOLD,
                         output_dir: str = None):
    """
    Legacy compatibility: redirect to predict_with_context.
    Converts old answer format to patient_info format.
    """
    # Map old questionnaire answers to patient_info
    patient_info = {}
    for key, val in answers.items():
        if key in ("age", "diabetic", "hypertension", "vision_issues"):
            patient_info[key] = val
    return predict_with_context(image_path, patient_info, model,
                                threshold, output_dir)


# ------------------------------------------------------------------ #
#  CLI ENTRY POINT                                                    #
# ------------------------------------------------------------------ #

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"Error: file not found — {image_path}")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH} …")
    model, thresholds = load_efficientnet_model(MODEL_PATH)

    detected, probs = predict_single(image_path, model, thresholds)

    print("\n===== Prediction Results =====")
    print(f"Image: {image_path}\n")

    print("Probabilities:")
    for disease, prob in probs.items():
        marker = " ◀" if disease in detected else ""
        print(f"  {disease:40s}  {prob:.4f}{marker}")

    print("\nDetected Diseases:")
    if detected:
        for d in detected:
            print(f"  • {d}")
    else:
        print("  (none above threshold)")


if __name__ == "__main__":
    main()
