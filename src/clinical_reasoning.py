"""
clinical_reasoning.py
---------------------
Clinical decision support reasoning layer for RetinAI.

Consumes model predictions, clinical CV features, and patient context
to produce a structured clinical assessment — WITHOUT ever modifying
model confidence scores.

██  SAFETY  ██  This module is a READ-ONLY consumer of model output.
All reasoning is additive annotation. Raw predictions pass through
unchanged.

Literature References
---------------------
-  ETDRS (1991): DR severity grading
-  AAO PPP for Glaucoma (2020): Risk factors, CDR thresholds
-  AAO PPP for DR (2019): Screening intervals
-  AREDS / AREDS2: AMD staging and progression risk
-  WHO Global Report on Vision (2019): Cataract prevalence
-  Wong & Mitchell (2004), Lancet: Hypertensive retinopathy
-  Holden et al. (2016): Myopia global prevalence
-  Ferris et al. (2013): Beckman AMD classification
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ================================================================== #
#  DISEASE METADATA — Clinically accurate descriptions                 #
# ================================================================== #

DISEASE_INFO = {
    "Normal": {
        "description": "No significant retinal abnormalities detected.",
        "urgency_base": "routine",
        "recommendation": "Continue routine eye examinations as recommended for your age group.",
    },
    "Diabetes": {
        "description": "Diabetic Retinopathy — damage to retinal blood vessels caused by diabetes.",
        "urgency_base": "semi-urgent",
        "recommendation": "Refer to ophthalmologist for dilated fundus examination. If diabetic, ensure HbA1c and blood sugar monitoring.",
    },
    "Glaucoma": {
        "description": "Glaucoma — progressive optic nerve damage, often associated with elevated intraocular pressure.",
        "urgency_base": "semi-urgent",
        "recommendation": "Refer for intraocular pressure (IOP) measurement and visual field testing. Early detection is critical to prevent irreversible vision loss.",
    },
    "Cataract": {
        "description": "Cataract — clouding of the eye's natural lens, causing blurred vision.",
        "urgency_base": "routine",
        "recommendation": "Refer for slit-lamp examination. Surgical evaluation if vision is significantly affected.",
    },
    "Age-related Macular Degeneration": {
        "description": "AMD — deterioration of the macula affecting central vision.",
        "urgency_base": "semi-urgent",
        "recommendation": "Refer for OCT imaging and fundus fluorescein angiography. Consider AREDS2 supplementation if indicated.",
    },
    "Hypertension": {
        "description": "Hypertensive Retinopathy — retinal vascular changes due to high blood pressure.",
        "urgency_base": "semi-urgent",
        "recommendation": "Blood pressure assessment recommended. Refer for cardiovascular evaluation if not already managed.",
    },
    "Myopia": {
        "description": "Pathological Myopia — degenerative changes in the eye associated with high myopia.",
        "urgency_base": "routine",
        "recommendation": "Monitor for myopic degeneration. Regular follow-up with refraction assessment.",
    },
    "Other Abnormalities": {
        "description": "Other retinal abnormalities detected that do not fit standard classification categories.",
        "urgency_base": "semi-urgent",
        "recommendation": "Comprehensive ophthalmic evaluation recommended to characterise the finding.",
    },
}

# Urgency rankings for prioritisation
URGENCY_RANK = {"urgent": 3, "semi-urgent": 2, "routine": 1}


# ================================================================== #
#  RISK STRATIFICATION — Per-class threshold–relative                  #
# ================================================================== #

def stratify_risk(probability: float, threshold: float,
                  margin: float = 0.15) -> str:
    """
    Assign a risk level relative to the per-class optimised threshold.

    Parameters
    ----------
    probability : float
        Raw model sigmoid output for this class.
    threshold : float
        Per-class optimised threshold from model checkpoint.
    margin : float
        Margin around threshold for borderline classification.

    Returns
    -------
    str : "High Risk" | "Borderline" | "Low Risk"

    Safety
    ------
    Any probability >= 0.80 is ALWAYS classified as "High Risk"
    regardless of per-class threshold. This absolute safety floor
    prevents clinically dangerous mislabeling for diseases with
    high optimised thresholds (e.g., Glaucoma 0.86, AMD 0.93).

    Literature
    ----------
    Risk stratification relative to operating-point thresholds is
    standard in clinical decision support (Moons et al., 2015,
    Transparent Reporting of Prediction Models — TRIPOD).
    """
    # ██ SAFETY FLOOR ██
    # Absolute ceiling: very high model confidence must always be High Risk.
    # Without this, diseases with thresholds > 0.85 would require prob >= 0.99
    # for "High Risk", mislabeling 0.95 confidence as "Borderline".
    if probability >= 0.80:
        return "High Risk"

    # Relative stratification for moderate probabilities
    upper_bound = threshold + margin

    if probability >= upper_bound:
        return "High Risk"
    elif probability >= max(threshold - margin, 0.0):
        return "Borderline"
    else:
        return "Low Risk"


# ================================================================== #
#  CLINICAL REASONING — Main function                                  #
# ================================================================== #

def clinical_reasoning(
    predictions: dict,
    confidences: dict,
    clinical_features: dict,
    clinical_flags: list,
    patient_context: dict,
    thresholds: dict,
) -> dict:
    """
    Produce a structured clinical assessment from model output,
    clinical features, and patient context.

    ██  CRITICAL  ██  This function NEVER modifies ``confidences``.
    All model predictions pass through unchanged.

    Parameters
    ----------
    predictions : dict
        Disease name → bool (detected/not detected).
    confidences : dict
        Disease name → float (raw sigmoid probabilities, UNCHANGED).
    clinical_features : dict
        Output of ``extract_clinical_features()``.
    clinical_flags : list[dict]
        Output of ``apply_validation_rules()`` — clinical annotations.
    patient_context : dict
        Output of ``interpret_patient_context()`` — patient metadata.
    thresholds : dict
        Per-class optimised thresholds: column_key → float.
        e.g. {'N': 0.5, 'D': 0.45, ...}

    Returns
    -------
    dict:
        raw_predictions : dict  — UNCHANGED model output
        raw_confidences : dict  — UNCHANGED model probabilities
        risk_levels : dict      — disease → risk level string
        key_findings : list     — only significant findings
        all_findings : list     — all 8 classes with details
        recommendations : list  — actionable recommendations
        urgency : str           — overall urgency level
        explanation : str       — template-based plain-English summary
        clinical_flags : list   — validation annotations
        patient_summary : str   — one-liner patient profile
        model_version : str     — for audit trail
    """
    from src.data_loader import DISEASE_COLUMNS, DISEASE_NAMES

    # Build threshold lookup by disease name
    thresh_by_name = {}
    for col in DISEASE_COLUMNS:
        name = DISEASE_NAMES[col]
        thresh_by_name[name] = thresholds.get(col, 0.5)

    # ---- Risk stratification for each class ---- #
    risk_levels = {}
    all_findings = []

    for disease_name, prob in confidences.items():
        thresh = thresh_by_name.get(disease_name, 0.5)
        risk = stratify_risk(prob, thresh)
        risk_levels[disease_name] = risk

        # Get clinical flags for this disease
        disease_flags = [f for f in clinical_flags
                         if f.get("disease") == disease_name]

        # Get patient context for this disease
        context_notes = []
        risk_modifier = "neutral"
        if patient_context:
            for rc in patient_context.get("risk_context", []):
                if rc.get("disease") == disease_name:
                    context_notes.append(rc["context_note"])
                    risk_modifier = rc["risk_modifier"]

        info = DISEASE_INFO.get(disease_name, {})

        finding = {
            "disease": disease_name,
            "probability": prob,
            "threshold": thresh,
            "risk_level": risk,
            "detected": predictions.get(disease_name, False),
            "description": info.get("description", ""),
            "recommendation": info.get("recommendation", ""),
            "supporting_evidence": [f["evidence"] for f in disease_flags
                                    if f["flag_type"] == "supporting"],
            "contradicting_evidence": [f["evidence"] for f in disease_flags
                                       if f["flag_type"] == "contradicting"],
            "patient_context": context_notes,
            "risk_modifier": risk_modifier,
        }
        all_findings.append(finding)

    # ---- Key findings: only significant ones ---- #
    key_findings = [f for f in all_findings
                    if f["risk_level"] in ("High Risk", "Borderline")
                    and f["disease"] != "Normal"]

    # Sort by risk (High first) then by probability
    risk_order = {"High Risk": 0, "Borderline": 1, "Low Risk": 2}
    key_findings.sort(key=lambda f: (risk_order.get(f["risk_level"], 3),
                                      -f["probability"]))

    # ---- Overall urgency ---- #
    urgency = "routine"
    for f in key_findings:
        base = DISEASE_INFO.get(f["disease"], {}).get("urgency_base", "routine")
        if f["risk_level"] == "High Risk":
            # Elevate urgency for high-risk findings
            if base == "semi-urgent":
                base = "urgent"
        if URGENCY_RANK.get(base, 0) > URGENCY_RANK.get(urgency, 0):
            urgency = base

    # ---- Summary status (decoupled from urgency) ---- #
    # This ensures the screening summary accurately reflects findings
    # even for diseases with "routine" urgency_base (e.g., Myopia, Cataract).
    has_high = any(f["risk_level"] == "High Risk" for f in key_findings)
    has_borderline = any(f["risk_level"] == "Borderline" for f in key_findings)

    if has_high:
        summary_status = "ABNORMAL — Findings detected"
    elif has_borderline:
        summary_status = "MONITOR — Possible abnormalities"
    else:
        summary_status = "NORMAL — No significant abnormalities detected"

    # ---- Recommendations from key findings ---- #
    recommendations = []
    seen_recs = set()
    for f in key_findings:
        rec = f.get("recommendation", "")
        if rec and rec not in seen_recs:
            recommendations.append(rec)
            seen_recs.add(rec)

    # ---- Uncertainty detection ---- #
    all_low = all(f["risk_level"] == "Low Risk" for f in all_findings
                  if f["disease"] != "Normal")
    normal_conf = confidences.get("Normal", 0.0)
    normal_thresh = thresh_by_name.get("Normal", 0.5)

    uncertain = False
    if all_low and normal_conf < normal_thresh:
        uncertain = True

    # ---- Entropy-based OOD check ---- #
    probs = list(confidences.values())
    # High entropy = model is uncertain about everything
    entropy = -sum(p * _safe_log(p) + (1-p) * _safe_log(1-p)
                   for p in probs) / max(len(probs), 1)
    ood_suspected = entropy > 0.65  # empirical threshold

    if ood_suspected:
        uncertain = True

    # ---- Generate explanation ---- #
    explanation = _generate_explanation(
        key_findings, all_findings, uncertain,
        patient_context.get("patient_summary", "") if patient_context else ""
    )

    return {
        "raw_predictions": dict(predictions),
        "raw_confidences": dict(confidences),
        "risk_levels": risk_levels,
        "key_findings": [_clean_finding(f) for f in key_findings],
        "all_findings": [_clean_finding(f) for f in all_findings],
        "recommendations": recommendations,
        "summary_status": summary_status,
        "urgency": urgency,
        "uncertain": uncertain,
        "ood_suspected": ood_suspected,
        "explanation": explanation,
        "clinical_flags": clinical_flags,
        "patient_summary": patient_context.get("patient_summary", "") if patient_context else "",
        "model_version": "EfficientNet-B3 v2.0 (ODIR-5K)",
    }


# ================================================================== #
#  TEMPLATE-BASED EXPLANATION GENERATOR                                #
# ================================================================== #

def _generate_explanation(key_findings, all_findings, uncertain,
                          patient_summary) -> str:
    """Generate a plain-English explanation of the screening results."""

    parts = []

    if patient_summary:
        parts.append(f"**Patient Profile:** {patient_summary}.")

    if uncertain:
        parts.append(
            "**⚠ Note:** The AI screening produced uncertain results. "
            "This may be due to image quality or an unusual presentation. "
            "Clinical evaluation is recommended regardless of the AI output."
        )
        return "\n\n".join(parts)

    if not key_findings:
        parts.append(
            "**Screening Result:** No significant retinal abnormalities "
            "were detected in this image. The retinal fundus appears "
            "within normal limits based on AI analysis."
        )
        parts.append(
            "**Recommendation:** Continue routine eye examinations as "
            "appropriate for your age and risk factors."
        )
        return "\n\n".join(parts)

    # Significant findings
    parts.append("**Screening Result:** The AI analysis identified the "
                 "following findings that warrant clinical attention:")

    for f in key_findings:
        risk_emoji = "🔴" if f["risk_level"] == "High Risk" else "🟡"
        line = f"\n{risk_emoji} **{f['disease']}** — {f['risk_level']}"
        if f.get("description"):
            line += f"\n  {f['description']}"

        # Supporting evidence
        if f.get("supporting_evidence"):
            line += "\n  _Supporting evidence:_ " + "; ".join(f["supporting_evidence"][:2])

        # Contradicting evidence (important for clinical safety)
        if f.get("contradicting_evidence"):
            line += "\n  _⚠ Note:_ " + "; ".join(f["contradicting_evidence"][:2])

        # Patient context
        if f.get("patient_context"):
            line += "\n  _Patient context:_ " + "; ".join(f["patient_context"][:2])

        parts.append(line)

    # Recommendations
    parts.append("\n**Recommendations:**")
    for i, rec in enumerate(key_findings[:3], 1):
        if rec.get("recommendation"):
            parts.append(f"{i}. {rec['recommendation']}")

    return "\n\n".join(parts)


# ================================================================== #
#  HELPERS                                                             #
# ================================================================== #

def _safe_log(x):
    """Safe log for entropy calculation."""
    import math
    return math.log(max(x, 1e-10))


def _clean_finding(finding: dict) -> dict:
    """Ensure all values are JSON-serialisable."""
    return {
        "disease": finding["disease"],
        "probability": round(finding["probability"], 4),
        "threshold": round(finding["threshold"], 4),
        "risk_level": finding["risk_level"],
        "detected": bool(finding["detected"]),
        "description": finding.get("description", ""),
        "recommendation": finding.get("recommendation", ""),
        "supporting_evidence": finding.get("supporting_evidence", []),
        "contradicting_evidence": finding.get("contradicting_evidence", []),
        "patient_context": finding.get("patient_context", []),
        "risk_modifier": finding.get("risk_modifier", "neutral"),
    }
