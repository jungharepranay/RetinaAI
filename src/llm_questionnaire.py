"""
llm_questionnaire.py
--------------------
Patient context module for the RetinAI diagnostic assistant.

Provides a minimal clinical intake form and interprets patient
context for use by the clinical reasoning layer.

██  SAFETY  ██  This module NEVER modifies model predictions or
confidence scores.  Patient information is used ONLY for:
  - Adjusting risk interpretation
  - Contextualising recommendations
  - Improving explanation text

Literature References
---------------------
- ETDRS (1991): Diabetes duration as strongest DR risk factor
- AAO PPP Glaucoma (2020): Age >40, family history as risk factors
- Klein et al. (2002), WESDR: Hypertension doubles DR risk
- WHO (2019): Age >60 primary risk factor for cataract and AMD
"""


# ------------------------------------------------------------------ #
#  PATIENT INTAKE FORM — Minimal clinical context                     #
# ------------------------------------------------------------------ #

INTAKE_FIELDS = {
    "age": {
        "label": "Age",
        "type": "number",
        "required": True,
        "placeholder": "e.g. 55",
        "min": 1,
        "max": 120,
    },
    "diabetic": {
        "label": "Diabetic",
        "type": "yes_no",
        "required": True,
    },
    "hypertension": {
        "label": "Hypertension",
        "type": "yes_no",
        "required": False,
    },
    "vision_issues": {
        "label": "Vision Concerns",
        "type": "select",
        "required": False,
        "options": [
            "None",
            "Blurry or hazy vision",
            "Difficulty seeing from the sides",
            "Things far away look blurry",
            "Seeing spots, floaters, or flashes of light",
            "Other concerns",
        ],
    },
}


def get_intake_fields() -> dict:
    """Return the intake form field definitions for the frontend."""
    return dict(INTAKE_FIELDS)


# ------------------------------------------------------------------ #
#  PATIENT CONTEXT INTERPRETATION                                      #
# ------------------------------------------------------------------ #

def interpret_patient_context(
    predictions: dict,
    confidences: dict,
    patient_info: dict,
) -> dict:
    """
    Interpret patient context to produce risk modifiers and contextual
    notes for the clinical reasoning layer.

    ██  CRITICAL  ██  This function NEVER modifies predictions or
    confidence scores.  It returns interpretation metadata ONLY.

    Parameters
    ----------
    predictions : dict
        Disease name → bool.
    confidences : dict
        Disease name → float (raw, unmodified).
    patient_info : dict
        Keys: age (int|None), diabetic (str|None "yes"/"no"),
              hypertension (str|None), vision_issues (str|None).

    Returns
    -------
    dict with keys:
        risk_context : list[dict]
            Per-disease context notes with keys:
            {disease, context_note, risk_modifier, literature_ref}
            risk_modifier: "elevated" | "reduced" | "neutral"
        recommendation_notes : list[str]
            Additional recommendation text based on patient profile.
        patient_summary : str
            Human-readable one-liner summarising the patient profile.
    """
    age = patient_info.get("age")
    diabetic = str(patient_info.get("diabetic", "")).strip().lower()
    hypertension = str(patient_info.get("hypertension", "")).strip().lower()
    vision_issues = str(patient_info.get("vision_issues", "")).strip().lower()

    # Try to parse age
    try:
        age = int(age) if age else None
    except (ValueError, TypeError):
        age = None

    risk_context = []
    recommendation_notes = []

    # --- Build patient summary --- #
    parts = []
    if age:
        parts.append(f"{age}-year-old")
    if diabetic == "yes":
        parts.append("diabetic")
    elif diabetic == "no":
        parts.append("non-diabetic")
    if hypertension == "yes":
        parts.append("hypertensive")
    if vision_issues and vision_issues not in ("none", ""):
        parts.append(f"reports {vision_issues}")
    patient_summary = " ".join(parts) + " patient" if parts else "Patient (no demographics provided)"

    # ---------------------------------------------------------------- #
    #  DIABETIC RETINOPATHY CONTEXT                                     #
    # ---------------------------------------------------------------- #
    if diabetic == "no":
        risk_context.append({
            "disease": "Diabetes",
            "context_note": "Patient reports no diabetes history — DR "
                            "without underlying diabetes is extremely rare",
            "risk_modifier": "reduced",
            "literature_ref": "ETDRS (1991): DR requires underlying "
                              "diabetes; non-diabetic retinopathy has "
                              "different aetiology",
        })
    elif diabetic == "yes":
        risk_context.append({
            "disease": "Diabetes",
            "context_note": "Confirmed diabetes history — consistent "
                            "with DR risk profile",
            "risk_modifier": "elevated",
            "literature_ref": "ETDRS: Diabetes duration is the strongest "
                              "risk factor for DR development and "
                              "progression",
        })
        recommendation_notes.append(
            "Diabetic patient: recommend annual dilated fundus "
            "examination per AAO guidelines."
        )

        if age and age > 50:
            recommendation_notes.append(
                "Age >50 with diabetes: increased risk of diabetic "
                "macular edema — consider OCT if available."
            )

    # ---------------------------------------------------------------- #
    #  GLAUCOMA CONTEXT                                                  #
    # ---------------------------------------------------------------- #
    if age and age > 40:
        risk_context.append({
            "disease": "Glaucoma",
            "context_note": f"Patient age {age} >40 — elevated "
                            f"glaucoma risk",
            "risk_modifier": "elevated",
            "literature_ref": "AAO PPP Glaucoma (2020): Prevalence "
                              "increases significantly after age 40; "
                              "Quigley & Broman (2006) global "
                              "prevalence study",
        })

    if vision_issues and "sides" in vision_issues:
        risk_context.append({
            "disease": "Glaucoma",
            "context_note": "Patient reports peripheral vision loss — "
                            "classic symptom of glaucomatous damage",
            "risk_modifier": "elevated",
            "literature_ref": "AAO PPP: Arcuate scotoma and peripheral "
                              "field loss are hallmarks of glaucoma",
        })

    # ---------------------------------------------------------------- #
    #  CATARACT CONTEXT                                                  #
    # ---------------------------------------------------------------- #
    if age and age > 60:
        risk_context.append({
            "disease": "Cataract",
            "context_note": f"Patient age {age} >60 — age-related "
                            f"cataract is very common",
            "risk_modifier": "elevated",
            "literature_ref": "WHO (2019): Age >60 is the primary risk "
                              "factor for cataract; prevalence >50% "
                              "above age 70",
        })
    if diabetic == "yes":
        risk_context.append({
            "disease": "Cataract",
            "context_note": "Diabetes accelerates cataract formation "
                            "(osmotic stress in lens)",
            "risk_modifier": "elevated",
            "literature_ref": "Klein et al. (1998) WESDR: Diabetic "
                              "patients develop cataracts 2-5x faster "
                              "than non-diabetics",
        })

    # ---------------------------------------------------------------- #
    #  AMD CONTEXT                                                       #
    # ---------------------------------------------------------------- #
    if age and age > 55:
        risk_context.append({
            "disease": "Age-related Macular Degeneration",
            "context_note": f"Patient age {age} >55 — primary AMD risk "
                            f"factor",
            "risk_modifier": "elevated",
            "literature_ref": "AREDS Report No. 3: Age is the strongest "
                              "risk factor; prevalence increases sharply "
                              "after age 55",
        })

    if hypertension == "yes":
        risk_context.append({
            "disease": "Age-related Macular Degeneration",
            "context_note": "Hypertension may increase AMD risk via "
                            "choroidal perfusion changes",
            "risk_modifier": "elevated",
            "literature_ref": "Hyman et al. (2000): Hypertension "
                              "associated with increased neovascular "
                              "AMD risk",
        })

    # ---------------------------------------------------------------- #
    #  HYPERTENSION CONTEXT                                              #
    # ---------------------------------------------------------------- #
    if hypertension == "yes":
        risk_context.append({
            "disease": "Hypertension",
            "context_note": "Confirmed hypertension — retinal vascular "
                            "changes expected",
            "risk_modifier": "elevated",
            "literature_ref": "Wong & Mitchell (2004), Lancet: "
                              "Hypertensive retinopathy prevalence "
                              "correlates with BP severity and duration",
        })
    elif hypertension == "no":
        risk_context.append({
            "disease": "Hypertension",
            "context_note": "No hypertension reported — reduces baseline "
                            "risk of hypertensive retinopathy",
            "risk_modifier": "reduced",
            "literature_ref": "Keith-Wagener-Barker: Retinopathy staging "
                              "requires sustained hypertension",
        })

    # ---------------------------------------------------------------- #
    #  MYOPIA CONTEXT                                                    #
    # ---------------------------------------------------------------- #
    if vision_issues and "far away" in vision_issues:
        risk_context.append({
            "disease": "Myopia",
            "context_note": "Patient reports difficulty seeing at "
                            "distance — symptom consistent with myopia",
            "risk_modifier": "elevated",
            "literature_ref": "Holden et al. (2016), Ophthalmology: "
                              "Distance blur is the primary symptom "
                              "of myopic refractive error",
        })

    # General recommendations
    if age and age > 40:
        recommendation_notes.append(
            "Age >40: recommend comprehensive eye examination every "
            "1-2 years per AAO screening guidelines."
        )

    if not recommendation_notes:
        recommendation_notes.append(
            "Follow up with an ophthalmologist for comprehensive "
            "clinical evaluation."
        )

    return {
        "risk_context": risk_context,
        "recommendation_notes": recommendation_notes,
        "patient_summary": patient_summary,
    }


# ------------------------------------------------------------------ #
#  QUICK TEST                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import json

    preds = {"Diabetes": True, "Glaucoma": True, "Myopia": False,
             "Normal": False, "Hypertension": False}
    confs = {"Diabetes": 0.78, "Glaucoma": 0.65, "Myopia": 0.12,
             "Hypertension": 0.10, "Normal": 0.05}
    patient = {"age": 58, "diabetic": "yes", "hypertension": "no",
               "vision_issues": "Blurred vision"}

    result = interpret_patient_context(preds, confs, patient)
    print(json.dumps(result, indent=2))
