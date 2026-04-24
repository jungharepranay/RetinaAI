"""
validation_rules.py
-------------------
Clinically grounded rule-based validation layer.

Cross-references CNN predictions against extracted clinical features
to produce structured clinical flags — WITHOUT modifying any
prediction scores.

██  SAFETY  ██  This module NEVER alters model confidence values.
All output is annotation/flagging only, consumed by the clinical
reasoning layer for interpretation.

Literature References
---------------------
- ETDRS (Early Treatment Diabetic Retinopathy Study): DR grading
- AAO Preferred Practice Patterns (PPP): Glaucoma, DR management
- Jonas et al. (1999): ISNT rule for optic disc evaluation
- Keith-Wagener-Barker classification: Hypertensive retinopathy grading
- META-PM / Ohno-Matsui (2014): Pathological myopia classification
"""


def apply_validation_rules(predictions: dict, confidences: dict,
                           clinical_features: dict) -> tuple:
    """
    Cross-validate CNN confidence scores against clinical feature evidence.

    ██  CRITICAL  ██  Confidence values are NEVER modified.
    This function returns the ORIGINAL confidences unchanged, plus a list
    of clinical flags that annotate findings for downstream reasoning.

    Parameters
    ----------
    predictions : dict
        Disease names → bool (detected or not based on threshold).
    confidences : dict
        Disease names → float (raw sigmoid probabilities).
    clinical_features : dict
        Output of ``extract_clinical_features()``.

    Returns
    -------
    unchanged_confidences : dict
        Disease names → float — IDENTICAL to input ``confidences``.
    clinical_flags : list[dict]
        Dicts with keys: ``disease``, ``flag_type``, ``severity``,
        ``evidence``, ``literature_ref``.
        flag_type: "supporting" | "contradicting" | "informational"
        severity:  "high" | "moderate" | "low"
    """
    # ██ SAFETY: Return original confidences, NEVER modify ██
    unchanged = dict(confidences)
    flags = []

    dr_features = clinical_features.get("dr", {})
    glaucoma_features = clinical_features.get("glaucoma", {})
    myopia_features = clinical_features.get("myopia", {})
    cataract_features = clinical_features.get("cataract", {})
    amd_features = clinical_features.get("amd", {})
    hypertension_features = clinical_features.get("hypertension", {})

    hemorrhage_count = dr_features.get("hemorrhage_count", 0)
    exudate_count = dr_features.get("exudate_count", 0)
    microaneurysm_count = dr_features.get("microaneurysm_count", 0)
    lesion_area = dr_features.get("lesion_area", 0.0)
    cdr = glaucoma_features.get("cdr", 0.0)
    texture_var = myopia_features.get("texture_variance", 0.0)
    brightness = myopia_features.get("brightness", 0.0)
    vessel_visibility = myopia_features.get("vessel_visibility", 0.0)
    edge_density = myopia_features.get("edge_density", 0.0)

    # ---------------------------------------------------------------- #
    #  CROSS-DISEASE: Normal vs. Disease contradiction                  #
    # ---------------------------------------------------------------- #
    detected_diseases = [k for k, v in predictions.items()
                         if v and k != "Normal"]
    if predictions.get("Normal", False) and detected_diseases:
        flags.append({
            "disease": "Normal",
            "flag_type": "contradicting",
            "severity": "high",
            "evidence": f"Normal predicted alongside {', '.join(detected_diseases)}"
                        f" — mutual exclusion expected in standard grading",
            "literature_ref": "Clinical logic: Normal and disease labels "
                              "are mutually exclusive in ODIR grading schema",
        })

    # ---------------------------------------------------------------- #
    #  DIABETIC RETINOPATHY (ETDRS, AAO PPP for DR)                     #
    # ---------------------------------------------------------------- #
    if predictions.get("Diabetes", False):
        # --- Microaneurysm evidence (earliest sign per ETDRS) --- #
        if microaneurysm_count > 0:
            flags.append({
                "disease": "Diabetes",
                "flag_type": "supporting",
                "severity": "high",
                "evidence": f"{microaneurysm_count} microaneurysm(s) detected "
                            f"— earliest clinical sign of DR",
                "literature_ref": "ETDRS Report No. 10: Microaneurysms as "
                                  "the defining feature of mild NPDR",
            })

        # --- Lesion area severity grading (ETDRS-inspired) --- #
        if lesion_area > 0.02:
            flags.append({
                "disease": "Diabetes",
                "flag_type": "supporting",
                "severity": "high",
                "evidence": f"Lesion area {lesion_area:.4f} > 2% — severe "
                            f"lesion burden detected",
                "literature_ref": "ETDRS severity grading: extensive lesion "
                                  "area correlates with proliferative risk",
            })
        elif lesion_area > 0.005:
            flags.append({
                "disease": "Diabetes",
                "flag_type": "supporting",
                "severity": "moderate",
                "evidence": f"Lesion area {lesion_area:.4f} (moderate "
                            f"lesion burden)",
                "literature_ref": "ETDRS: moderate lesion area consistent "
                                  "with moderate NPDR",
            })

        # --- No clinical evidence at all --- #
        if (hemorrhage_count == 0 and exudate_count == 0
                and microaneurysm_count == 0):
            flags.append({
                "disease": "Diabetes",
                "flag_type": "contradicting",
                "severity": "high",
                "evidence": "No hemorrhages, exudates, or microaneurysms "
                            "detected — clinical features do not support DR",
                "literature_ref": "AAO PPP for DR: Diagnosis requires "
                                  "presence of at least microaneurysms",
            })
        elif hemorrhage_count == 0 and microaneurysm_count == 0:
            flags.append({
                "disease": "Diabetes",
                "flag_type": "contradicting",
                "severity": "moderate",
                "evidence": "No hemorrhages or microaneurysms — primary "
                            "DR markers absent",
                "literature_ref": "ETDRS: Hemorrhages and microaneurysms "
                                  "are primary markers of DR progression",
            })

        # --- Strong combined evidence --- #
        if hemorrhage_count > 3 and exudate_count > 2:
            flags.append({
                "disease": "Diabetes",
                "flag_type": "supporting",
                "severity": "high",
                "evidence": "Strong combined evidence: multiple hemorrhages "
                            "and exudates observed",
                "literature_ref": "ETDRS: Co-presence of hemorrhages and "
                                  "exudates indicates moderate-severe NPDR",
            })

        # --- Cross-disease: DR + Hypertension co-occurrence --- #
        if predictions.get("Hypertension", False):
            flags.append({
                "disease": "Diabetes",
                "flag_type": "informational",
                "severity": "moderate",
                "evidence": "DR + Hypertension co-occurrence — both "
                            "conditions share hemorrhage features",
                "literature_ref": "Clinical co-morbidity: Diabetes and "
                                  "hypertension frequently co-occur and "
                                  "both cause retinal hemorrhages "
                                  "(Klein et al., WESDR Study)",
            })

    # ---------------------------------------------------------------- #
    #  GLAUCOMA (AAO PPP for Glaucoma, Jonas et al. ISNT rule)          #
    # ---------------------------------------------------------------- #
    if predictions.get("Glaucoma", False):
        if cdr > 0.8:
            flags.append({
                "disease": "Glaucoma",
                "flag_type": "supporting",
                "severity": "high",
                "evidence": f"CDR={cdr:.2f} > 0.8 — strongly indicative "
                            f"of glaucomatous optic neuropathy",
                "literature_ref": "AAO PPP for Glaucoma: CDR > 0.8 is "
                                  "a strong indicator of glaucomatous "
                                  "optic neuropathy",
            })
        elif cdr > 0.6:
            flags.append({
                "disease": "Glaucoma",
                "flag_type": "supporting",
                "severity": "moderate",
                "evidence": f"CDR={cdr:.2f} > 0.6 — clinically suspicious",
                "literature_ref": "AAO PPP: CDR > 0.6 warrants further "
                                  "evaluation; Jonas et al. ISNT rule "
                                  "notes rim thinning patterns are key",
            })
        elif cdr < 0.4:
            flags.append({
                "disease": "Glaucoma",
                "flag_type": "contradicting",
                "severity": "moderate",
                "evidence": f"CDR={cdr:.2f} < 0.4 — within normal range, "
                            f"clinical features do not support glaucoma",
                "literature_ref": "AAO PPP: CDR < 0.4 is considered "
                                  "normal in most populations; Jonas et "
                                  "al. (1999) ISNT rule",
            })
        else:
            flags.append({
                "disease": "Glaucoma",
                "flag_type": "informational",
                "severity": "low",
                "evidence": f"CDR={cdr:.2f} is borderline (0.4–0.6) — "
                            f"inconclusive without IOP / VF testing",
                "literature_ref": "AAO PPP: Borderline CDR requires "
                                  "IOP measurement and visual field "
                                  "testing for confirmation",
            })

    # ---------------------------------------------------------------- #
    #  MYOPIA (META-PM classification, Ohno-Matsui 2014)                #
    # ---------------------------------------------------------------- #
    if predictions.get("Myopia", False):
        if edge_density < 0.05 and brightness > 150:
            flags.append({
                "disease": "Myopia",
                "flag_type": "supporting",
                "severity": "moderate",
                "evidence": f"Low edge density ({edge_density:.3f}) + "
                            f"high brightness ({brightness:.0f}) — "
                            f"consistent with myopic fundus degeneration",
                "literature_ref": "META-PM (Ohno-Matsui 2014): Diffuse "
                                  "chorioretinal atrophy shows reduced "
                                  "fundus features with pale appearance",
            })

        if vessel_visibility < 8.0:
            flags.append({
                "disease": "Myopia",
                "flag_type": "supporting",
                "severity": "moderate",
                "evidence": f"Low vessel visibility ({vessel_visibility:.1f} "
                            f"< 8.0) — reduced vessel contrast",
                "literature_ref": "Ohno-Matsui: High myopia causes "
                                  "choroidal thinning, reducing vessel "
                                  "visibility on fundus examination",
            })

        if texture_var < 500:
            flags.append({
                "disease": "Myopia",
                "flag_type": "contradicting",
                "severity": "low",
                "evidence": f"texture_variance={texture_var:.0f} < 500 "
                            f"— low texture may indicate image quality "
                            f"issue rather than disease",
                "literature_ref": "META-PM: Myopic degeneration typically "
                                  "shows characteristic patterns, not "
                                  "uniform low texture",
            })

        if brightness > 180:
            flags.append({
                "disease": "Myopia",
                "flag_type": "supporting",
                "severity": "moderate",
                "evidence": f"High brightness={brightness:.0f} — consistent "
                            f"with myopic pale fundus",
                "literature_ref": "META-PM: Tessellated and atrophic "
                                  "myopic fundi appear brighter due to "
                                  "RPE and choroidal thinning",
            })

    # ---------------------------------------------------------------- #
    #  HYPERTENSION (Keith-Wagener-Barker classification)               #
    # ---------------------------------------------------------------- #
    if predictions.get("Hypertension", False):
        if hemorrhage_count > 0 and exudate_count == 0:
            flags.append({
                "disease": "Hypertension",
                "flag_type": "supporting",
                "severity": "moderate",
                "evidence": f"{hemorrhage_count} hemorrhage(s) without "
                            f"exudates — pattern consistent with "
                            f"hypertensive retinopathy",
                "literature_ref": "Keith-Wagener-Barker Grade II: "
                                  "Flame-shaped hemorrhages are "
                                  "characteristic of hypertensive "
                                  "retinopathy",
            })
        elif hemorrhage_count == 0:
            flags.append({
                "disease": "Hypertension",
                "flag_type": "contradicting",
                "severity": "moderate",
                "evidence": "No hemorrhages detected — key clinical "
                            "sign absent for hypertensive retinopathy",
                "literature_ref": "Keith-Wagener-Barker: Retinal "
                                  "hemorrhages are a hallmark of "
                                  "hypertensive retinopathy Grade II+",
            })

        # Vessel narrowing from hypertension features
        av_ratio = hypertension_features.get("av_ratio", 0.0)
        if av_ratio > 0 and av_ratio < 0.6:
            flags.append({
                "disease": "Hypertension",
                "flag_type": "supporting",
                "severity": "high",
                "evidence": f"AV ratio={av_ratio:.2f} < 0.6 — significant "
                            f"arteriolar narrowing detected",
                "literature_ref": "Wong & Mitchell (2004), Lancet: Reduced "
                                  "AV ratio is a hallmark of hypertensive "
                                  "retinopathy; normal AV ratio ~0.67-0.75",
            })

    # ---------------------------------------------------------------- #
    #  CATARACT (WHO blindness survey protocol, Chylack LOCS III)       #
    # ---------------------------------------------------------------- #
    if predictions.get("Cataract", False):
        blur = cataract_features.get("blur_score", 0.0)
        contrast = cataract_features.get("contrast_ratio", 0.0)
        haze = cataract_features.get("haze_index", 0.0)

        if blur < 50 and haze > 0.3:
            flags.append({
                "disease": "Cataract",
                "flag_type": "supporting",
                "severity": "moderate",
                "evidence": f"Low image sharpness (blur={blur:.0f}) with "
                            f"elevated haze (haze_index={haze:.2f}) — "
                            f"consistent with lens opacity",
                "literature_ref": "Chylack et al. (1993) LOCS III: Lens "
                                  "opacities cause progressive image "
                                  "degradation and contrast loss; "
                                  "WHO Blindness Survey Protocol",
            })
        elif blur > 200:
            flags.append({
                "disease": "Cataract",
                "flag_type": "contradicting",
                "severity": "low",
                "evidence": f"Image is sharp (blur_score={blur:.0f}) — "
                            f"less consistent with significant cataract",
                "literature_ref": "LOCS III: Advanced cataract causes "
                                  "marked image degradation detectable "
                                  "on fundus photography",
            })

    # ---------------------------------------------------------------- #
    #  AMD (AREDS Study, AREDS2, Ferris et al. 2013)                    #
    # ---------------------------------------------------------------- #
    if predictions.get("Age-related Macular Degeneration", False):
        drusen = amd_features.get("drusen_count", 0)
        macular_irregularity = amd_features.get("macular_irregularity", 0.0)

        if drusen > 3:
            flags.append({
                "disease": "Age-related Macular Degeneration",
                "flag_type": "supporting",
                "severity": "high",
                "evidence": f"{drusen} drusen-like deposits detected — "
                            f"consistent with AMD pathology",
                "literature_ref": "AREDS Report No. 8: Presence of large "
                                  "drusen (>125 μm) is a primary risk "
                                  "factor for AMD progression; "
                                  "Ferris et al. (2013) Beckman classification",
            })
        elif drusen > 0:
            flags.append({
                "disease": "Age-related Macular Degeneration",
                "flag_type": "supporting",
                "severity": "moderate",
                "evidence": f"{drusen} drusen-like deposit(s) detected",
                "literature_ref": "AREDS: Small drusen (<63 μm) are "
                                  "common with aging; intermediate drusen "
                                  "indicate early AMD per Beckman",
            })

        if macular_irregularity > 0.3:
            flags.append({
                "disease": "Age-related Macular Degeneration",
                "flag_type": "supporting",
                "severity": "moderate",
                "evidence": f"Macular irregularity={macular_irregularity:.2f} "
                            f"— textural changes in macular region",
                "literature_ref": "AREDS2: Pigmentary changes and RPE "
                                  "irregularities in the macula are "
                                  "associated with AMD progression",
            })

    return unchanged, flags


if __name__ == "__main__":
    # Quick demo
    preds = {"Diabetes": True, "Glaucoma": True, "Myopia": False,
             "Hypertension": False, "Normal": False}
    confs = {"Diabetes": 0.78, "Glaucoma": 0.65, "Myopia": 0.12,
             "Hypertension": 0.10, "Normal": 0.05}
    feats = {
        "dr": {"exudate_count": 0, "hemorrhage_count": 2,
               "microaneurysm_count": 3, "lesion_area": 0.001},
        "glaucoma": {"cdr": 0.35},
        "myopia": {"brightness": 120, "texture_variance": 800,
                   "vessel_visibility": 12.3, "edge_density": 0.08},
        "cataract": {"blur_score": 150, "contrast_ratio": 0.6,
                     "haze_index": 0.1},
        "amd": {"drusen_count": 0, "macular_irregularity": 0.1},
        "hypertension": {"av_ratio": 0.7, "vessel_tortuosity": 0.05},
    }
    unchanged, flags = apply_validation_rules(preds, confs, feats)
    print("Confidences (unchanged):", unchanged)
    print()
    for f in flags:
        print(f"  [{f['flag_type'].upper()}] {f['disease']}: {f['evidence']}")
        print(f"  📚 {f['literature_ref']}")
        print()
