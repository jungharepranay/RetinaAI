"""
pdf_report.py
-------------
Generate a structured PDF clinical screening report for RetinAI.

Uses reportlab to produce a professional report containing:
  - Patient details
  - Screening summary with risk level
  - Key findings
  - Recommendations
  - Questionnaire insights (if available)
  - Grad-CAM heatmap (optional)
  - Disclaimer

Returns PDF bytes for streaming download — no server-side file storage.
"""

import io
import os
from datetime import datetime

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, HRFlowable, KeepTogether,
    )
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ─── Color Palette ─── #
COLOR_PRIMARY = colors.HexColor("#0a1628")
COLOR_ACCENT = colors.HexColor("#00b894")
COLOR_DANGER = colors.HexColor("#e74c3c")
COLOR_WARNING = colors.HexColor("#f39c12")
COLOR_SUCCESS = colors.HexColor("#27ae60")
COLOR_MUTED = colors.HexColor("#7f8c8d")
COLOR_LIGHT_BG = colors.HexColor("#f8f9fa")
COLOR_BORDER = colors.HexColor("#dee2e6")


def _get_risk_color(urgency: str):
    """Return a color based on risk/urgency level."""
    if urgency == "urgent":
        return COLOR_DANGER
    elif urgency == "semi-urgent":
        return COLOR_WARNING
    return COLOR_SUCCESS


def _get_styles():
    """Build custom paragraph styles for the report."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=COLOR_PRIMARY,
        spaceAfter=4,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    ))

    styles.add(ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=COLOR_MUTED,
        spaceAfter=16,
        alignment=TA_CENTER,
    ))

    styles.add(ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=COLOR_PRIMARY,
        spaceBefore=16,
        spaceAfter=8,
        fontName="Helvetica-Bold",
        borderWidth=0,
        borderPadding=0,
    ))

    styles.add(ParagraphStyle(
        "BodyText2",
        parent=styles["Normal"],
        fontSize=10,
        textColor=COLOR_PRIMARY,
        spaceAfter=6,
        leading=14,
    ))

    styles.add(ParagraphStyle(
        "SmallMuted",
        parent=styles["Normal"],
        fontSize=8,
        textColor=COLOR_MUTED,
        spaceAfter=4,
    ))

    styles.add(ParagraphStyle(
        "DisclaimerText",
        parent=styles["Normal"],
        fontSize=8,
        textColor=COLOR_MUTED,
        spaceBefore=12,
        spaceAfter=4,
        leading=11,
        alignment=TA_CENTER,
    ))

    styles.add(ParagraphStyle(
        "FindingHigh",
        parent=styles["Normal"],
        fontSize=10,
        textColor=COLOR_DANGER,
        fontName="Helvetica-Bold",
        spaceAfter=2,
    ))

    styles.add(ParagraphStyle(
        "FindingBorderline",
        parent=styles["Normal"],
        fontSize=10,
        textColor=COLOR_WARNING,
        fontName="Helvetica-Bold",
        spaceAfter=2,
    ))

    styles.add(ParagraphStyle(
        "FindingNormal",
        parent=styles["Normal"],
        fontSize=10,
        textColor=COLOR_SUCCESS,
        fontName="Helvetica-Bold",
        spaceAfter=2,
    ))

    return styles


def generate_pdf_report(
    assessment_data: dict,
    patient_name: str = "",
    patient_age: str = "",
    questionnaire_data: dict = None,
    gradcam_path: str = None,
) -> bytes:
    """
    Generate a PDF clinical screening report.

    Parameters
    ----------
    assessment_data : dict
        Full prediction output from /predict/initial.
    patient_name : str
        Patient name for the report header.
    patient_age : str
        Patient age.
    questionnaire_data : dict or None
        Post-prediction questionnaire responses and consistency result.
    gradcam_path : str or None
        Absolute path to Grad-CAM heatmap image.

    Returns
    -------
    bytes
        PDF file content.
    """
    if not HAS_REPORTLAB:
        raise ImportError(
            "reportlab is not installed. Run: pip install reportlab"
        )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )

    styles = _get_styles()
    elements = []

    # ─── 1. Header ─── #
    elements.append(Paragraph("RetinAI Clinical Screening Report", styles["ReportTitle"]))
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    elements.append(Paragraph(
        f"Generated on {report_date} • AI-Assisted Screening",
        styles["ReportSubtitle"],
    ))
    elements.append(HRFlowable(
        width="100%", thickness=1, color=COLOR_BORDER,
        spaceAfter=12, spaceBefore=4,
    ))

    # ─── 2. Patient Details ─── #
    assessment = assessment_data.get("clinical_assessment", {})
    clinical_context = assessment_data.get("patient_context", {})
    patient_summary = (
        clinical_context.get("patient_summary")
        or assessment.get("patient_summary")
        or ""
    )

    elements.append(Paragraph("Patient Information", styles["SectionHeader"]))

    patient_table_data = []
    if patient_name:
        patient_table_data.append(["Name:", patient_name])
    if patient_age:
        patient_table_data.append(["Age:", str(patient_age)])
    if patient_summary:
        patient_table_data.append(["Profile:", patient_summary])

    if patient_table_data:
        t = Table(patient_table_data, colWidths=[80, 400])
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR", (0, 0), (0, -1), COLOR_MUTED),
            ("TEXTCOLOR", (1, 0), (1, -1), COLOR_PRIMARY),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(t)
    else:
        elements.append(Paragraph(
            "No patient information provided.",
            styles["SmallMuted"],
        ))

    elements.append(Spacer(1, 8))

    # ─── 3. Screening Summary ─── #
    urgency = assessment.get("urgency", "routine")

    # Use summary_status (derived from risk levels) instead of urgency
    # This fixes the bug where summary said NORMAL even when diseases detected
    summary_status = assessment.get("summary_status", "")
    if not summary_status:
        # Fallback: derive from urgency for backward compatibility
        summary_status = {
            "urgent": "URGENT \u2014 Immediate clinical evaluation recommended",
            "semi-urgent": "FOLLOW-UP \u2014 Clinical evaluation recommended",
            "routine": "NORMAL \u2014 No significant abnormalities detected",
        }.get(urgency, "ROUTINE")

    # Pick banner color from summary content
    if "ABNORMAL" in summary_status or "URGENT" in summary_status:
        risk_color = COLOR_DANGER
    elif "MONITOR" in summary_status or "FOLLOW" in summary_status:
        risk_color = COLOR_WARNING
    else:
        risk_color = COLOR_SUCCESS

    elements.append(Paragraph("Screening Summary", styles["SectionHeader"]))

    # Risk level banner
    risk_table = Table(
        [[Paragraph(f"<b>{summary_status}</b>", ParagraphStyle(
            "RiskText", fontSize=12, textColor=colors.white,
            alignment=TA_CENTER, fontName="Helvetica-Bold",
        ))]],
        colWidths=[480],
    )
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), risk_color),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    elements.append(risk_table)
    elements.append(Spacer(1, 8))

    # Final decision
    final_decision = assessment_data.get("final_decision", "")
    if final_decision:
        elements.append(Paragraph(final_decision, styles["BodyText2"]))

    elements.append(Spacer(1, 8))

    # ─── 4. Key Findings Table ─── #
    findings = assessment.get("key_findings", [])

    # Confidence lookup: try multiple keys for robustness
    confidences = assessment_data.get("confidence",
                  assessment_data.get("ensemble_confidence", {}))

    if findings:
        elements.append(Paragraph("Key Findings", styles["SectionHeader"]))

        table_data = [["Condition", "Risk Level", "Confidence"]]
        for f in findings:
            disease = f.get("disease", "Unknown")
            risk_level = f.get("risk_level", "Low Risk")

            # Match confidence: prefer direct lookup, fallback to finding's own probability
            raw_conf = confidences.get(disease, f.get("probability", 0.0))
            conf_pct = round(raw_conf * 100, 1)
            conf_str = f"{conf_pct}%"

            table_data.append([disease, risk_level, conf_str])

        t = Table(table_data, colWidths=[200, 140, 140])
        table_style = [
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
        ]

        # Highlight highest-risk row and color-code risk badges
        max_prob = 0.0
        max_row_idx = -1
        for i, row in enumerate(table_data[1:]):
            r_level = row[1]
            if r_level == "High Risk":
                table_style.append(("BACKGROUND", (1, i+1), (1, i+1), COLOR_DANGER))
                table_style.append(("TEXTCOLOR", (1, i+1), (1, i+1), colors.white))
            elif r_level == "Borderline":
                table_style.append(("BACKGROUND", (1, i+1), (1, i+1), COLOR_WARNING))
                table_style.append(("TEXTCOLOR", (1, i+1), (1, i+1), colors.white))
            else:
                table_style.append(("BACKGROUND", (1, i+1), (1, i+1), COLOR_SUCCESS))
                table_style.append(("TEXTCOLOR", (1, i+1), (1, i+1), colors.white))

            # Track highest-probability finding for highlighting
            finding_prob = findings[i].get("probability", 0.0)
            if finding_prob > max_prob:
                max_prob = finding_prob
                max_row_idx = i + 1

        # Bold the highest-risk row
        if max_row_idx > 0:
            table_style.append(("FONTNAME", (0, max_row_idx), (-1, max_row_idx), "Helvetica-Bold"))
            table_style.append(("BACKGROUND", (0, max_row_idx), (0, max_row_idx),
                                colors.HexColor("#fff3cd")))

        t.setStyle(TableStyle(table_style))
        elements.append(t)
        elements.append(Spacer(1, 16))

    # ─── 4b. All Predictions Table (comprehensive, all 8 diseases) ─── #
    all_findings = assessment.get("all_findings", [])
    if all_findings:
        elements.append(Paragraph("All Predictions", styles["SectionHeader"]))
        elements.append(Paragraph(
            "Complete screening results for all evaluated conditions, sorted by confidence.",
            styles["SmallMuted"],
        ))

        # Sort by probability descending
        sorted_findings = sorted(all_findings, key=lambda x: x.get("probability", 0), reverse=True)

        all_table_data = [["Condition", "Risk Level", "Confidence"]]
        for af in sorted_findings:
            disease = af.get("disease", "Unknown")
            risk_level = af.get("risk_level", "Low Risk")
            raw_conf = confidences.get(disease, af.get("probability", 0.0))
            conf_pct = round(raw_conf * 100, 1)
            conf_str = f"{conf_pct}%"
            all_table_data.append([disease, risk_level, conf_str])

        at = Table(all_table_data, colWidths=[200, 140, 140])
        all_table_style = [
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
        ]

        for i, row in enumerate(all_table_data[1:]):
            r_level = row[1]
            if r_level == "High Risk":
                all_table_style.append(("BACKGROUND", (1, i+1), (1, i+1), COLOR_DANGER))
                all_table_style.append(("TEXTCOLOR", (1, i+1), (1, i+1), colors.white))
            elif r_level == "Borderline":
                all_table_style.append(("BACKGROUND", (1, i+1), (1, i+1), COLOR_WARNING))
                all_table_style.append(("TEXTCOLOR", (1, i+1), (1, i+1), colors.white))
            else:
                all_table_style.append(("BACKGROUND", (1, i+1), (1, i+1), COLOR_SUCCESS))
                all_table_style.append(("TEXTCOLOR", (1, i+1), (1, i+1), colors.white))
            # Alternating row backgrounds for readability
            if i % 2 == 1:
                all_table_style.append(("BACKGROUND", (0, i+1), (0, i+1), COLOR_LIGHT_BG))
                all_table_style.append(("BACKGROUND", (2, i+1), (2, i+1), COLOR_LIGHT_BG))

        at.setStyle(TableStyle(all_table_style))
        elements.append(at)
        elements.append(Spacer(1, 16))
    elif not findings:
        elements.append(Paragraph("Key Findings", styles["SectionHeader"]))
        elements.append(Paragraph(
            "\u2713 No significant retinal abnormalities detected in this screening.",
            styles["BodyText2"],
        ))
        elements.append(Spacer(1, 16))

    # ─── 5. Clinical Interpretation ─── #
    elements.append(Paragraph("Clinical Interpretation", styles["SectionHeader"]))
    if findings:
        for f in findings:
            disease = f.get("disease", "Unknown")
            desc = f.get("description", "")
            if desc:
                elements.append(Paragraph(f"<b>{disease}:</b> {desc}", styles["BodyText2"]))
    else:
         elements.append(Paragraph("The screening indicates a normal retinal profile with no immediate signs of pathology.", styles["BodyText2"]))
    elements.append(Spacer(1, 16))

    # ─── 6. Recommendations ─── #
    recommendations = assessment.get("recommendations", [])
    if recommendations:
        elements.append(Paragraph("Recommendations", styles["SectionHeader"]))
        for rec in recommendations:
            elements.append(Paragraph(f"• {rec}", styles["BodyText2"]))
        elements.append(Spacer(1, 16))

    # ─── 7. Symptom Assessment ─── #
    if questionnaire_data:
        elements.append(Paragraph("Symptom Assessment", styles["SectionHeader"]))

        consistency = questionnaire_data.get("consistency_statement", "")
        case_type = questionnaire_data.get("case_type", "minimal")
        selected_symptoms = questionnaire_data.get("selected_symptoms", [])

        # Color code the statement based on case type
        color_hex = "#27ae60" if case_type == "consistent" else "#f39c12" if case_type == "partial" else "#3b82f6"
        if consistency:
            elements.append(Paragraph(f'<font color="{color_hex}"><b>{consistency}</b></font>', styles["BodyText2"]))

        if selected_symptoms:
            elements.append(Spacer(1, 4))
            for sym in selected_symptoms:
                import re
                clean_sym = re.sub(r'<[^>]+>', '', sym).strip()
                elements.append(Paragraph(f"• {clean_sym}", styles["BodyText2"]))
        
        elements.append(Spacer(1, 16))

    # ─── 8. Grad-CAM Image ─── #
    if gradcam_path:
        # Resolve relative paths
        if not os.path.isabs(gradcam_path):
            abs_path = os.path.join(PROJECT_ROOT, gradcam_path.lstrip("/"))
        else:
            abs_path = gradcam_path

        if os.path.isfile(abs_path):
            try:
                elements.append(Paragraph(
                    "AI Attention Heatmap (Grad-CAM)", styles["SectionHeader"],
                ))
                img = RLImage(abs_path, width=3.5 * inch, height=3.5 * inch)
                img.hAlign = "CENTER"
                elements.append(img)
                elements.append(Paragraph(
                    "Highlighted regions indicate areas the AI model focused on "
                    "during prediction.",
                    styles["SmallMuted"],
                ))
            except Exception:
                pass  # Skip if image can't be loaded

    # ─── 9. Disclaimer ─── #
    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(
        width="100%", thickness=0.5, color=COLOR_BORDER,
        spaceAfter=8, spaceBefore=8,
    ))
    elements.append(Paragraph(
        "<b>Disclaimer:</b> This report is generated by an AI-based screening "
        "system and should be reviewed by a qualified eye specialist.",
        styles["DisclaimerText"],
    ))
    elements.append(Paragraph(
        "RetinAI Clinical Screening Platform \u2022 Created By Group_15",
        styles["DisclaimerText"],
    ))

    # Build PDF
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


# ─── CLI test ─── #
if __name__ == "__main__":
    # Quick test — generate a sample report
    sample_data = {
        "clinical_assessment": {
            "urgency": "semi-urgent",
            "key_findings": [
                {
                    "disease": "Diabetic Retinopathy",
                    "risk_level": "High Risk",
                    "description": "Signs of moderate diabetic retinopathy detected.",
                    "recommendation": "Refer to ophthalmologist for detailed examination.",
                },
            ],
            "recommendations": [
                "Schedule an appointment with an ophthalmologist.",
                "Continue monitoring blood sugar levels.",
            ],
        },
        "final_decision": "Findings: Diabetic Retinopathy (High Risk). Clinical follow-up recommended.",
    }

    pdf = generate_pdf_report(
        sample_data,
        patient_name="John Doe",
        patient_age="58",
    )

    out_path = os.path.join(PROJECT_ROOT, "reports", "test_report.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(pdf)
    print(f"Test report saved to {out_path} ({len(pdf)} bytes)")
