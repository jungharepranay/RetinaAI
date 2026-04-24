"""
run_pipeline.py
---------------
CLI test script for the full RetinAI hybrid clinical pipeline.

Usage::

    python run_pipeline.py <image_path> [--threshold 0.5]

Runs the complete pipeline:
  1. Image quality check
  2. Clinical feature extraction (DR, Glaucoma, Myopia)
  3. CNN prediction (EfficientNet-B3)
  4. Rule-based clinical validation
  5. Grad-CAM explainability
  6. Clinical question generation
  7. Final structured output
"""

import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="RetinAI — Full Hybrid Clinical Pipeline"
    )
    parser.add_argument(
        "image", type=str,
        help="Path to a retinal fundus image."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification threshold (default: 0.5)."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for Grad-CAM heatmaps (default: reports/gradcam/)."
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: file not found — {args.image}")
        sys.exit(1)

    from src.predict import predict_initial

    print("=" * 60)
    print("  RetinAI — Hybrid Clinical Pipeline")
    print("=" * 60)
    print(f"\nImage:     {args.image}")
    print(f"Threshold: {args.threshold}")
    print()

    result = predict_initial(
        args.image,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )

    # ---- Quality Check ---- #
    qc = result["quality_check"]
    print("━" * 50)
    print("📋 IMAGE QUALITY CHECK")
    print("━" * 50)
    if qc.get("is_valid"):
        print(f"  ✅ PASSED")
        print(f"     Blur score:  {qc.get('blur_score', 'N/A')}")
        print(f"     Brightness:  {qc.get('brightness_mean', 'N/A')}")
        print(f"     Contrast:    {qc.get('contrast_std', 'N/A')}")
    else:
        print(f"  ❌ FAILED — {qc.get('reason', 'unknown')}")
        print(f"\n{result['final_decision']}")
        return

    # ---- CNN Predictions ---- #
    print(f"\n{'━' * 50}")
    print("🧠 CNN PREDICTIONS")
    print("━" * 50)
    for disease, conf in result["confidence"].items():
        flag = result["predictions"].get(disease, False)
        marker = " ◀ DETECTED" if flag else ""
        print(f"  {disease:40s}  {conf:.4f}{marker}")

    # ---- Clinical Features ---- #
    print(f"\n{'━' * 50}")
    print("🔬 CLINICAL FEATURES")
    print("━" * 50)
    feats = result["clinical_features"]

    dr = feats.get("dr", {})
    print(f"  DR:       exudates={dr.get('exudate_count', 0)}, "
          f"hemorrhages={dr.get('hemorrhage_count', 0)}, "
          f"microaneurysms={dr.get('microaneurysm_count', 0)}, "
          f"lesion_area={dr.get('lesion_area', 0):.6f}")

    gl = feats.get("glaucoma", {})
    print(f"  Glaucoma: CDR={gl.get('cdr', 0):.4f}")

    my = feats.get("myopia", {})
    print(f"  Myopia:   brightness={my.get('brightness', 0):.1f}, "
          f"texture_var={my.get('texture_variance', 0):.1f}, "
          f"vessel_vis={my.get('vessel_visibility', 0):.1f}")

    # ---- Validation Rules ---- #
    print(f"\n{'━' * 50}")
    print("⚖️  VALIDATION RULES")
    print("━" * 50)
    rules = result.get("validation_rules_applied", [])
    if rules:
        for r in rules:
            if isinstance(r, dict):
                print(f"  • {r.get('rule', str(r))}")
                ref = r.get('literature_ref', '')
                if ref:
                    print(f"    📚 {ref}")
            else:
                print(f"  • {r}")
    else:
        print("  (no rules triggered)")

    # ---- Adjusted Confidence ---- #
    adj = result.get("adjusted_confidence", {})
    changed = {k: v for k, v in adj.items()
               if abs(v - result["confidence"].get(k, v)) > 0.001}
    if changed:
        print(f"\n  Adjusted confidence:")
        for disease, conf in changed.items():
            orig = result["confidence"].get(disease, conf)
            print(f"    {disease}: {orig:.4f} → {conf:.4f}")

    # ---- Grad-CAM ---- #
    print(f"\n{'━' * 50}")
    print("🔥 GRAD-CAM HEATMAPS")
    print("━" * 50)
    gcpaths = result.get("gradcam_paths", {})
    if gcpaths:
        for disease, path in gcpaths.items():
            print(f"  {disease}: {path}")
    else:
        print("  (no heatmaps generated)")

    # ---- Clinical Questions ---- #
    print(f"\n{'━' * 50}")
    print("🩺 CLINICAL QUESTIONS")
    print("━" * 50)
    questions = result.get("questions", {}).get("questions", [])
    if questions:
        for q in questions:
            print(f"  {q['id'].upper()}. [{q.get('disease', '')}] {q['text']}")
    else:
        print("  (no questions generated)")

    # ---- Final Decision ---- #
    print(f"\n{'━' * 50}")
    print("📋 FINAL DECISION")
    print("━" * 50)
    print(f"  {result['final_decision']}")
    print()

    # ---- Full JSON ---- #
    json_path = os.path.join(
        PROJECT_ROOT, "reports", "last_pipeline_result.json"
    )
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    serializable = dict(result)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"Full result saved to: {json_path}")


if __name__ == "__main__":
    main()
