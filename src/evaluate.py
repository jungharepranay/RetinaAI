"""
evaluate.py
-----------
Evaluate the trained retinal disease classifier on a test split.

Computes:  Precision, Recall, F1-Score (macro & micro), ROC-AUC, Hamming Loss
Saves:     classification report, metrics summary, and confusion matrices
           to ``reports/``.

NOTE: Multi-label medical datasets typically require a lower prediction
threshold (e.g. 0.3) to improve recall — missing a disease is far worse
than a false positive in clinical screening.

Usage::

    python -m src.evaluate
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    multilabel_confusion_matrix,
    roc_auc_score,
    hamming_loss,
    precision_score,
    recall_score,
    f1_score,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_dataset, DISEASE_COLUMNS
from src.dataset_builder import build_dataset

# ----- Paths ----- #
CSV_PATH = os.path.join(PROJECT_ROOT, "dataset", "odir", "full_df.csv")
IMAGE_DIR = os.path.join(PROJECT_ROOT, "dataset", "odir", "preprocessed_images")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "retinal_model.keras")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

SEED = 42
BATCH_SIZE = 32

# A lower threshold improves recall for rare diseases.
# In multi-label medical imaging, missing a condition (false negative) is
# clinically more dangerous than a false alarm (false positive), so we
# default to 0.3 instead of the conventional 0.5.
PREDICTION_THRESHOLD = 0.3


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load data — use same split as training
    image_paths, labels, _ = load_dataset(CSV_PATH, IMAGE_DIR)
    _, X_test, _, y_test = train_test_split(
        image_paths, labels, test_size=0.2, random_state=SEED,
    )

    # Build dataset
    test_ds = build_dataset(X_test, y_test, BATCH_SIZE,
                            shuffle=False, augment=False)

    # Load model
    print(f"Loading model from {MODEL_PATH} …")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Predictions
    y_pred_proba = model.predict(test_ds)
    y_pred = (y_pred_proba >= PREDICTION_THRESHOLD).astype(np.int32)
    y_true = y_test.astype(np.int32)

    print(f"\n[evaluate] Using prediction threshold = {PREDICTION_THRESHOLD}")

    # ----- Classification Report ----- #
    report = classification_report(
        y_true, y_pred, target_names=DISEASE_COLUMNS, zero_division=0,
    )
    print("\n===== Classification Report =====")
    print(report)

    # ----- Macro metrics ----- #
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # ----- Micro metrics ----- #
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

    # ----- ROC-AUC & Hamming Loss ----- #
    try:
        roc_auc_macro = roc_auc_score(y_true, y_pred_proba, average='macro')
        roc_auc_micro = roc_auc_score(y_true, y_pred_proba, average='micro')
    except ValueError:
        # Handles the edge case where a class has no positive samples in test
        roc_auc_macro = 0.0
        roc_auc_micro = 0.0

    h_loss = hamming_loss(y_true, y_pred)

    summary = {
        'prediction_threshold': PREDICTION_THRESHOLD,
        'precision_macro': round(float(precision_macro), 4),
        'recall_macro': round(float(recall_macro), 4),
        'f1_macro': round(float(f1_macro), 4),
        'precision_micro': round(float(precision_micro), 4),
        'recall_micro': round(float(recall_micro), 4),
        'f1_micro': round(float(f1_micro), 4),
        'roc_auc_macro': round(float(roc_auc_macro), 4),
        'roc_auc_micro': round(float(roc_auc_micro), 4),
        'hamming_loss': round(float(h_loss), 4),
    }

    print("\n===== Summary Metrics =====")
    for k, v in summary.items():
        print(f"  {k:25s}: {v}")

    # Confusion matrices
    cm = multilabel_confusion_matrix(y_true, y_pred)

    # ----- Save reports ----- #
    report_path = os.path.join(REPORTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Prediction threshold: {PREDICTION_THRESHOLD}\n\n")
        f.write(report)

    summary_path = os.path.join(REPORTS_DIR, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    np.save(os.path.join(REPORTS_DIR, "confusion_matrices.npy"), cm)

    print(f"\n✓ Reports saved to {REPORTS_DIR}/")
    print(f"  • {report_path}")
    print(f"  • {summary_path}")


if __name__ == "__main__":
    main()
