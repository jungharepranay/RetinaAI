"""
train.py
--------
Two-phase training pipeline for the retinal disease classifier.

Phase 1 — frozen backbone, train classifier head (15 epochs)
Phase 2 — unfreeze top layers, fine-tune (35 epochs)

Improvements over baseline:
  • Dynamic class-weight computation for imbalanced labels
  • Mixed-precision training when GPU is available
  • Dataset statistics logging for diagnosing class imbalance
  • Extended training duration (50 total epochs)

Usage::

    python -m src.train              # uses default paths
    python src/train.py              # also works
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ---- Ensure project root is on sys.path ---- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_dataset, DISEASE_COLUMNS
from src.dataset_builder import build_dataset
from src.model import build_model, unfreeze_top_layers


# ----- Reproducibility ----- #
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----- Paths ----- #
CSV_PATH = os.path.join(PROJECT_ROOT, "dataset", "odir", "full_df.csv")
IMAGE_DIR = os.path.join(PROJECT_ROOT, "dataset", "odir", "preprocessed_images")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "retinal_model.keras")

# ----- Hyper-parameters ----- #
BATCH_SIZE = 32
PHASE1_EPOCHS = 15   # increased from 10
PHASE2_EPOCHS = 35   # increased from 20  (total = 50)
VALIDATION_SPLIT = 0.2

# Set to 'focal' to use Focal Loss instead of BinaryCrossentropy
LOSS_TYPE = 'bce'


# ------------------------------------------------------------------ #
#  Mixed precision — enable only when a GPU is available
# ------------------------------------------------------------------ #
def _enable_mixed_precision():
    """Enable mixed-precision (float16) training if a GPU is detected."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print(f"[train] Mixed precision enabled on {len(gpus)} GPU(s).")
    else:
        print("[train] No GPU detected — using default float32 policy.")


# ------------------------------------------------------------------ #
#  Class weights
# ------------------------------------------------------------------ #
def compute_class_weights(labels: np.ndarray) -> dict:
    """
    Compute per-class weights to counter label imbalance.

    For each label column:
        weight = num_negative / (num_positive + eps)

    Weights are normalised so the mean weight equals 1.

    Returns a dict mapping class index → weight, suitable for
    ``model.fit(class_weight=...)``.
    """
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    class_weights = neg_counts / (pos_counts + 1e-6)

    # Normalise so mean weight = 1
    class_weights = class_weights / class_weights.mean()

    print("\n[train] Class weights (normalised):")
    for i, col in enumerate(DISEASE_COLUMNS):
        print(f"  {col} ({DISEASE_COLUMNS[i]:>1}):  "
              f"pos={int(pos_counts[i]):>5}  neg={int(neg_counts[i]):>5}  "
              f"weight={class_weights[i]:.4f}")

    return {i: float(class_weights[i]) for i in range(len(class_weights))}


# ------------------------------------------------------------------ #
#  Callbacks
# ------------------------------------------------------------------ #
def get_callbacks(phase: str):
    """Create standard callbacks."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=5, mode='max',
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc', factor=0.5, patience=3,
            mode='max', min_lr=1e-7, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH, monitor='val_auc', mode='max',
            save_best_only=True, verbose=1,
        ),
    ]


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    # Enable mixed precision if GPU is available
    _enable_mixed_precision()

    # Ensure model directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # ----- Load data ----- #
    print("=" * 60)
    print("Loading dataset …")
    image_paths, labels, _ = load_dataset(CSV_PATH, IMAGE_DIR)

    # ----- Dataset statistics ----- #
    print(f"\nDataset size: {len(image_paths)}")
    print(f"Label distribution: {labels.sum(axis=0).astype(int)}")
    print(f"Label columns:      {DISEASE_COLUMNS}")

    # ----- Compute class weights ----- #
    class_weights = compute_class_weights(labels)

    # ----- Train / Validation split ----- #
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels,
        test_size=VALIDATION_SPLIT,
        random_state=SEED,
    )
    print(f"\nTrain: {len(X_train)}  |  Val: {len(X_val)}")

    # ----- Build tf.data pipelines ----- #
    train_ds = build_dataset(X_train, y_train, BATCH_SIZE,
                             shuffle=True, augment=True)
    val_ds = build_dataset(X_val, y_val, BATCH_SIZE,
                           shuffle=False, augment=False)

    # ----- Build model ----- #
    model = build_model(loss_type=LOSS_TYPE)

    # ======================= Phase 1 ======================= #
    print("\n" + "=" * 60)
    print(f"Phase 1: Training classifier head (backbone frozen) — {PHASE1_EPOCHS} epochs")
    print("=" * 60)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks("phase1"),
    )

    # ======================= Phase 2 ======================= #
    print("\n" + "=" * 60)
    print(f"Phase 2: Fine-tuning top layers — {PHASE2_EPOCHS} epochs")
    print("=" * 60)
    model = unfreeze_top_layers(model, loss_type=LOSS_TYPE)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks("phase2"),
    )

    print(f"\n✓ Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
