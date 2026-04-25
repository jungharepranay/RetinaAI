"""
ensemble_model.py
-----------------
Production ensemble inference for retinal disease classification.

Loads multiple trained models (EfficientNet-B3 + DenseNet-121 + ConvNeXt-Tiny)
and combines predictions via weighted soft-voting.

Usage::

    from src.ensemble_model import load_ensemble, predict_ensemble

    ensemble = load_ensemble("models/ensemble_config.json")
    detected, probabilities = predict_ensemble("path/to/fundus.jpg", ensemble)

██  SAFETY  ██  Model predictions are raw ensemble outputs.
No clinical interpretation modifies these probabilities.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import timm
import cv2

from src.data_loader import DISEASE_COLUMNS, DISEASE_NAMES

# ────────────────────────── Constants ────────────────────────── #

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 224
NUM_CLASSES = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ────────────────────────── Model Class ────────────────────────── #

class MultiLabelClassifier(nn.Module):
    """Generic multi-label classifier using any timm backbone.

    Same architecture used during training — must match exactly.
    """

    def __init__(self, backbone_name: str, num_classes: int = 8,
                 dropout: float = 0.3, pretrained: bool = False):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=0.0,
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits


# ────────────────────────── Preprocessing ────────────────────────── #

def apply_clahe_lab(image_rgb: np.ndarray,
                    clip_limit: float = 2.0,
                    tile_size: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE on LAB luminance channel."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_enhanced = clahe.apply(l_ch)
    lab_out = cv2.merge([l_enhanced, a_ch, b_ch])
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)


def preprocess_image(image_path: str, target_size: int = IMG_SIZE,
                     use_clahe: bool = True) -> torch.Tensor:
    """Load, CLAHE-enhance, resize, normalise → batch tensor (1, 3, H, W)."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if use_clahe:
        img = apply_clahe_lab(img)
    img = cv2.resize(img, (target_size, target_size))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def preprocess_image_from_array(image_rgb: np.ndarray, target_size: int = IMG_SIZE,
                                use_clahe: bool = True) -> torch.Tensor:
    """Same as preprocess_image but accepts an in-memory RGB uint8 array."""
    img = image_rgb.copy()
    if use_clahe:
        img = apply_clahe_lab(img)
    img = cv2.resize(img, (target_size, target_size))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


# ────────────────────────── Ensemble Loading ────────────────────────── #

class EnsembleModel:
    """Holds multiple loaded models and ensemble configuration.

    Attributes
    ----------
    models : list[nn.Module]
        Loaded models in eval mode.
    model_names : list[str]
        Human-readable model names.
    weights : dict
        Model name → weight (for weighted averaging).
    per_class_weights : dict or None
        Disease column → {model_name: weight} (for per-class weighting).
    thresholds : dict
        Disease column → optimal threshold.
    ensemble_method : str
        "Equal Average", "AUC-Weighted", or "Per-Class AUC".
    use_clahe : bool
    img_size : int
    """

    def __init__(self):
        self.models = []
        self.model_names = []
        self.weights = {}
        self.per_class_weights = None
        self.thresholds = {}
        self.ensemble_method = "Equal Average"
        self.use_clahe = True
        self.img_size = IMG_SIZE


def load_ensemble(config_path: str, device: torch.device = DEVICE) -> EnsembleModel:
    """Load ensemble from a config JSON and individual model checkpoints.

    Parameters
    ----------
    config_path : str
        Path to `ensemble_config.json` produced by training.
    device : torch.device

    Returns
    -------
    EnsembleModel with all models loaded in eval mode.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Ensemble config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    ensemble = EnsembleModel()
    ensemble.ensemble_method = config.get('ensemble_method', 'Equal Average')
    ensemble.thresholds = config.get('ensemble_optimal_thresholds', {c: 0.5 for c in DISEASE_COLUMNS})
    ensemble.use_clahe = config.get('use_clahe', True)
    ensemble.img_size = config.get('img_size', IMG_SIZE)

    # Load weights config
    ensemble.weights = config.get('model_weights', {})
    ensemble.per_class_weights = config.get('per_class_weights', None)

    # Models directory (same dir as config file)
    models_dir = os.path.dirname(config_path)

    print(f"[ensemble] Loading ensemble: {ensemble.ensemble_method}")
    print(f"[ensemble] Models directory: {models_dir}")

    for model_info in config['models']:
        name = model_info['name']
        timm_name = model_info['timm_name']
        dropout = model_info.get('dropout', 0.3)
        weight_path = model_info['weight_path']

        # Try weight_path as-is first, then look in models_dir
        if not os.path.isfile(weight_path):
            # Try relative to models_dir
            alt_path = os.path.join(models_dir, os.path.basename(weight_path))
            if os.path.isfile(alt_path):
                weight_path = alt_path
            else:
                print(f"[ensemble] Warning: Weight file not found for {name}: {weight_path}")
                continue

        # Build model
        model = MultiLabelClassifier(
            backbone_name=timm_name,
            num_classes=NUM_CLASSES,
            dropout=dropout,
            pretrained=False,
        )

        # Load weights
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[ensemble]   OK {name} loaded (AUC={checkpoint.get('best_auc', 'N/A')})")
        else:
            model.load_state_dict(checkpoint)
            print(f"[ensemble]   OK {name} loaded (weights only)")

        model = model.to(device)
        model.eval()

        ensemble.models.append(model)
        ensemble.model_names.append(name)

    # Default equal weights if not specified
    if not ensemble.weights:
        n = len(ensemble.model_names)
        ensemble.weights = {name: 1.0 / n for name in ensemble.model_names}

    print(f"[ensemble] {len(ensemble.models)} models loaded on {device}")
    print(f"[ensemble] Method: {ensemble.ensemble_method}")
    print(f"[ensemble] Weights: {ensemble.weights}")
    print(f"[ensemble] Thresholds: {ensemble.thresholds}")

    return ensemble


# ────────────────────────── Post-Processing ────────────────────────── #

def apply_normal_constraint(detected, probabilities):
    """Enforce Normal ⊥ Disease mutual exclusivity.

    Clinical logic: a fundus cannot be simultaneously Normal and diseased.
    If any disease is detected, Normal is removed from the detected list.

    ██  SAFETY  ██  Raw probabilities are NEVER modified. Only the
    'detected' list is filtered. This is consistent with the system's
    safety invariants documented in validation_rules.py.

    Why this helps:
    - Normal precision=0.44 means 56% of Normal predictions are wrong.
    - Those false Normal flags are effectively missed diseases (FN).
    - Removing contradictory Normal flags makes clinical output cleaner.
    """
    normal_name = "Normal"
    diseases = [d for d in detected if d != normal_name]

    if diseases and normal_name in detected:
        detected = [d for d in detected if d != normal_name]

    return detected


# ────────────────────────── Prediction ────────────────────────── #

@torch.no_grad()
def predict_ensemble(
    image_path: str,
    ensemble: EnsembleModel,
    device: torch.device = DEVICE,
) -> tuple:
    """Run ensemble prediction for a single retinal fundus image.

    Parameters
    ----------
    image_path : str
    ensemble : EnsembleModel
    device : torch.device

    Returns
    -------
    detected : list[str]     Human-readable disease names above threshold
    probabilities : dict     disease_name → float probability
    """
    tensor = preprocess_image(
        image_path,
        target_size=ensemble.img_size,
        use_clahe=ensemble.use_clahe,
    ).to(device)

    # Get predictions from each model
    model_probs = []
    for model in ensemble.models:
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # (8,)
        model_probs.append(probs)

    # Combine predictions
    if ensemble.per_class_weights is not None:
        # Per-class weighted combination
        combined = np.zeros(NUM_CLASSES)
        for c, col in enumerate(DISEASE_COLUMNS):
            col_weights = ensemble.per_class_weights.get(col, {})
            for j, name in enumerate(ensemble.model_names):
                w = col_weights.get(name, 1.0 / len(ensemble.model_names))
                combined[c] += w * model_probs[j][c]
    else:
        # Global weighted average
        combined = np.zeros(NUM_CLASSES)
        for j, name in enumerate(ensemble.model_names):
            w = ensemble.weights.get(name, 1.0 / len(ensemble.model_names))
            combined += w * model_probs[j]

    # Apply thresholds
    probabilities = {}
    detected = []
    for i, col in enumerate(DISEASE_COLUMNS):
        name = DISEASE_NAMES[col]
        prob = float(round(combined[i], 4))
        probabilities[name] = prob
        thresh = ensemble.thresholds.get(col, 0.5)
        if prob >= thresh:
            detected.append(name)
    # Apply Normal constraint (mutual exclusivity)
    detected = apply_normal_constraint(detected, probabilities)

    return detected, probabilities


@torch.no_grad()
def predict_ensemble_from_array(
    image_rgb: np.ndarray,
    ensemble: EnsembleModel,
    device: torch.device = DEVICE,
) -> tuple:
    """Same as predict_ensemble but from in-memory RGB array."""
    tensor = preprocess_image_from_array(
        image_rgb,
        target_size=ensemble.img_size,
        use_clahe=ensemble.use_clahe,
    ).to(device)

    model_probs = []
    for model in ensemble.models:
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        model_probs.append(probs)

    if ensemble.per_class_weights is not None:
        combined = np.zeros(NUM_CLASSES)
        for c, col in enumerate(DISEASE_COLUMNS):
            col_weights = ensemble.per_class_weights.get(col, {})
            for j, name in enumerate(ensemble.model_names):
                w = col_weights.get(name, 1.0 / len(ensemble.model_names))
                combined[c] += w * model_probs[j][c]
    else:
        combined = np.zeros(NUM_CLASSES)
        for j, name in enumerate(ensemble.model_names):
            w = ensemble.weights.get(name, 1.0 / len(ensemble.model_names))
            combined += w * model_probs[j]

    probabilities = {}
    detected = []
    for i, col in enumerate(DISEASE_COLUMNS):
        name = DISEASE_NAMES[col]
        prob = float(round(combined[i], 4))
        probabilities[name] = prob
        thresh = ensemble.thresholds.get(col, 0.5)
        if prob >= thresh:
            detected.append(name)
    # Apply Normal constraint (mutual exclusivity)
    detected = apply_normal_constraint(detected, probabilities)

    return detected, probabilities


@torch.no_grad()
def predict_ensemble_with_tta(
    image_path: str,
    ensemble: EnsembleModel,
    device: torch.device = DEVICE,
) -> tuple:
    """Ensemble prediction with 3-flip Test-Time Augmentation.

    Averages predictions over 3 views: original, horizontal flip,
    vertical flip. This smooths probability estimates and reduces
    orientation-dependent prediction noise.

    Why this helps:
    - Fundus images have no canonical orientation. Flipped views may
      reveal lesion patterns missed in the original orientation.
    - Averaging reduces outlier predictions, making threshold decisions
      more stable.
    - FN effect: slight reduction (flipped views catch missed features).
    - FP effect: slight reduction (averaging suppresses spurious peaks).

    Cost: 3x inference time (~300ms -> ~900ms). Acceptable for screening.
    """
    tensor = preprocess_image(
        image_path,
        target_size=ensemble.img_size,
        use_clahe=ensemble.use_clahe,
    ).to(device)

    # 3 views: original, horizontal flip, vertical flip
    views = [
        tensor,
        torch.flip(tensor, dims=[3]),  # H-flip
        torch.flip(tensor, dims=[2]),  # V-flip
    ]

    all_view_probs = []
    for view in views:
        # Get predictions from each model for this view
        model_probs = []
        for model in ensemble.models:
            logits = model(view)
            probs = torch.sigmoid(logits).cpu().numpy()[0]  # (8,)
            model_probs.append(probs)

        # Combine model predictions (same logic as predict_ensemble)
        if ensemble.per_class_weights is not None:
            combined = np.zeros(NUM_CLASSES)
            for c, col in enumerate(DISEASE_COLUMNS):
                col_weights = ensemble.per_class_weights.get(col, {})
                for j, name in enumerate(ensemble.model_names):
                    w = col_weights.get(name, 1.0 / len(ensemble.model_names))
                    combined[c] += w * model_probs[j][c]
        else:
            combined = np.zeros(NUM_CLASSES)
            for j, name in enumerate(ensemble.model_names):
                w = ensemble.weights.get(name, 1.0 / len(ensemble.model_names))
                combined += w * model_probs[j]

        all_view_probs.append(combined)

    # Average across TTA views
    avg_probs = np.mean(all_view_probs, axis=0)

    # Apply thresholds
    probabilities = {}
    detected = []
    for i, col in enumerate(DISEASE_COLUMNS):
        name = DISEASE_NAMES[col]
        prob = float(round(avg_probs[i], 4))
        probabilities[name] = prob
        thresh = ensemble.thresholds.get(col, 0.5)
        if prob >= thresh:
            detected.append(name)

    # Apply Normal constraint
    detected = apply_normal_constraint(detected, probabilities)

    return detected, probabilities


# ────────────────────────── CLI ────────────────────────── #

if __name__ == "__main__":
    import sys
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    config_path = os.path.join(PROJECT_ROOT, "models", "ensemble_config.json")
    if os.path.isfile(config_path):
        ens = load_ensemble(config_path)
        print(f"\nEnsemble loaded: {len(ens.models)} models")
        print(f"Models: {ens.model_names}")

        if len(sys.argv) > 1:
            img_path = sys.argv[1]
            detected, probs = predict_ensemble(img_path, ens)
            print(f"\nPredictions for: {img_path}")
            print("Probabilities:")
            for disease, prob in probs.items():
                marker = " ◀" if disease in detected else ""
                print(f"  {disease:40s}  {prob:.4f}{marker}")
            print(f"\nDetected: {detected}")
    else:
        print(f"Ensemble config not found: {config_path}")
        print("Train the ensemble first using notebooks/ensemble_odir_training.py")
