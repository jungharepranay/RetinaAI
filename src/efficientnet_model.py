"""
efficientnet_model.py
---------------------
EfficientNet-B3 (PyTorch / timm) multi-label retinal disease classifier.

Provides:
  - Model architecture (must match training exactly)
  - Weight loading from .pth checkpoint or state_dict
  - CLAHE + ImageNet-normalised preprocessing
  - Single-image prediction helper
"""

import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm

from src.data_loader import DISEASE_COLUMNS, DISEASE_NAMES

# ────────────────────────── Constants ────────────────────────── #

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_SIZE = 224
NUM_CLASSES = 8

# Default per-class thresholds (fallback if checkpoint doesn't contain them)
DEFAULT_THRESHOLDS = {c: 0.5 for c in DISEASE_COLUMNS}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ────────────────────────── Model Class ────────────────────────── #

class EfficientNetB3Classifier(nn.Module):
    """EfficientNet-B3 backbone with custom multi-label classification head.

    Architecture must match the training script exactly:
        backbone (timm efficientnet_b3, no head) → Dropout → Linear(1536, 8)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=False,       # we load our own weights
            num_classes=0,          # remove classifier → returns features
            drop_rate=0.0,
        )
        in_features = self.backbone.num_features  # 1536 for B3
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)     # [B, 1536]
        logits = self.head(features)    # [B, num_classes]
        return logits


# ────────────────────────── Loading ────────────────────────── #

def load_efficientnet_model(
    model_path: str,
    device: torch.device = DEVICE,
) -> tuple:
    """Load an EfficientNet-B3 model from a .pth file.

    Handles both formats saved by the training script:
      1. Full checkpoint  (dict with 'model_state_dict', 'optimal_thresholds', …)
      2. Bare state_dict  (OrderedDict of tensors)

    Returns
    -------
    model : EfficientNetB3Classifier  (eval mode, on *device*)
    thresholds : dict   column_key → float threshold
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = EfficientNetB3Classifier(
        num_classes=NUM_CLASSES,
        dropout=0.3,
    )

    # Detect format
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Full checkpoint
        model.load_state_dict(checkpoint["model_state_dict"])
        thresholds = checkpoint.get("optimal_thresholds", DEFAULT_THRESHOLDS)
        best_auc = checkpoint.get("best_auc", "N/A")
        print(f"[efficientnet] Loaded FULL checkpoint  (best AUC={best_auc})")
        print(f"[efficientnet] Optimal thresholds: {thresholds}")
    else:
        # Bare state_dict
        model.load_state_dict(checkpoint)
        thresholds = dict(DEFAULT_THRESHOLDS)
        print("[efficientnet] Loaded state_dict (weights only, using default thresholds)")

    model = model.to(device)
    model.eval()
    print(f"[efficientnet] Model on {device}, eval mode")
    return model, thresholds


# ────────────────────────── Preprocessing ────────────────────────── #

def apply_clahe_lab(image_rgb: np.ndarray,
                    clip_limit: float = 2.0,
                    tile_size: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE on LAB luminance channel (matches training pipeline).

    Parameters
    ----------
    image_rgb : np.ndarray  uint8, shape (H, W, 3), RGB

    Returns
    -------
    np.ndarray  uint8, shape (H, W, 3), RGB
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_enhanced = clahe.apply(l_ch)
    lab_out = cv2.merge([l_enhanced, a_ch, b_ch])
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)


def preprocess_image(image_path: str,
                     target_size: int = IMG_SIZE,
                     use_clahe: bool = True) -> torch.Tensor:
    """Load, CLAHE-enhance, resize, normalise, and return a batch tensor.

    The pipeline exactly mirrors the *validation* transforms used during
    training (CLAHE → Resize → ImageNet normalise → ToTensor).

    Returns
    -------
    torch.Tensor  shape (1, 3, H, W), float32
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # uint8 RGB

    if use_clahe:
        img = apply_clahe_lab(img)

    img = cv2.resize(img, (target_size, target_size))

    # Normalise to [0,1] then apply ImageNet stats
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    # HWC → CHW → add batch dim
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def preprocess_image_from_array(image_rgb: np.ndarray,
                                target_size: int = IMG_SIZE,
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


# ────────────────────────── Prediction ────────────────────────── #

@torch.no_grad()
def predict_single(
    image_path: str,
    model: nn.Module,
    thresholds: dict,
    device: torch.device = DEVICE,
) -> tuple:
    """Run prediction for a single retinal fundus image.

    Returns
    -------
    detected : list[str]     human-readable disease names above threshold
    probabilities : dict     disease_name → float probability
    """
    tensor = preprocess_image(image_path).to(device)
    logits = model(tensor)
    probs = torch.sigmoid(logits).cpu().numpy()[0]  # (8,)

    probabilities = {}
    detected = []
    for i, col in enumerate(DISEASE_COLUMNS):
        name = DISEASE_NAMES[col]
        prob = float(round(probs[i], 4))
        probabilities[name] = prob
        thresh = thresholds.get(col, 0.5)
        if prob >= thresh:
            detected.append(name)

    return detected, probabilities


if __name__ == "__main__":
    import sys
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Quick smoke test
    ckpt = os.path.join(PROJECT_ROOT, "models", "efficientnet_odir_final_checkpoint.pth")
    if os.path.isfile(ckpt):
        m, t = load_efficientnet_model(ckpt)
        print(f"Model loaded. Thresholds: {t}")
    else:
        print(f"Checkpoint not found at {ckpt}")
