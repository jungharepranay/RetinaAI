"""
explainability.py
-----------------
Grad-CAM visual explanation module for EfficientNet-B3 (PyTorch).

Generates class-discriminative heatmap overlays showing which regions
of the fundus image the CNN focused on for each prediction.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.efficientnet_model import (
    preprocess_image_from_array,
    DEVICE,
    IMG_SIZE,
)


# ────────────────────────── Grad-CAM Core ────────────────────────── #

class GradCAMPyTorch:
    """Grad-CAM for any PyTorch CNN.

    Hooks into a target convolutional layer, computes:
        weights = GAP(d_output / d_features)
        cam     = ReLU( Σ weights * features )

    For EfficientNet-B3 via timm, the last conv block is usually
    ``model.backbone.conv_head`` or the last block in ``model.backbone.blocks``.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or self._auto_find_layer(model)
        self._features = None
        self._gradients = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_features)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    @staticmethod
    def _auto_find_layer(model):
        """Auto-detect the last Conv2d layer in the backbone."""
        last_conv = None
        # Search through backbone (timm EfficientNet structure)
        backbone = getattr(model, "backbone", model)
        for module in backbone.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer found in the model.")
        return last_conv

    def _save_features(self, module, input, output):
        self._features = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_index: int) -> np.ndarray:
        """Generate a Grad-CAM heatmap for the given class.

        Parameters
        ----------
        input_tensor : torch.Tensor   shape (1, 3, H, W)
        class_index  : int

        Returns
        -------
        cam : np.ndarray  shape (H_feat, W_feat), normalised [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward
        logits = self.model(input_tensor)
        target = logits[0, class_index]

        # Backward
        self.model.zero_grad()
        target.backward(retain_graph=True)

        gradients = self._gradients  # (1, C, H, W)
        features = self._features   # (1, C, H, W)

        if gradients is None or features is None:
            return np.zeros((7, 7))  # fallback

        # Global average pooling of gradients → channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination
        cam = (weights * features).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalise
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


# ────────────────────────── Public API ────────────────────────── #

def generate_gradcam(image_rgb: np.ndarray,
                     model: torch.nn.Module,
                     class_index: int = 0) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap overlay for a given class.

    Parameters
    ----------
    image_rgb   : np.ndarray  uint8 RGB, any size
    model       : EfficientNetB3Classifier
    class_index : int

    Returns
    -------
    overlay : np.ndarray  uint8 RGB (224×224) — original + heatmap blend
    """
    # Preprocess for model input
    input_tensor = preprocess_image_from_array(image_rgb).to(DEVICE)

    # Generate CAM
    grad_cam = GradCAMPyTorch(model)
    cam = grad_cam.generate(input_tensor, class_index)

    # Resize CAM to image dimensions
    display_img = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

    # Apply colourmap
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = cv2.addWeighted(display_img, 0.6, heatmap, 0.4, 0)
    return overlay


def generate_gradcam_for_all(image_rgb: np.ndarray,
                              model: torch.nn.Module,
                              predictions: dict,
                              output_dir: str,
                              disease_columns: list = None,
                              disease_names: dict = None,
                              threshold: float = 0.3) -> dict:
    """
    Generate and save Grad-CAM heatmaps for all detected diseases.

    Parameters
    ----------
    image_rgb    : np.ndarray  uint8 RGB (raw fundus image)
    model        : EfficientNetB3Classifier
    predictions  : dict  disease_name → confidence
    output_dir   : str
    threshold    : float

    Returns
    -------
    dict  disease_name → saved file path
    """
    if disease_columns is None:
        from src.data_loader import DISEASE_COLUMNS, DISEASE_NAMES
        disease_columns = DISEASE_COLUMNS
        if disease_names is None:
            disease_names = DISEASE_NAMES

    os.makedirs(output_dir, exist_ok=True)
    heatmap_paths = {}

    name_to_col = {v: k for k, v in disease_names.items()}

    for disease_name, confidence in predictions.items():
        if confidence < threshold:
            continue

        col_key = name_to_col.get(disease_name)
        if col_key is None:
            continue

        class_index = disease_columns.index(col_key)

        try:
            overlay = generate_gradcam(image_rgb, model, class_index)
            safe_name = disease_name.replace(" ", "_").lower()
            filename = f"gradcam_{safe_name}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath,
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            heatmap_paths[disease_name] = filepath
        except Exception as e:
            print(f"[explainability] Grad-CAM failed for "
                  f"{disease_name}: {e}")

    return heatmap_paths


if __name__ == "__main__":
    print("Explainability module loaded (PyTorch Grad-CAM). "
          "Use generate_gradcam() or generate_gradcam_for_all().")
