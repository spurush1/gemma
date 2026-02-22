"""
CheXNet Client — Standardized Chest X-Ray Analysis
====================================================
Uses torchxrayvision DenseNet-121 (NIH ChestX-ray14 weights) to produce
structured, standardized pathology labels from chest X-ray images.

Purpose: Fix MedGemma's visual encoder failure problem (Problem 1 FPs).
  MedGemma:  "bilateral_infiltrates" or "" (missed entirely)
  CheXNet:   {"Consolidation": 0.87} → "lobar_consolidation", "consolidation"

CheXNet findings are merged with MedGemma findings in analyze_image_only().
The merged set provides maximum coverage before the causal audit evaluation.

Architecture decision:
  - torchxrayvision is an OPTIONAL dependency (graceful fallback to MedGemma-only)
  - Model weights (~25MB) are cached in ~/.cache/torchxrayvision/ automatically
  - CPU-only inference — no GPU required (runs on M4 Mac Mini)
  - First call downloads weights; subsequent calls use cache (~100ms inference)
"""

import os
import base64
import logging
from io import BytesIO
from typing import Optional

logger = logging.getLogger("chexnet_client")

# Confidence threshold for detecting a finding
# 0.4 = standard threshold from original CheXNet paper (Rajpurkar et al. 2017)
CHEXNET_THRESHOLD = float(os.getenv("CHEXNET_THRESHOLD", "0.4"))

# torchxrayvision model weights: NIH ChestX-ray14 dataset
# This is the closest publicly available weights to the original CheXNet
CHEXNET_WEIGHTS = os.getenv("CHEXNET_WEIGHTS", "densenet121-res224-nih")

# =========================================================
# CHEXNET LABEL → KNOWLEDGE ENGINE TERM MAPPING
# Maps NIH ChestX-ray14 standardized labels to our causal requirement terms
# =========================================================

CHEXNET_TO_KE = {
    "Atelectasis":          ["atelectasis", "volume_loss"],
    "Consolidation":        ["lobar_consolidation", "consolidation"],
    "Infiltration":         ["consolidation", "pulmonary_infiltrates", "airspace_disease"],
    "Pneumothorax":         ["pneumothorax"],
    "Edema":                ["pulmonary_edema", "kerley_b_lines", "vascular_redistribution"],
    "Emphysema":            ["hyperinflation", "air_trapping", "flattened_diaphragms"],
    "Fibrosis":             ["pulmonary_fibrosis"],
    "Effusion":             ["pleural_effusion", "costophrenic_blunting", "layering_fluid"],
    "Pneumonia":            ["lobar_consolidation", "pneumonia", "air_bronchograms"],
    "Pleural_Thickening":   ["pleural_thickening"],
    "Cardiomegaly":         ["cardiomegaly", "enlarged_cardiac_silhouette"],
    "Nodule":               ["pulmonary_nodule", "nodule"],
    "Mass":                 ["pulmonary_mass", "mass"],
    "Hernia":               ["diaphragmatic_hernia"],
    "No Finding":           ["normal_chest", "clear_lung_fields", "no_acute_findings"],
}

# Lazy-loaded model cache (loaded once, reused for all requests)
_model = None
_xrv = None


def _load_model():
    """
    Load CheXNet model with lazy initialization.
    Returns (model, xrv_module) or (None, None) if unavailable.
    """
    global _model, _xrv

    if _model is not None:
        return _model, _xrv

    try:
        import torchxrayvision as xrv
        import torch

        logger.info(f"Loading CheXNet ({CHEXNET_WEIGHTS}) — first call only...")
        model = xrv.models.DenseNet(weights=CHEXNET_WEIGHTS)
        model.eval()

        # Move to CPU explicitly (M4 GPU not used)
        model = model.cpu()

        _model = model
        _xrv = xrv
        logger.info("CheXNet model loaded and cached.")
        return _model, _xrv

    except ImportError:
        logger.info(
            "torchxrayvision not installed — CheXNet disabled. "
            "Using MedGemma-only image analysis. "
            "Install with: pip install torchxrayvision"
        )
        return None, None

    except Exception as e:
        logger.warning(f"CheXNet load failed ({e}) — falling back to MedGemma-only.")
        return None, None


def analyze_with_chexnet(image_base64: str) -> dict:
    """
    Run CheXNet (DenseNet-121, NIH weights) on a chest X-ray.

    Returns:
        {
            "available": bool,
            "findings": list[str],          # knowledge-engine-compatible terms
            "probabilities": dict,           # raw NIH label → probability
            "positive_labels": list[str],    # CheXNet labels above threshold
            "error": str (only on failure)
        }
    """
    model, xrv = _load_model()

    if model is None:
        return {
            "available": False,
            "findings": [],
            "probabilities": {},
            "positive_labels": [],
        }

    try:
        import torch
        import numpy as np
        from PIL import Image

        # Decode base64 → PIL image → grayscale numpy array
        img_bytes = base64.b64decode(image_base64)
        img = Image.open(BytesIO(img_bytes)).convert("L")   # grayscale
        img_np = np.array(img)

        # torchxrayvision normalization: scale to [-1024, 1024]
        img_np = xrv.datasets.normalize(img_np, maxval=255, reshape=True)

        # Center-crop and resize to 224×224
        transform = xrv.datasets.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
        ])
        img_np = transform(img_np)

        # Add batch dimension and run inference
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()

        with torch.no_grad():
            preds = model(img_tensor)
            probs = torch.sigmoid(preds).squeeze().cpu().numpy()

        # Build label → probability dict
        labels = model.pathologies
        prob_dict = {label: float(prob) for label, prob in zip(labels, probs)}

        # Filter positive findings above threshold
        positive_labels = [
            label for label, prob in prob_dict.items()
            if prob >= CHEXNET_THRESHOLD and label in CHEXNET_TO_KE
        ]

        # Map to knowledge engine terms (union of all mapped terms)
        ke_findings = []
        for label in positive_labels:
            ke_findings.extend(CHEXNET_TO_KE.get(label, []))
        ke_findings = list(set(ke_findings))

        logger.info(
            f"CheXNet: {positive_labels} (threshold={CHEXNET_THRESHOLD}) "
            f"→ {ke_findings}"
        )

        return {
            "available": True,
            "findings": ke_findings,
            "probabilities": prob_dict,
            "positive_labels": positive_labels,
        }

    except Exception as e:
        logger.error(f"CheXNet inference error: {e}")
        return {
            "available": False,
            "findings": [],
            "probabilities": {},
            "positive_labels": [],
            "error": str(e),
        }


def is_available() -> bool:
    """Check if CheXNet is available without triggering full model load."""
    try:
        import torchxrayvision  # noqa: F401
        import torch             # noqa: F401
        return True
    except ImportError:
        return False


# =========================================================
# STANDALONE SMOKE TEST
# =========================================================

if __name__ == "__main__":
    import sys

    print(f"CheXNet available: {is_available()}")
    if not is_available():
        print("Install with: pip install torchxrayvision torch torchvision")
        sys.exit(0)

    # Use a sample image if provided
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        result = analyze_with_chexnet(img_b64)
        print(f"Positive labels: {result['positive_labels']}")
        print(f"KE findings:     {result['findings']}")
        print(f"Top probabilities:")
        for label, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1])[:5]:
            print(f"  {label}: {prob:.3f}")
    else:
        print("Usage: python chexnet_client.py <chest_xray.jpg>")
        model, _ = _load_model()
        print(f"Model loaded: {model is not None}")
