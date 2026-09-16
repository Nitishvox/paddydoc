"""
MobileViT-S Leaf Disease Classification Handler.

Architecture:
- Backbone: MobileViT-S (Vision Transformer with multi-head self-attention)
- Head: Dropout(0.2) + Linear(640, 6)
- Test Accuracy: 97.44% | Macro F1: 97.60%
- Classes:
    1. bacterial_leaf_blight
    2. brown_spot
    3. healthy
    4. leaf_blast
    5. leaf_scald
    6. narrow_brown_spot

Features:
- Global self-attention modeling for whole-leaf symptom pattern recognition
- ZeroGPU dynamic allocation support on Hugging Face Spaces (@spaces.GPU)
- Thread-safe model caching and warm-up
"""

import os
import threading
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# ---------------------------------------------------------------------------
# Hugging Face ZeroGPU decorator (activates GPU dynamically on HF Spaces)
# ---------------------------------------------------------------------------
try:
    import spaces
    gpu_decorator = spaces.GPU
except Exception:
    def gpu_decorator(fn):
        return fn

_classifier_cache = {}
_classifier_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Architecture Definition
# ---------------------------------------------------------------------------
class RiceMobileViT(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        import timm
        self.backbone = timm.create_model("mobilevit_s", pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(640, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)


# ---------------------------------------------------------------------------
# Preprocessing Transforms
# ---------------------------------------------------------------------------
TRANSFORM_256 = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------
def load_classifier(model_path: str, device: Optional[str] = None) -> RiceMobileViT:
    """Load and cache the MobileViT classifier."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with _classifier_lock:
        if model_path not in _classifier_cache:
            if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
                raise FileNotFoundError(f"Classifier weight file not found at: {model_path}")

            model = RiceMobileViT(num_classes=6)
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            # Warm-up pass
            try:
                dummy = torch.randn(1, 3, 256, 256, device=device)
                with torch.no_grad():
                    model(dummy)
            except Exception:
                pass

            _classifier_cache[model_path] = (model, device)

    return _classifier_cache[model_path][0]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@gpu_decorator
def run_mobilevit_classification(
    pil_image: Image.Image,
    model_path: str,
    class_names: list[str],
) -> tuple[str, float, dict[str, float], Optional[str]]:
    """
    Run MobileViT-S classification on a PIL leaf image.

    Returns:
    --------
    top_class : str
    confidence : float
    prob_dict  : dict mapping formatted class label to confidence (0.0 to 1.0)
    error      : Optional error string
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_classifier(model_path, device=device)

        rgb_image = pil_image.convert("RGB")
        input_tensor = TRANSFORM_256(rgb_image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        prob_dict = {}
        for idx, prob in enumerate(probs):
            name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
            prob_dict[name] = float(prob)

        top_idx = int(probs.argmax())
        top_class = class_names[top_idx]
        confidence = float(probs[top_idx])

        return top_class, confidence, prob_dict, None
    except Exception as exc:
        return "Unknown", 0.0, {}, str(exc)
