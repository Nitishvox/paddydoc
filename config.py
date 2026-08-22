"""
Central configuration for the RiceGuard — Rice Leaf Disease Detector.
All model paths, registry entries, and runtime settings live here.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"

# ---------------------------------------------------------------------------
# Class names and display colours (BGR for OpenCV, hex for Gradio)
# ---------------------------------------------------------------------------
CLASS_NAMES = ["Blast", "Blight", "Brownspot", "Healthy"]

CLASS_COLORS_HEX = {
    "Blast":     "#FF4444",   # vivid red
    "Blight":    "#FF8C00",   # dark orange
    "Brownspot": "#FFD700",   # gold
    "Healthy":   "#00C851",   # green
}

CLASS_COLORS_BGR = {
    "Blast":     (68,  68,  255),
    "Blight":    (0,  140,  255),
    "Brownspot": (0,  215,  255),
    "Healthy":   (81, 200,  0),
}

# ---------------------------------------------------------------------------
# Roboflow config  (values come from .env or HF Spaces secrets)
# ---------------------------------------------------------------------------
ROBOFLOW_API_URL       = "https://serverless.roboflow.com"
ROBOFLOW_WORKSPACE     = os.environ.get("ROBOFLOW_WORKSPACE", "m-nitish-46wkd")
ROBOFLOW_WORKFLOW_ID   = os.environ.get(
    "ROBOFLOW_WORKFLOW_ID",
    "leaf-disease-vleaf-disease-xi4nf-m3rft-1-rfdetr-medium-t1-logic"
)
ROBOFLOW_API_KEY       = os.environ.get("ROBOFLOW_API_KEY", "")

# ---------------------------------------------------------------------------
# Model registry — every entry becomes a selectable dropdown option
# ---------------------------------------------------------------------------
@dataclass
class ModelEntry:
    key:         str
    label:       str
    kind:        str            # "local" | "roboflow"
    path:        str  = ""
    description: str  = ""
    map50:       float = 0.0
    epochs:      int   = 0
    imgsz:       int   = 640
    available:   bool  = True   # set False at runtime if weights file missing


def _check(path: str) -> bool:
    """Return True only if the weight file exists and is non-empty."""
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


MODEL_REGISTRY: list[ModelEntry] = [
    ModelEntry(
        key="v1",
        label="🟡  Stage 1 — YOLOv8s · 65 epochs · 640 px",
        kind="local",
        path=str(WEIGHTS_DIR / "stage1_best.pt"),
        description=(
            "Baseline model trained for 65 epochs at 640×640. "
            "Solid starting point. mAP50: 0.557"
        ),
        map50=0.557,
        epochs=65,
        imgsz=640,
    ),
    ModelEntry(
        key="v2",
        label="🟢  Stage 2 — YOLOv8s · 100 epochs · 832 px  ★ Best",
        kind="local",
        path=str(WEIGHTS_DIR / "stage2_best.pt"),
        description=(
            "Continued from Stage 1 for 35 more epochs at higher 832×832 resolution. "
            "Best local model. mAP50: 0.569"
        ),
        map50=0.569,
        epochs=100,
        imgsz=832,
    ),
    ModelEntry(
        key="finetune",
        label="🔴  Fine-tune — experimental (not recommended)",
        kind="local",
        path=str(WEIGHTS_DIR / "finetune_best.pt"),
        description=(
            "15-epoch fine-tune with stronger augmentation — actually regressed. "
            "Kept for honest comparison only. mAP50: 0.483"
        ),
        map50=0.483,
        epochs=115,
        imgsz=832,
    ),
    ModelEntry(
        key="roboflow",
        label="🔵  Roboflow — RF-DETR Medium (cloud hosted)",
        kind="roboflow",
        description=(
            "Externally hosted RF-DETR model via Roboflow Workflows API. "
            "Requires API key. mAP: 52.7 %, Precision: 57.9 %, Recall: 58.2 %"
        ),
        map50=0.527,
    ),
]

# Mark unavailable models (missing / empty weight files)
for _m in MODEL_REGISTRY:
    if _m.kind == "local":
        _m.available = _check(_m.path)

MODEL_LOOKUP: dict[str, ModelEntry] = {m.key: m for m in MODEL_REGISTRY}

# ---------------------------------------------------------------------------
# Inference defaults
# ---------------------------------------------------------------------------
DEFAULT_CONF_THRESHOLD   = 0.40
DEFAULT_FRAME_STRIDE     = 2          # process every Nth frame in video mode
MAX_VIDEO_FRAMES         = 500        # hard cap to keep processing time sane
REQUEST_TIMEOUT_SECONDS  = 30
MAX_RETRIES              = 2