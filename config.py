"""
Central configuration for RiceGuard — Rice Disease Diagnosis & Detection System.
Keeps model paths, disease knowledge base, evaluation metrics, and registry in one place.
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
ASSETS_DIR = BASE_DIR / "assets"

CLASSIFIER_PATH = WEIGHTS_DIR / "rice_leaf_final_model.pt"
CONFUSION_MATRIX_IMAGE = ASSETS_DIR / "mobilevit_confusion_matrix.png"

# ---------------------------------------------------------------------------
# MobileViT-S 6-Class Disease Taxonomy & Agronomic Recommendations
# ---------------------------------------------------------------------------
CLASSIFIER_RAW_CLASSES = [
    "brown_spot",
    "healthy",
    "leaf_blast",
    "leaf_scald",
    "narrow_brown_spot",
    "bacterial_leaf_blight",
]

DISEASE_INFO = {
    "bacterial_leaf_blight": {
        "display_name": "Bacterial Leaf Blight",
        "scientific_name": "Xanthomonas oryzae pv. oryzae",
        "badge_color": "#FF5722",
        "severity": "High",
        "description": "Water-soaked to yellowish-white lesions with wavy margins starting from leaf tips, rapidly wilting leaves.",
        "symptoms": "Translucent stripes along leaf margins turning yellow-orange and drying up to grayish white.",
        "treatment": "Apply Copper Hydroxide (Kocide) or Streptocycline; ensure balanced nitrogen fertilizer application and drain flooded fields.",
        "f1_score": 1.000,
        "precision": 1.000,
        "recall": 1.000,
    },
    "brown_spot": {
        "display_name": "Brown Spot",
        "scientific_name": "Bipolaris oryzae (Cochliobolus miyabeanus)",
        "badge_color": "#FFC107",
        "severity": "Medium-High",
        "description": "Oval-shaped dark brown lesions with yellow halo, strongly linked to nutrient-deficient soil or water stress.",
        "symptoms": "Numerous small circular brown spots expanding to oval spots with gray centers and yellow chlorotic margins.",
        "treatment": "Foliar spray of Mancozeb, Carbendazim, or Tricyclazole; rectify soil potassium and silicon deficiencies.",
        "f1_score": 0.941,
        "precision": 0.930,
        "recall": 0.952,
    },
    "healthy": {
        "display_name": "Healthy Rice Leaf",
        "scientific_name": "Oryza sativa (Disease-free)",
        "badge_color": "#4CAF50",
        "severity": "None",
        "description": "Vibrant green leaf blade with intact cuticle, normal chlorophyll distribution, and no visible fungal or bacterial lesions.",
        "symptoms": "Uniform green coloration without necrotic spots, chlorotic margins, or lesion bands.",
        "treatment": "Maintain regular irrigation, crop rotation, and optimal NPK nutrient schedule.",
        "f1_score": 0.988,
        "precision": 0.976,
        "recall": 1.000,
    },
    "leaf_blast": {
        "display_name": "Rice Leaf Blast",
        "scientific_name": "Magnaporthe oryzae (Pyricularia oryzae)",
        "badge_color": "#E91E63",
        "severity": "Critical",
        "description": "Spindle/diamond-shaped lesions with gray/whitish centers and brown-red margins, capable of causing devastating crop loss.",
        "symptoms": "Elliptical spots with pointed ends; multiple lesions coalesce and kill entire leaf blades prematurely.",
        "treatment": "Spray Tricyclazole (75 WP), Isoprothiolane, or Kasugamycin at first sign; avoid excessive nitrogen fertilizer.",
        "f1_score": 0.927,
        "precision": 0.950,
        "recall": 0.905,
    },
    "leaf_scald": {
        "display_name": "Leaf Scald",
        "scientific_name": "Microdochium oryzae (Monographella albescens)",
        "badge_color": "#9C27B0",
        "severity": "Medium",
        "description": "Concentric wavy bands of alternating light brown and olive-green on leaf tips and margins giving a scalded appearance.",
        "symptoms": "Zonate chevron-like patterns advancing along the leaf blade; leaves become brittle and dry.",
        "treatment": "Apply Benomyl or Carbendazim; use certified clean seed and ensure balanced potassium levels.",
        "f1_score": 1.000,
        "precision": 1.000,
        "recall": 1.000,
    },
    "narrow_brown_spot": {
        "display_name": "Narrow Brown Leaf Spot",
        "scientific_name": "Cercospora janseana",
        "badge_color": "#795548",
        "severity": "Moderate",
        "description": "Short, linear narrow brown lesions running parallel to leaf veins, typically appearing during late crop stages.",
        "symptoms": "Distinctly narrow brown to reddish-brown streaks (2-10 mm long) strictly delimited by leaf veins.",
        "treatment": "Foliar application of Propiconazole or Azoxystrobin; maintain adequate soil potassium.",
        "f1_score": 1.000,
        "precision": 1.000,
        "recall": 1.000,
    },
}

# ---------------------------------------------------------------------------
# YOLOv8 Spot Detection Classes (Stage 1 & Stage 2 Bounding Box models)
# ---------------------------------------------------------------------------
YOLO_CLASSES = ["Blast", "Blight", "Brownspot", "Healthy"]

# ---------------------------------------------------------------------------
# Benchmark & Evaluation Metrics (MobileViT-S Final Test Set)
# ---------------------------------------------------------------------------
TEST_BENCHMARK = {
    "test_accuracy": 0.9743589743589743,
    "test_macro_f1": 0.975992924351639,
    "per_class": {
        "bacterial_leaf_blight": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 36},
        "narrow_brown_spot": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 39},
        "leaf_scald": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 34},
        "healthy": {"precision": 0.97619, "recall": 1.0, "f1": 0.98795, "support": 41},
        "brown_spot": {"precision": 0.93023, "recall": 0.95238, "f1": 0.94118, "support": 42},
        "leaf_blast": {"precision": 0.95, "recall": 0.90476, "f1": 0.92683, "support": 42},
    },
}

# ---------------------------------------------------------------------------
# Model Registry (for YOLO Local Detection models)
# ---------------------------------------------------------------------------
@dataclass
class ModelEntry:
    key:         str
    label:       str
    path:        str  = ""
    description: str  = ""
    map50:       float = 0.0
    epochs:      int   = 0
    imgsz:       int   = 640
    available:   bool  = True


def _check(path: str) -> bool:
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


MODEL_REGISTRY: list[ModelEntry] = [
    ModelEntry(
        key="v2",
        label="🟢 Stage 2 YOLOv8s — High-Res Spot Detector (832px, 100 ep)",
        path=str(WEIGHTS_DIR / "stage2_best.pt"),
        description="Best spot detection checkpoint. Higher 832px resolution for pinpointing small lesions.",
        map50=0.569,
        epochs=100,
        imgsz=832,
    ),
    ModelEntry(
        key="v1",
        label="🟡 Stage 1 YOLOv8s — Baseline Spot Detector (640px, 65 ep)",
        path=str(WEIGHTS_DIR / "stage1_best.pt"),
        description="Baseline spot detection model trained for 65 epochs at 640×640.",
        map50=0.557,
        epochs=65,
        imgsz=640,
    ),
]

for _m in MODEL_REGISTRY:
    _m.available = _check(_m.path)

MODEL_LOOKUP: dict[str, ModelEntry] = {m.key: m for m in MODEL_REGISTRY}

DEFAULT_CONF_THRESHOLD = 0.40
DEFAULT_FRAME_STRIDE = 2
MAX_VIDEO_FRAMES = 500