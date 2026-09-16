---
title: RiceGuard — Rice Leaf Disease Diagnosis & Detection
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.25.0
app_file: app.py
pinned: false
license: mit
---

<div align="center">

# 🌾 RiceGuard

### Rice Leaf Disease Diagnosis with MobileViT-S Vision Transformer (97.4% Accuracy) & YOLOv8s Spot Detector

[![GitHub](https://img.shields.io/badge/GitHub-Nitishvox%2Fpaddydoc-181717?logo=github)](https://github.com/Nitishvox/paddydoc)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Vision_Transformer-MobileViT--S-10b981)](https://github.com/huggingface/pytorch-image-models)
[![Model](https://img.shields.io/badge/Object_Detection-YOLOv8s-16a34a)](https://github.com/ultralytics/ultralytics)
[![ZeroGPU](https://img.shields.io/badge/HuggingFace-ZeroGPU_Ready-blue)](https://huggingface.co/spaces)
[![License](https://img.shields.io/badge/License-MIT-f59e0b)](LICENSE)

</div>

---

## 📌 Overview

**RiceGuard** is a dual-engine AI web application designed for comprehensive diagnosis and localized spot detection of rice crop diseases:

1. **🔬 Whole-Leaf Disease Diagnosis (MobileViT-S Vision Transformer):**
   - Employs **multi-head self-attention** to evaluate global leaf patterns, chlorosis gradients, and vein-delimited symptoms.
   - Evaluated on **234 test images** across 6 distinct disease states, achieving **97.44% Test Accuracy** and **97.60% Macro F1-Score**.
   - Achieves **100% precision and 100% recall on Bacterial Leaf Blight**, completely resolving the background-confusion challenge common to pure CNN architectures.
   - Delivers actionable **agronomic treatment recommendations and severity ratings** for farm management.

2. **🎯 Lesion Spot Detector (YOLOv8s Multi-Stage):**
   - Pinpoints and outlines individual lesion spots on the leaf blade with bounding box coordinates.
   - Supports **real-time field video analysis** frame-by-frame with customizable stride.

---

## 📊 MobileViT-S Benchmark & Test Metrics

### Test Set Performance Summary
- **Overall Test Accuracy:** `97.44%`
- **Macro F1-Score:** `97.60%`
- **Weighted Average F1-Score:** `97.42%`
- **Test Samples:** `234` images

### Detailed Per-Class Classification Report

| Disease Class | Precision | Recall | F1-Score | Test Support | Severity Level |
|---|---|---|---|---|---|
| **Bacterial Leaf Blight** (*X. oryzae*) | **1.000** | **1.000** | **1.000** | 36 | High |
| **Narrow Brown Spot** (*C. janseana*) | **1.000** | **1.000** | **1.000** | 39 | Moderate |
| **Leaf Scald** (*M. oryzae*) | **1.000** | **1.000** | **1.000** | 34 | Medium |
| **Healthy Leaf** (Disease-free) | **0.976** | **1.000** | **0.988** | 41 | None |
| **Brown Spot** (*B. oryzae*) | **0.930** | **0.952** | **0.941** | 42 | Medium-High |
| **Leaf Blast** (*M. oryzae*) | **0.950** | **0.905** | **0.927** | 42 | Critical |
| **Macro Average** | **0.976** | **0.976** | **0.976** | 234 | — |

---

## 💡 Architectural Insights: Why MobileViT Solved the Blight Problem

In earlier iterations using localized patch detectors (YOLOv8s), **Bacterial Leaf Blight** suffered from high false negatives (~37% recall) due to symptom confusion with natural leaf variations and background soil:
- Traditional convolutional kernels operate within small receptive fields (e.g. 3×3 or 5×5), struggling to separate diffuse marginal wilting from background.
- **MobileViT Solution:** Integrates Transformer blocks with **Multi-Head Self-Attention** inside MobileNet inverted residual stages. Every spatial patch attends to all other patches across the entire leaf, learning global leaf margins, vein borders, and necrosis progression.
- This architectural shift achieved **zero false positives and zero false negatives** for Bacterial Leaf Blight on the test benchmark.

---

## 🚀 Quick Start (Local Setup)

```bash
# 1. Clone the repository
git clone https://github.com/Nitishvox/paddydoc.git
cd paddydoc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Gradio application
python app.py
```
Open your browser at `http://localhost:7860`.

---

## 🌐 Deployment to Hugging Face Spaces

1. Create a Space on [Hugging Face Spaces](https://huggingface.co/new-space) (SDK: **Gradio**).
2. Set Space Hardware to **ZeroGPU (Free)** for dynamic Nvidia GPU acceleration.
3. Push your repository:
   ```bash
   git remote add hf https://huggingface.co/spaces/Nitishvox/paddydoc
   git push hf main
   ```

---

## 📁 Repository Structure

```text
rice_disease_detector/
├── app.py                      # Main Gradio application (4 interactive tabs)
├── config.py                   # Central configuration & disease knowledge base
├── models/
│   ├── classifier_handler.py   # MobileViT-S Vision Transformer inference
│   └── yolo_handler.py         # YOLOv8s spot detection & video processor
├── utils/
│   └── image_utils.py          # Image resizing, table formatting & draw helpers
├── weights/
│   ├── rice_leaf_final_model.pt # MobileViT-S weights (97.4% accuracy)
│   ├── stage1_best.pt          # YOLOv8s Stage 1 weights (640px)
│   └── stage2_best.pt          # YOLOv8s Stage 2 weights (832px)
├── assets/
│   └── mobilevit_confusion_matrix.png # Test set confusion matrix plot
├── requirements.txt            # Pinned dependencies (timm, ultralytics, gradio, etc.)
└── README.md                   # Project documentation
```

---

## 📜 License

Distributed under the [MIT License](LICENSE).
