---
title: RiceGuard — Rice Leaf Disease Detector
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.25.0
app_file: app.py
pinned: false
license: mit
---

# 🌾 RiceGuard — Rice Leaf Disease Detector

AI-powered rice leaf disease detection using YOLOv8 object detection.
Detects **Blast**, **Blight**, **Brownspot**, and **Healthy** leaves in images and videos.

## Features

| Feature | Details |
|---|---|
| 📷 Image detection | Upload, webcam, or clipboard paste |
| 🎬 Video detection | Frame-by-frame annotated video output |
| ⚖️ Model comparison | All three models side by side |
| 📊 Training history | Full metrics and honest limitation notes |
| ⚡ Model caching | Switch models without reloading from disk |
| 🔁 Roboflow cloud | Optional RF-DETR hosted model via API |

## Models

| Model | Epochs | Resolution | mAP50 |
|---|---|---|---|
| Stage 1 (baseline) | 65 | 640 px | 0.557 |
| **Stage 2 ★ Best** | **100** | **832 px** | **0.569** |
| Fine-tune (experimental) | 115 | 832 px | 0.483 |
| Roboflow RF-DETR (cloud) | — | — | ~0.527 |

> **⚠️ Known limitation:** Blight detection recall is only ~37%. The model struggles to distinguish Blight lesions from background — a documented hard problem in rice disease research.

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-username/rice_disease_detector.git
cd rice_disease_detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy env file and fill in your Roboflow key (optional)
cp .env.example .env

# 4. Run
python app.py
# → Open http://localhost:7860
```

## Deploy to Hugging Face Spaces

```bash
# 1. Create a new Space at huggingface.co/new-space
#    SDK: Gradio  |  Hardware: CPU Basic (free)

# 2. Install Git LFS (one time)
git lfs install

# 3. Push to the Space's repo
git remote add space https://huggingface.co/spaces/your-username/riceguard
git push space main
```

Weights (`weights/*.pt`) are tracked via **Git LFS** so they upload correctly.
Set your `ROBOFLOW_API_KEY` as a **Secret** in Space Settings if you want the cloud model.

## Project Structure

```
rice_disease_detector/
├── app.py                  # Gradio UI — entry point
├── config.py               # Model registry, paths, env config
├── models/
│   ├── yolo_handler.py     # Local YOLOv8 inference + caching
│   └── roboflow_handler.py # Roboflow Workflows API + retry logic
├── utils/
│   └── image_utils.py      # Shared helpers (resize, table format)
├── weights/
│   ├── stage1_best.pt      # Stage 1 checkpoint (21.5 MB)
│   ├── stage2_best.pt      # Stage 2 checkpoint (21.5 MB)
│   └── finetune_best.pt    # Fine-tune checkpoint (optional)
├── projec_files/           # Stage 1 training artifacts (curves, CSV)
├── project_files/          # Stage 2 training artifacts (curves, CSV)
├── requirements.txt
├── .env.example
├── .gitattributes          # Git LFS config for *.pt files
└── README.md
```

## Training Background

Trained on ~7,400 labeled rice leaf images across 4 classes using YOLOv8s on Google Colab.
Training split across sessions due to free-tier GPU limits. Full write-up in the About tab of the app.

## License

MIT
