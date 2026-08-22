"""
RiceGuard — Rice Leaf Disease Detector
Gradio demo app  |  HF Spaces ready
"""

import os
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from config import (
    MODEL_REGISTRY, MODEL_LOOKUP,
    CLASS_NAMES, CLASS_COLORS_HEX,
    ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_WORKFLOW_ID,
    DEFAULT_CONF_THRESHOLD, DEFAULT_FRAME_STRIDE, MAX_VIDEO_FRAMES,
    MAX_RETRIES, REQUEST_TIMEOUT_SECONDS,
)
from models.yolo_handler import run_yolo_on_image, run_yolo_on_video, preload_models
from models.roboflow_handler import run_roboflow_api
from utils.image_utils import (
    resize_for_display, format_detections_table, format_class_counts,
)

# ---------------------------------------------------------------------------
# Build dropdown choices  (only show unavailable models with a warning label)
# ---------------------------------------------------------------------------
def _model_choices() -> list[tuple[str, str]]:
    choices = []
    for m in MODEL_REGISTRY:
        if m.kind == "local" and not m.available:
            choices.append((m.label + "  ⚠️ weights missing", m.key))
        else:
            choices.append((m.label, m.key))
    return choices


MODEL_CHOICES = _model_choices()
DEFAULT_MODEL = "v2"   # best local model


# ---------------------------------------------------------------------------
# Custom CSS — dark premium theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ── Global ── */
:root {
    --rice-green:   #00C851;
    --rice-yellow:  #FFD700;
    --rice-red:     #FF4444;
    --rice-orange:  #FF8C00;
    --accent:       #6ee7b7;
    --bg-dark:      #0f1117;
    --bg-card:      rgba(255,255,255,0.04);
    --border:       rgba(255,255,255,0.10);
}

body, .gradio-container { background: var(--bg-dark) !important; }

/* ── Header ── */
#rice-header {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 40%, #0f766e 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 4px;
    box-shadow: 0 8px 32px rgba(0,200,81,0.15);
    border: 1px solid rgba(110,231,183,0.20);
}
#rice-header h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #6ee7b7, #fbbf24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}
#rice-header p { color: #a7f3d0; margin: 0; font-size: 1rem; }

/* ── Tabs ── */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: #9ca3af !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
}
.tab-nav button.selected {
    color: #6ee7b7 !important;
    border-bottom: 2px solid #6ee7b7 !important;
}

/* ── Cards ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
}

/* ── Blight warning ── */
#blight-warning {
    background: linear-gradient(90deg, rgba(255,140,0,0.15), rgba(255,68,68,0.12));
    border: 1px solid rgba(255,140,0,0.4);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.9rem;
    color: #fed7aa;
}

/* ── Detection table ── */
#det-table table { font-size: 0.88rem !important; }
#det-table thead th { color: #6ee7b7 !important; font-weight: 700 !important; }

/* ── Buttons ── */
#run-btn, #run-video-btn, #compare-btn {
    background: linear-gradient(135deg, #059669, #0d9488) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
#run-btn:hover, #run-video-btn:hover, #compare-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(5,150,105,0.4) !important;
}

/* ── Model info badge ── */
#model-info {
    background: rgba(110,231,183,0.08);
    border-left: 3px solid #6ee7b7;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 0.88rem;
    color: #a7f3d0;
    min-height: 48px;
}

/* ── Status / error ── */
.status-ok  { color: var(--rice-green)  !important; font-weight: 600; }
.status-err { color: var(--rice-red)    !important; font-weight: 600; }

/* ── About metrics table ── */
#metrics-table table th { color: var(--accent) !important; }
"""


# ---------------------------------------------------------------------------
# Core inference helpers
# ---------------------------------------------------------------------------

def _run_local(image, model_key, conf):
    model = MODEL_LOOKUP[model_key]
    if not model.available:
        return None, [], f"⚠️ Weights for '{model.label}' are missing. Check weights/ folder."
    annotated, detections, err = run_yolo_on_image(image, model.path, conf=conf)
    return annotated, detections, err


def _run_roboflow(image, api_key_override, conf):
    key = api_key_override.strip() if api_key_override else ROBOFLOW_API_KEY
    annotated, detections, err = run_roboflow_api(
        image, key, ROBOFLOW_WORKSPACE, ROBOFLOW_WORKFLOW_ID,
        conf=conf, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=MAX_RETRIES,
    )
    return annotated, detections, err


def _blight_warning(detections: list[dict]) -> str:
    has_blight = any(d["class_name"] == "Blight" for d in detections)
    if has_blight:
        return (
            "⚠️ **Blight detected** — Note: This model has a known weakness for Blight. "
            "It only catches ~37 % of real Blight cases and can confuse lesions with background. "
            "A negative result does **not** rule out Blight."
        )
    return ""


# ---------------------------------------------------------------------------
# Tab 1 — Image Detection
# ---------------------------------------------------------------------------

def detect_image(image, model_key, conf, rf_api_key):
    if image is None:
        return None, [], "", "⚠️ Please upload an image or use your webcam."

    model = MODEL_LOOKUP.get(model_key)
    if model is None:
        return None, [], "", "⚠️ Unknown model selected."

    if model.kind == "roboflow":
        annotated, detections, err = _run_roboflow(image, rf_api_key, conf)
    else:
        annotated, detections, err = _run_local(image, model_key, conf)

    if err:
        return image, [], "", f"❌ {err}"

    table_rows  = format_detections_table(detections)
    blight_warn = _blight_warning(detections)
    status      = f"✅ {len(detections)} detection(s) found."

    annotated = resize_for_display(annotated)
    return annotated, table_rows, blight_warn, status


def update_model_info(model_key):
    m = MODEL_LOOKUP.get(model_key)
    if not m:
        return ""
    avail = "✅ Ready" if (m.kind == "roboflow" or m.available) else "⚠️ Weights missing"
    if m.kind == "roboflow":
        return f"**{m.label}** — {m.description}  |  Status: {avail}"
    return (
        f"**mAP50:** {m.map50:.3f}  ·  "
        f"**Epochs:** {m.epochs}  ·  "
        f"**Res:** {m.imgsz}px  ·  "
        f"Status: {avail}"
    )


# ---------------------------------------------------------------------------
# Tab 2 — Video Detection
# ---------------------------------------------------------------------------

def detect_video(video_path, model_key, conf, frame_stride, progress=gr.Progress()):
    if video_path is None:
        return None, "⚠️ Please upload a video file."

    model = MODEL_LOOKUP.get(model_key)
    if model is None or not model.available:
        return None, "⚠️ Selected model weights are not available."

    progress(0, desc="Starting video processing…")

    out_path, class_counts, err = run_yolo_on_video(
        video_path,
        model.path,
        conf=conf,
        frame_stride=int(frame_stride),
        max_frames=MAX_VIDEO_FRAMES,
        progress=progress,
    )

    if err:
        return None, f"❌ {err}"

    summary = format_class_counts(class_counts)
    return out_path, summary


# ---------------------------------------------------------------------------
# Tab 3 — Model Comparison
# ---------------------------------------------------------------------------

def compare_models(image, conf):
    if image is None:
        return None, None, None, "⚠️ Please upload an image first."

    results = {}
    for key in ("v1", "v2", "finetune"):
        m = MODEL_LOOKUP[key]
        if m.available:
            ann, dets, err = _run_local(image, key, conf)
            results[key] = (resize_for_display(ann) if ann else image, len(dets), err)
        else:
            results[key] = (image, 0, "weights missing")

    def _caption(key, count, err):
        m = MODEL_LOOKUP[key]
        if err and "weights missing" in err:
            return f"{m.label}\n⚠️ Weights not available"
        return f"{m.label}\n{count} detection(s)  |  mAP50: {m.map50}"

    img1, c1, e1 = results["v1"]
    img2, c2, e2 = results["v2"]
    img3, c3, e3 = results["finetune"]

    status = "✅ Comparison complete."
    return img1, img2, img3, status


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="🌾 RiceGuard — Rice Disease Detector",
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="rice-header">
            <h1>🌾 RiceGuard</h1>
            <p>
                AI-powered rice leaf disease detection using YOLOv8.
                Detects <strong>Blast</strong>, <strong>Blight</strong>,
                <strong>Brownspot</strong>, and <strong>Healthy</strong> leaves.
                Upload an image, take a photo, or run detection on a video clip.
            </p>
        </div>
        """)

        with gr.Tabs():

            # ════════════════════════════════════════════════════════════════
            # TAB 1 — Image Detection
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("📷  Image Detection"):
                with gr.Row():
                    # Left column — inputs
                    with gr.Column(scale=5):
                        img_input = gr.Image(
                            label="Upload image or use webcam",
                            sources=["upload", "webcam"],
                            type="pil",
                            height=340,
                        )
                        with gr.Row():
                            model_dd = gr.Dropdown(
                                label="Model",
                                choices=MODEL_CHOICES,
                                value=DEFAULT_MODEL,
                            )
                            conf_slider = gr.Slider(
                                label="Confidence threshold",
                                minimum=0.10, maximum=0.90,
                                step=0.05, value=DEFAULT_CONF_THRESHOLD,
                            )
                        rf_key_box = gr.Textbox(
                            label="Roboflow API Key (only needed for cloud model)",
                            placeholder="Paste your key here, or set ROBOFLOW_API_KEY in .env",
                            type="password",
                            value=ROBOFLOW_API_KEY,
                        )
                        model_info_md = gr.Markdown(
                            value=update_model_info(DEFAULT_MODEL),
                            elem_id="model-info",
                        )
                        run_btn = gr.Button("🔍  Run Detection", elem_id="run-btn", variant="primary")

                    # Right column — outputs
                    with gr.Column(scale=5):
                        img_output   = gr.Image(label="Annotated Result", height=340)
                        status_label = gr.Markdown(value="", elem_id="status-md")
                        blight_warn  = gr.Markdown(value="", elem_id="blight-warning")
                        det_table    = gr.Dataframe(
                            label="Detections",
                            headers=["#", "Class", "Confidence", "Bounding Box"],
                            elem_id="det-table",
                            interactive=False,
                        )

                # Wire events
                model_dd.change(
                    fn=update_model_info,
                    inputs=model_dd,
                    outputs=model_info_md,
                )
                run_btn.click(
                    fn=detect_image,
                    inputs=[img_input, model_dd, conf_slider, rf_key_box],
                    outputs=[img_output, det_table, blight_warn, status_label],
                )
                img_input.change(          # clear results when a new image is loaded
                    fn=lambda _: (None, [], "", ""),
                    inputs=img_input,
                    outputs=[img_output, det_table, blight_warn, status_label],
                )

            # ════════════════════════════════════════════════════════════════
            # TAB 2 — Video Detection
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("🎬  Video Detection"):
                gr.Markdown(
                    f"> Processes up to **{MAX_VIDEO_FRAMES} frames**. "
                    "Use **Frame Stride** to skip frames and speed things up. "
                    "Stride 2 = process every 2nd frame (2× faster, half the detections)."
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        video_input = gr.Video(
                            label="Upload video",
                            sources=["upload"],
                        )
                        with gr.Row():
                            vid_model_dd = gr.Dropdown(
                                label="Model",
                                choices=[c for c in MODEL_CHOICES if "Roboflow" not in c[0]],
                                value=DEFAULT_MODEL,
                            )
                            vid_conf = gr.Slider(
                                label="Confidence",
                                minimum=0.10, maximum=0.90,
                                step=0.05, value=DEFAULT_CONF_THRESHOLD,
                            )
                        stride_slider = gr.Slider(
                            label="Frame Stride (higher = faster)",
                            minimum=1, maximum=10,
                            step=1, value=DEFAULT_FRAME_STRIDE,
                        )
                        run_video_btn = gr.Button("▶  Process Video", elem_id="run-video-btn", variant="primary")

                    with gr.Column(scale=5):
                        video_output = gr.Video(label="Annotated Video")
                        vid_summary  = gr.Markdown(label="Detection Summary")

                run_video_btn.click(
                    fn=detect_video,
                    inputs=[video_input, vid_model_dd, vid_conf, stride_slider],
                    outputs=[video_output, vid_summary],
                )

            # ════════════════════════════════════════════════════════════════
            # TAB 3 — Model Comparison
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("⚖️  Compare Models"):
                gr.Markdown(
                    "Upload one image and see all three local models' outputs side by side. "
                    "Great for seeing how each training stage changed the results."
                )
                with gr.Row():
                    cmp_input = gr.Image(
                        label="Upload Image",
                        sources=["upload", "webcam"],
                        type="pil",
                        scale=3,
                    )
                    with gr.Column(scale=2):
                        cmp_conf   = gr.Slider(
                            label="Confidence",
                            minimum=0.10, maximum=0.90,
                            step=0.05, value=DEFAULT_CONF_THRESHOLD,
                        )
                        compare_btn = gr.Button("⚖️  Compare All Models", elem_id="compare-btn", variant="primary")
                        cmp_status  = gr.Markdown()

                with gr.Row():
                    out_v1 = gr.Image(
                        label="Stage 1 — 65 ep · mAP50 0.557",
                        show_label=True,
                    )
                    out_v2 = gr.Image(
                        label="Stage 2 — 100 ep · mAP50 0.569  ★ Best",
                        show_label=True,
                    )
                    out_ft = gr.Image(
                        label="Fine-tune — Experimental · mAP50 0.483",
                        show_label=True,
                    )

                compare_btn.click(
                    fn=compare_models,
                    inputs=[cmp_input, cmp_conf],
                    outputs=[out_v1, out_v2, out_ft, cmp_status],
                )

            # ════════════════════════════════════════════════════════════════
            # TAB 4 — About / Training History
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("📊  About & Training"):
                gr.Markdown("""
## About RiceGuard

RiceGuard detects four classes of rice leaf condition using YOLOv8:

| Class | Colour | Description |
|---|---|---|
| 🔴 Blast | Red | Lesions caused by *Magnaporthe oryzae* |
| 🟠 Blight | Orange | Bacterial leaf blight (*Xanthomonas oryzae*) |
| 🟡 Brownspot | Yellow | Brown spot disease (*Bipolaris oryzicola*) |
| 🟢 Healthy | Green | No visible disease symptoms |

---

## Training Journey

### Stage 1 — Baseline (65 epochs, 640 px)
- Model: YOLOv8s pretrained on COCO
- Dataset: ~7,400 labelled images
- Result: **mAP50 = 0.557**

### Stage 2 — Higher Resolution (35 more epochs, 832 px)
- Continued from Stage 1 checkpoint
- Resolution increased to 832×832 to improve small-lesion detection
- Result: **mAP50 = 0.569** ← best model

### Fine-tune Experiment (15 epochs — failed)
- Attempted aggressive augmentation + lower LR to fix Blight recall
- Result: **mAP50 = 0.483** — every class got worse
- Lesson: fine-tuning a converged model needs much gentler augmentation

---

## ⚠️ Known Limitation — Blight Detection

Even the best model only catches **~37 % of real Blight cases**.
Disease lesions are visually similar to plain background, a documented
hard problem in rice disease detection research. Fixing it requires
architectural changes (e.g., attention mechanisms), not just more training.

**Do not rely on this model for clinical or field decisions.**

---

## 📈 Model Metrics

| Model | Epochs | Resolution | mAP50 | Precision | Recall |
|---|---|---|---|---|---|
| Stage 1 | 65 | 640 px | 0.557 | 56.6 % | 51.2 % |
| Stage 2 ★ | 100 | 832 px | **0.569** | 56.1 % | 52.1 % |
| Fine-tune | 115 | 832 px | 0.483 | — | — |
| Roboflow RF-DETR | — | — | 0.527 | 57.9 % | 58.2 % |

---

## 🚀 Deployment
This app runs on **Hugging Face Spaces** (free tier).
Source code: [GitHub](https://github.com)
""")

        # ── Footer ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; padding: 20px; color: #6b7280; font-size:0.85rem;">
            RiceGuard · YOLOv8 Rice Leaf Disease Detector ·
            Built with <a href="https://gradio.app" style="color:#6ee7b7">Gradio</a> ·
            Model weights trained on Google Colab
        </div>
        """)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Preload local models to ensure instant inference with zero first-request lag
    local_weights = [m.path for m in MODEL_REGISTRY if m.kind == "local" and m.available]
    if local_weights:
        print("[YOLO] Preloading local YOLOv8 model weights...")
        preload_models(local_weights)
        print("[YOLO] Models preloaded and ready!")

    app = build_app()
    app.queue()  # Enable Gradio queue for reliable websocket & long request handling

    port_env = os.environ.get("PORT")
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "show_error": True,
        "css": CUSTOM_CSS,
        "theme": gr.themes.Base(
            primary_hue="emerald",
            neutral_hue="zinc",
            font=gr.themes.GoogleFont("Inter"),
        ),
    }
    if port_env:
        launch_kwargs["server_port"] = int(port_env)

    app.launch(**launch_kwargs)
