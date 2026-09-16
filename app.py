"""
RiceGuard — Rice Leaf Disease Diagnosis & Detection System
Powered by MobileViT-S Vision Transformer (97.4% Accuracy) & YOLOv8s Lesion Spot Detector
Hugging Face Spaces & ZeroGPU Ready
"""

import os
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from config import (
    CLASSIFIER_PATH,
    CLASSIFIER_RAW_CLASSES,
    DISEASE_INFO,
    TEST_BENCHMARK,
    CONFUSION_MATRIX_IMAGE,
    MODEL_REGISTRY,
    MODEL_LOOKUP,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_FRAME_STRIDE,
    MAX_VIDEO_FRAMES,
)
from models.classifier_handler import run_mobilevit_classification, load_classifier
from models.yolo_handler import run_yolo_on_image, run_yolo_on_video, preload_models
from utils.image_utils import resize_for_display, format_detections_table, format_class_counts

# ---------------------------------------------------------------------------
# Preload Models on Startup
# ---------------------------------------------------------------------------
print("[System] Initializing model engines...")
try:
    if os.path.exists(CLASSIFIER_PATH):
        print("[MobileViT] Preloading MobileViT-S Vision Transformer...")
        load_classifier(str(CLASSIFIER_PATH))
        print("[MobileViT] MobileViT-S ready (97.4% accuracy).")
except Exception as e:
    print(f"[MobileViT] Note on preload: {e}")

local_yolo_weights = [m.path for m in MODEL_REGISTRY if m.available]
if local_yolo_weights:
    print("[YOLO] Preloading YOLOv8 spot detectors...")
    preload_models(local_yolo_weights)
    print("[YOLO] YOLOv8 detectors ready.")

# ---------------------------------------------------------------------------
# Model choices for YOLO tab
# ---------------------------------------------------------------------------
YOLO_CHOICES = [(m.label, m.key) for m in MODEL_REGISTRY if m.available]
DEFAULT_YOLO = "v2"

# ---------------------------------------------------------------------------
# Custom CSS Theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
:root {
    --rice-green:   #10b981;
    --rice-amber:   #f59e0b;
    --rice-red:     #ef4444;
    --accent:       #34d399;
    --bg-dark:      #0b0f19;
    --card-bg:      #111827;
    --border:       rgba(255, 255, 255, 0.1);
}

body, .gradio-container { background: var(--bg-dark) !important; color: #f3f4f6; }

#rice-header {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 45%, #047857 100%);
    border-radius: 16px;
    padding: 26px 32px;
    margin-bottom: 12px;
    border: 1px solid rgba(52, 211, 153, 0.3);
    box-shadow: 0 10px 30px rgba(6, 78, 59, 0.35);
}
#rice-header h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #6ee7b7, #fbbf24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}
#rice-header p { color: #d1fae5; margin: 0; font-size: 1.05rem; }

.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: #9ca3af !important;
    transition: all 0.2s ease !important;
}
.tab-nav button.selected {
    color: #34d399 !important;
    border-bottom: 2px solid #34d399 !important;
}

.metric-pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
    margin-right: 6px;
}

.advisor-card {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 10px;
}

#classify-btn, #run-yolo-btn, #run-video-btn {
    background: linear-gradient(135deg, #059669, #0d9488) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
}
"""

# ---------------------------------------------------------------------------
# Tab 1: MobileViT Classification Handler
# ---------------------------------------------------------------------------
def diagnose_leaf(image):
    if image is None:
        return (
            None,
            {},
            "⚠️ Please upload a rice leaf photo or capture one using your webcam.",
        )

    top_raw_class, conf, prob_dict, err = run_mobilevit_classification(
        image,
        str(CLASSIFIER_PATH),
        CLASSIFIER_RAW_CLASSES,
    )

    if err:
        return None, {}, f"❌ Error during diagnosis: {err}"

    # Formatted probability distribution for gr.Label
    formatted_probs = {}
    for raw_cls, p in prob_dict.items():
        disp = DISEASE_INFO.get(raw_cls, {}).get("display_name", raw_cls)
        formatted_probs[disp] = round(p, 4)

    info = DISEASE_INFO.get(top_raw_class, {})
    disp_name = info.get("display_name", top_raw_class)
    sci_name = info.get("scientific_name", "")
    severity = info.get("severity", "Unknown")
    color = info.get("badge_color", "#10b981")
    symptoms = info.get("symptoms", "")
    treatment = info.get("treatment", "")

    # -----------------------------------------------------------------------
    # Dynamic Confidence Assessment (Avoid False Positives on Multi-Leaf/Wide Shots)
    # -----------------------------------------------------------------------
    CONFIDENCE_THRESHOLD = 0.60  # Below 60% indicates multi-leaf or noisy framing

    if conf < CONFIDENCE_THRESHOLD:
        advisory_html = f"""
        <div style="background: rgba(245, 158, 11, 0.12); border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.4); border-left: 6px solid #f59e0b; padding: 18px; margin-top: 12px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <h3 style="margin: 0; color: #fbbf24; font-size: 1.3rem;">
                    ⚠️ Low Confidence Diagnosis ({conf:.1%}) — Image Guidance Notice
                </h3>
                <span style="background: #f59e0b; color: #1e1b4b; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 0.85rem;">
                    Tentative Guess: {disp_name} ({conf:.1%})
                </span>
            </div>
            <div style="background: rgba(0, 0, 0, 0.3); padding: 14px; border-radius: 8px; margin-top: 12px;">
                <strong style="color: #fef08a; font-size: 0.95rem;">📸 Why Did This Happen?</strong>
                <p style="margin: 6px 0 10px 0; color: #fde68a; font-size: 0.92rem; line-height: 1.5;">
                    The model detected high uncertainty across multiple classes. This occurs when:
                </p>
                <ul style="margin: 0 0 12px 20px; padding: 0; color: #fde68a; font-size: 0.9rem; line-height: 1.6;">
                    <li><strong>Multiple overlapping leaves or full crop bush:</strong> The AI was trained specifically on individual, isolated leaf blades. Wide-angle shots with dozens of leaves, stems, and soil introduce mixed textures.</li>
                    <li><strong>Distant or out-of-focus photography:</strong> Subtle lesion patterns and vein structures cannot be resolved.</li>
                </ul>
                <div style="background: rgba(16, 185, 129, 0.15); border-left: 4px solid #10b981; padding: 10px 14px; border-radius: 6px;">
                    <strong style="color: #6ee7b7;">💡 Action Required:</strong>
                    <p style="margin: 4px 0 0 0; color: #e2e8f0; font-size: 0.92rem;">
                        Please provide a <strong>clear close-up photo of a single affected leaf blade</strong> for a reliable, definitive diagnosis before taking any chemical action.
                    </p>
                </div>
            </div>
        </div>
        """
    elif top_raw_class == "healthy":
        advisory_html = f"""
        <div style="background: rgba(16, 185, 129, 0.12); border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.35); border-left: 6px solid #10b981; padding: 18px; margin-top: 12px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <h3 style="margin: 0; color: #6ee7b7; font-size: 1.35rem;">
                    ✅ Healthy Rice Leaf (Disease-Free)
                </h3>
                <div>
                    <span style="background: #10b981; color: white; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 0.85rem;">
                        Confidence: {conf:.1%}
                    </span>
                    <span style="background: rgba(255,255,255,0.1); color: #e2e8f0; padding: 4px 12px; border-radius: 999px; font-weight: 600; font-size: 0.85rem; margin-left: 6px;">
                        Severity: None
                    </span>
                </div>
            </div>
            <p style="margin: 4px 0 12px 0; color: #94a3b8; font-style: italic; font-size: 0.95rem;">
                Oryza sativa (Optimal Leaf Health)
            </p>
            <div style="background: #0f172a; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                <strong style="color: #38bdf8;">Visual Assessment:</strong>
                <p style="margin: 4px 0 0 0; color: #cbd5e1; font-size: 0.92rem;">{symptoms}</p>
            </div>
            <div style="background: #0f172a; padding: 12px; border-radius: 8px;">
                <strong style="color: #4ade80;">Maintenance Plan:</strong>
                <p style="margin: 4px 0 0 0; color: #e2e8f0; font-size: 0.92rem;">
                    No chemical fungicide or bactericide treatment required. Maintain balanced nitrogen-potassium fertilizer levels and continue routine field monitoring.
                </p>
            </div>
        </div>
        """
    else:
        advisory_html = f"""
        <div style="background: #1e293b; border-radius: 12px; border-left: 6px solid {color}; padding: 18px; margin-top: 12px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <h3 style="margin: 0; color: #f8fafc; font-size: 1.4rem;">
                    {disp_name}
                </h3>
                <div>
                    <span style="background: {color}; color: white; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 0.85rem;">
                        Confidence: {conf:.1%}
                    </span>
                    <span style="background: rgba(255,255,255,0.1); color: #e2e8f0; padding: 4px 12px; border-radius: 999px; font-weight: 600; font-size: 0.85rem; margin-left: 6px;">
                        Severity: {severity}
                    </span>
                </div>
            </div>
            <p style="margin: 4px 0 12px 0; color: #94a3b8; font-style: italic; font-size: 0.95rem;">
                {sci_name}
            </p>
            <div style="background: #0f172a; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                <strong style="color: #38bdf8;">Visual Symptoms:</strong>
                <p style="margin: 4px 0 0 0; color: #cbd5e1; font-size: 0.92rem;">{symptoms}</p>
            </div>
            <div style="background: #0f172a; padding: 12px; border-radius: 8px;">
                <strong style="color: #4ade80;">Recommended Agronomic Treatment:</strong>
                <p style="margin: 4px 0 0 0; color: #e2e8f0; font-size: 0.92rem;">{treatment}</p>
            </div>
        </div>
        """

    return resize_for_display(image), formatted_probs, advisory_html


# ---------------------------------------------------------------------------
# Tab 2: YOLOv8 Lesion Spot Detection Handler
# ---------------------------------------------------------------------------
def detect_spots(image, model_key, conf):
    if image is None:
        return None, [], "⚠️ Please upload an image."

    model = MODEL_LOOKUP.get(model_key)
    if not model or not model.available:
        return image, [], "⚠️ Selected YOLO model weights unavailable."

    annotated, detections, err = run_yolo_on_image(image, model.path, conf=conf)
    if err:
        return image, [], f"❌ Error: {err}"

    table_rows = format_detections_table(detections)
    status = f"✅ Detected {len(detections)} spot lesion(s) with confidence ≥ {conf:.0%}."
    return resize_for_display(annotated), table_rows, status


# ---------------------------------------------------------------------------
# Tab 3: YOLOv8 Video Spot Detection Handler
# ---------------------------------------------------------------------------
def process_video(video_path, model_key, conf, frame_stride, progress=gr.Progress()):
    if video_path is None:
        return None, "⚠️ Please upload a video file."

    model = MODEL_LOOKUP.get(model_key)
    if not model or not model.available:
        return None, "⚠️ Selected YOLO model weights unavailable."

    progress(0, desc="Analyzing video frames...")
    out_path, class_counts, err = run_yolo_on_video(
        video_path,
        model.path,
        conf=conf,
        frame_stride=int(frame_stride),
        max_frames=MAX_VIDEO_FRAMES,
        progress=progress,
    )
    if err:
        return None, f"❌ Video processing error: {err}"

    summary = format_class_counts(class_counts)
    return out_path, summary


# ---------------------------------------------------------------------------
# Build Gradio Blocks Application
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    with gr.Blocks(title="🌾 RiceGuard — Rice Leaf Disease Diagnosis") as app_demo:

        # Header
        gr.HTML("""
        <div id="rice-header">
            <h1>🌾 RiceGuard — Rice Leaf Disease Diagnosis & Detection</h1>
            <p>
                Dual-Engine AI System: <strong>MobileViT-S Vision Transformer (97.4% Test Accuracy)</strong> for whole-leaf disease diagnosis
                alongside <strong>YOLOv8s</strong> for localized lesion spot detection and video streams.
            </p>
        </div>
        """)

        with gr.Tabs():

            # ════════════════════════════════════════════════════════════════
            # TAB 1: MobileViT-S High-Accuracy Whole-Leaf Diagnosis
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("🔬 Leaf Diagnosis (MobileViT-S 97.4%)"):
                gr.Markdown("""
                > **Flagship Model**: Powered by **MobileViT-S** (Mobile Vision Transformer with Multi-Head Self-Attention).
                > Models **global contextual dependencies across the entire leaf blade**, achieving **97.44% test accuracy** and **100% precision & recall on Bacterial Leaf Blight**.
                """)

                # Best practices banner
                gr.HTML("""
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(52, 211, 153, 0.25); border-radius: 10px; padding: 12px 18px; margin-bottom: 14px;">
                    <div style="font-weight: 700; color: #34d399; margin-bottom: 6px; font-size: 0.98rem; display: flex; align-items: center; gap: 6px;">
                        <span>📸</span> Photo Guidelines for Accurate Diagnosis:
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 0.9rem;">
                        <div style="background: rgba(16, 185, 129, 0.12); padding: 10px 14px; border-radius: 8px; color: #d1fae5; border-left: 3px solid #10b981;">
                            <strong>✅ DO:</strong> Upload a clear close-up of a <u>single leaf blade</u> centered in the frame.
                        </div>
                        <div style="background: rgba(239, 68, 68, 0.12); padding: 10px 14px; border-radius: 8px; color: #fecaca; border-left: 3px solid #ef4444;">
                            <strong>❌ AVOID:</strong> Wide-angle field shots with multiple overlapping plants, grain heads, or distant crops.
                        </div>
                    </div>
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=5):
                        diag_input = gr.Image(
                            label="Upload Rice Leaf Photo or Use Webcam",
                            sources=["upload", "webcam"],
                            type="pil",
                            height=340,
                        )
                        diag_btn = gr.Button("🔍 Run Full Diagnosis", elem_id="classify-btn", variant="primary")

                    with gr.Column(scale=5):
                        diag_label = gr.Label(num_top_classes=6, label="Predicted Disease Probability Distribution")
                        diag_advisory = gr.HTML(label="Agronomic Diagnosis & Action Plan")

                diag_btn.click(
                    fn=diagnose_leaf,
                    inputs=[diag_input],
                    outputs=[diag_input, diag_label, diag_advisory],
                )

            # ════════════════════════════════════════════════════════════════
            # TAB 2: YOLOv8 Lesion Spot Detector
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("🎯 Lesion Spot Detector (YOLOv8)"):
                gr.Markdown("""
                > **Localized Spot Analysis**: Uses **YOLOv8s** object detection to pinpoint and outline individual disease lesion spots on the leaf blade with bounding boxes.
                """)
                with gr.Row():
                    with gr.Column(scale=5):
                        spot_input = gr.Image(
                            label="Upload Leaf Image",
                            sources=["upload", "webcam"],
                            type="pil",
                            height=320,
                        )
                        with gr.Row():
                            spot_model_dd = gr.Dropdown(
                                label="Spot Detection Checkpoint",
                                choices=YOLO_CHOICES,
                                value=DEFAULT_YOLO,
                            )
                            spot_conf = gr.Slider(
                                label="Confidence Threshold",
                                minimum=0.10, maximum=0.90,
                                step=0.05, value=DEFAULT_CONF_THRESHOLD,
                            )
                        spot_btn = gr.Button("🎯 Detect Lesion Spots", elem_id="run-yolo-btn", variant="primary")

                    with gr.Column(scale=5):
                        spot_output = gr.Image(label="Annotated Spot Detections", height=320)
                        spot_status = gr.Markdown()
                        spot_table = gr.Dataframe(
                            label="Detected Lesion Spots",
                            headers=["#", "Class", "Confidence", "Bounding Box Coordinates"],
                            interactive=False,
                        )

                spot_btn.click(
                    fn=detect_spots,
                    inputs=[spot_input, spot_model_dd, spot_conf],
                    outputs=[spot_output, spot_table, spot_status],
                )

            # ════════════════════════════════════════════════════════════════
            # TAB 3: Video Stream Lesion Detection
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("🎬 Video Lesion Stream"):
                gr.Markdown(f"""
                > Upload a field video clip. The detector processes frames up to **{MAX_VIDEO_FRAMES} frames**.
                > Use **Frame Stride** to speed up video processing (Stride 2 = every 2nd frame).
                """)
                with gr.Row():
                    with gr.Column(scale=5):
                        vid_input = gr.Video(label="Upload Field Video", sources=["upload"])
                        with gr.Row():
                            vid_model_dd = gr.Dropdown(
                                label="Model Checkpoint",
                                choices=YOLO_CHOICES,
                                value=DEFAULT_YOLO,
                            )
                            vid_conf = gr.Slider(
                                label="Confidence",
                                minimum=0.10, maximum=0.90,
                                step=0.05, value=DEFAULT_CONF_THRESHOLD,
                            )
                        vid_stride = gr.Slider(
                            label="Frame Stride (Higher = Faster)",
                            minimum=1, maximum=10,
                            step=1, value=DEFAULT_FRAME_STRIDE,
                        )
                        vid_btn = gr.Button("▶ Process Video", elem_id="run-video-btn", variant="primary")

                    with gr.Column(scale=5):
                        vid_output = gr.Video(label="Annotated Video Output")
                        vid_summary = gr.Markdown()

                vid_btn.click(
                    fn=process_video,
                    inputs=[vid_input, vid_model_dd, vid_conf, vid_stride],
                    outputs=[vid_output, vid_summary],
                )

            # ════════════════════════════════════════════════════════════════
            # TAB 4: Benchmark & Confusion Matrix
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("📊 Benchmark & Confusion Matrix"):
                gr.Markdown(f"""
                ## MobileViT-S Test Set Evaluation

                - **Overall Test Accuracy:** **`{TEST_BENCHMARK['test_accuracy'] * 100:.2f}%`**
                - **Macro F1-Score:** **`{TEST_BENCHMARK['test_macro_f1'] * 100:.2f}%`**
                - **Total Test Samples:** `234` images across 6 classes
                """)

                with gr.Row():
                    with gr.Column(scale=6):
                        if os.path.exists(CONFUSION_MATRIX_IMAGE):
                            gr.Image(
                                value=str(CONFUSION_MATRIX_IMAGE),
                                label="Test Set Confusion Matrix (MobileViT-S)",
                                interactive=False,
                            )
                        else:
                            gr.Markdown("Confusion matrix plot located in `assets/mobilevit_confusion_matrix.png`.")

                    with gr.Column(scale=4):
                        gr.Markdown("""
                        ### Per-Class Test Performance

                        | Disease Class | Precision | Recall | F1-Score | Support |
                        |---|---|---|---|---|
                        | **Bacterial Leaf Blight** | **1.000** | **1.000** | **1.000** | 36 |
                        | **Narrow Brown Spot** | **1.000** | **1.000** | **1.000** | 39 |
                        | **Leaf Scald** | **1.000** | **1.000** | **1.000** | 34 |
                        | **Healthy Leaf** | 0.976 | 1.000 | 0.988 | 41 |
                        | **Brown Spot** | 0.930 | 0.952 | 0.941 | 42 |
                        | **Leaf Blast** | 0.950 | 0.905 | 0.927 | 42 |
                        | **Macro Average** | **0.976** | **0.976** | **0.976** | 234 |
                        """)

                gr.Markdown("""
                ---
                ### 💡 Technical Deep Dive: Why Vision Transformers Solved the Blight Problem

                In earlier training stages with local convolution-only detectors (YOLOv8s), **Bacterial Leaf Blight** was the weakest link:
                - Local CNN kernels primarily extract localized edge and texture gradients within small spatial receptive fields (e.g. 3×3 or 5×5 pixels).
                - In severe Blight infections, broad water-soaked lesions across the leaf edge mimic plain background or normal leaf variations, causing convolution-only models to confuse disease symptoms with background noise.
                - **The MobileViT Solution:** MobileViT replaces local convolutions in the deeper stages with **Transformer Blocks equipped with Multi-Head Self-Attention**. This allows every spatial patch on the leaf to attend to every other patch across the entire leaf blade. As a result, the model captures **global lesion progression patterns, vein margins, and whole-leaf chlorosis gradients**, achieving **100% precision and 100% recall on Bacterial Leaf Blight**.
                """)

        # Footer
        gr.HTML("""
        <div style="text-align:center; padding: 24px; color: #9ca3af; font-size:0.85rem;">
            🌾 RiceGuard · MobileViT-S Vision Transformer & YOLOv8s Lesion Spot Detector ·
            Trained on PyTorch & Ultralytics · Deployed with ZeroGPU on Hugging Face Spaces
        </div>
        """)

    return app_demo


# ---------------------------------------------------------------------------
# Module Export for Hugging Face Spaces & Local Execution
# ---------------------------------------------------------------------------
demo = build_app()
demo.queue()
app = demo

if __name__ == "__main__":
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

    demo.launch(**launch_kwargs)