"""
YOLOv8 local inference handler.

Features:
- Thread-safe in-memory model cache (no disk re-loads when switching models)
- Image inference → annotated PIL image + structured detection list
- Video inference → annotated output video file + per-class counts
"""

import threading
import tempfile
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

_model_cache: dict = {}
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

def load_model(path: str):
    """Return a cached YOLO instance, loading from disk only on first call."""
    from ultralytics import YOLO  # lazy import

    with _cache_lock:
        if path not in _model_cache:
            m = YOLO(path)
            # Warm up model with a tiny 64x64 dummy pass to initialize weights in memory
            try:
                dummy = Image.new("RGB", (64, 64))
                m.predict(dummy, verbose=False)
            except Exception:
                pass
            _model_cache[path] = m
    return _model_cache[path]


def preload_models(paths: list[str]):
    """Preload models in background or startup to eliminate first-request delay."""
    for p in paths:
        if p and os.path.exists(p) and os.path.getsize(p) > 0:
            try:
                load_model(p)
            except Exception as e:
                print(f"[YOLO] Warning: Failed to preload {p}: {e}")


# ---------------------------------------------------------------------------
# Image inference
# ---------------------------------------------------------------------------

def run_yolo_on_image(
    pil_image: Image.Image,
    model_path: str,
    conf: float = 0.4,
) -> tuple[Image.Image, list[dict], Optional[str]]:
    """
    Run YOLOv8 on a PIL image.

    Returns
    -------
    annotated_image : PIL.Image
    detections      : list of {class_name, confidence, x1, y1, x2, y2}
    error           : str or None
    """
    try:
        model = load_model(model_path)
        results = model.predict(pil_image, conf=conf, verbose=False)[0]
        detections = _parse_results(results)
        annotated = _results_to_pil(results)
        return annotated, detections, None
    except Exception as exc:
        return pil_image, [], str(exc)


# ---------------------------------------------------------------------------
# Video inference
# ---------------------------------------------------------------------------

def run_yolo_on_video(
    video_path: str,
    model_path: str,
    conf: float = 0.4,
    frame_stride: int = 2,
    max_frames: int = 500,
    progress=None,          # optional Gradio gr.Progress() callback
) -> tuple[Optional[str], dict, Optional[str]]:
    """
    Process a video file frame by frame and write annotated output.

    Returns
    -------
    out_video_path : str or None
    class_counts   : dict {class_name: count}
    error          : str or None
    """
    import cv2

    try:
        model = load_model(model_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, {}, "Could not open video file."

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output FPS reduced proportionally to frame_stride
        out_fps = max(fps / frame_stride, 1.0)

        # Write to a temp mp4
        suffix = ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        out_path = tmp.name
        tmp.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, out_fps, (width, height))

        class_counts: dict[str, int] = {}
        processed = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if processed >= max_frames:
                break

            if frame_idx % frame_stride == 0:
                results = model.predict(frame, conf=conf, verbose=False)[0]
                annotated_frame = results.plot()
                writer.write(annotated_frame)

                # Accumulate class counts
                for det in _parse_results(results):
                    name = det["class_name"]
                    class_counts[name] = class_counts.get(name, 0) + 1

                processed += 1
                if progress is not None and total > 0:
                    progress(frame_idx / total, desc=f"Frame {frame_idx}/{total}")

            frame_idx += 1

        cap.release()
        writer.release()
        return out_path, class_counts, None

    except Exception as exc:
        return None, {}, str(exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_results(results) -> list[dict]:
    """Convert ultralytics Results object to a plain list of dicts."""
    detections = []
    if results.boxes is None:
        return detections

    names = results.names  # {0: 'Blast', 1: 'Blight', ...}
    boxes  = results.boxes

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf   = float(boxes.conf[i].item())
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        detections.append({
            "class_name": names.get(cls_id, str(cls_id)),
            "confidence": round(conf, 3),
            "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2),
        })

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def _results_to_pil(results) -> Image.Image:
    """Convert ultralytics annotated result to PIL Image."""
    annotated_bgr = results.plot()              # numpy BGR array
    annotated_rgb = annotated_bgr[..., ::-1]   # BGR → RGB
    return Image.fromarray(annotated_rgb)
