"""
Shared helpers for image manipulation, drawing detection boxes, and formatting results for Gradio.
"""

from __future__ import annotations
import cv2
import numpy as np
from PIL import Image

# Fixed color per class for visual consistency across all models
CLASS_COLORS = {
    "Blast": (239, 159, 39),      # amber
    "Blight": (226, 75, 74),      # red
    "Brownspot": (211, 90, 48),   # coral
    "Healthy": (99, 153, 34),     # green
}
DEFAULT_COLOR = (100, 100, 100)


def resize_for_display(img: Image.Image | np.ndarray, max_w: int = 1000) -> Image.Image:
    """Proportionally resize an image so its width does not exceed max_w."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    if img.width > max_w:
        ratio = max_w / float(img.width)
        new_h = int(float(img.height) * ratio)
        return img.resize((max_w, new_h), Image.Resampling.LANCZOS)
    return img


def format_detections_table(detections: list[dict]) -> list[list]:
    """Convert detection dicts into rows for gr.Dataframe."""
    rows = []
    for idx, d in enumerate(detections, 1):
        cls_name = d.get("class_name", "Unknown")
        conf = f"{d.get('confidence', 0.0):.1%}"
        coords = f"({d.get('x1', 0)}, {d.get('y1', 0)}) -> ({d.get('x2', 0)}, {d.get('y2', 0)})"
        rows.append([idx, cls_name, conf, coords])
    return rows


def format_class_counts(class_counts: dict[str, int]) -> str:
    """Format per-class counts for video summary."""
    if not class_counts:
        return "ℹ️ No lesions detected in the processed video frames."

    total = sum(class_counts.values())
    lines = [f"**Total Lesion Detections:** `{total}`\n"]
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = (count / total) * 100 if total > 0 else 0
        lines.append(f"- **{cls_name}**: {count} ({pct:.1f}%)")
    return "\n".join(lines)


def draw_detections(image_path: str, detections: list[dict]) -> np.ndarray:
    """Draws bounding boxes + labels on the source image."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    for det in detections:
        box = det.get("box_xyxy", [det.get("x1", 0), det.get("y1", 0), det.get("x2", 0), det.get("y2", 0)])
        x1, y1, x2, y2 = [int(v) for v in box]
        color = CLASS_COLORS.get(det["class_name"], DEFAULT_COLOR)
        label = f"{det['class_name']} {det['confidence']:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(
            image, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return image[:, :, ::-1]  # BGR -> RGB