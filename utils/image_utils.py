"""
Shared image utilities for RiceGuard.
"""

from PIL import Image


def resize_for_display(img: Image.Image, max_width: int = 900) -> Image.Image:
    """Proportionally resize an image so its width doesn't exceed max_width."""
    if img.width <= max_width:
        return img
    ratio  = max_width / img.width
    new_h  = int(img.height * ratio)
    return img.resize((max_width, new_h), Image.LANCZOS)


def pil_to_np(img: Image.Image):
    """Convert PIL Image to numpy RGB array."""
    import numpy as np
    return np.array(img.convert("RGB"))


def np_to_pil(arr) -> Image.Image:
    """Convert numpy array to PIL Image."""
    import numpy as np
    arr = np.asarray(arr)
    if arr.dtype != "uint8":
        arr = arr.astype("uint8")
    return Image.fromarray(arr)


def format_detections_table(detections: list[dict]) -> list[list]:
    """Convert detection dicts to a list-of-lists for gr.Dataframe."""
    rows = []
    for i, d in enumerate(detections, 1):
        rows.append([
            i,
            d["class_name"],
            f"{d['confidence']:.1%}",
            f"({d['x1']}, {d['y1']}) → ({d['x2']}, {d['y2']})",
        ])
    return rows


def format_class_counts(counts: dict) -> str:
    """Format per-class detection counts as a readable string."""
    if not counts:
        return "No detections found."
    total = sum(counts.values())
    lines = [f"**Total detections: {total}**"]
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"- {cls}: {cnt}")
    return "\n".join(lines)
