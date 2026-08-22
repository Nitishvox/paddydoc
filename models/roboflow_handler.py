"""
Roboflow hosted-model inference handler.

Calls the Roboflow Workflows API with retry/timeout logic and returns
the same detection structure as the local YOLO handler so the UI can
treat all models uniformly.
"""

import base64
import io
import time
from typing import Optional

import requests
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_roboflow_api(
    pil_image: Image.Image,
    api_key: str,
    workspace: str,
    workflow_id: str,
    conf: float = 0.4,
    timeout: int = 30,
    max_retries: int = 2,
) -> tuple[Optional[Image.Image], list[dict], Optional[str]]:
    """
    Run inference via Roboflow Workflows API.

    Returns
    -------
    annotated_image : PIL.Image or None
    detections      : list of {class_name, confidence, x1, y1, x2, y2}
    error           : str or None
    """
    if not api_key or api_key.strip() == "":
        return None, [], "No Roboflow API key provided. Set ROBOFLOW_API_KEY in .env or enter it in the UI."

    img_b64 = _image_to_b64(pil_image)
    url = (
        f"https://serverless.roboflow.com/{workspace}/{workflow_id}"
        f"?api_key={api_key}"
    )
    payload = {
        "inputs": {
            "image": {"type": "base64", "value": img_b64},
        }
    }

    last_error: str = ""
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            detections = _parse_roboflow_response(data, conf)
            annotated  = _draw_detections(pil_image.copy(), detections)
            return annotated, detections, None

        except requests.exceptions.Timeout:
            last_error = "Request timed out."
        except requests.exceptions.HTTPError as e:
            last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            break  # don't retry 4xx errors
        except Exception as e:
            last_error = str(e)

        if attempt < max_retries:
            time.sleep(2 ** attempt)  # exponential back-off

    return None, [], f"Roboflow API error after {max_retries + 1} attempts: {last_error}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_b64(pil_image: Image.Image) -> str:
    """Encode a PIL image as a JPEG base64 string."""
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_roboflow_response(data: dict, conf_threshold: float) -> list[dict]:
    """
    Parse Roboflow Workflows JSON response into a unified detection list.
    Handles both legacy predictions and modern workflow output formats.
    """
    detections: list[dict] = []

    # Try to find predictions in common locations
    outputs = data.get("outputs", [data])
    for output in outputs:
        preds = (
            output.get("predictions", {}).get("predictions")
            or output.get("predictions")
            or output.get("detections")
            or []
        )
        if isinstance(preds, dict):
            preds = preds.get("predictions", [])

        for p in preds:
            conf = float(p.get("confidence", 0))
            if conf < conf_threshold:
                continue

            # Roboflow uses centre-x/y/width/height
            cx = float(p.get("x", 0))
            cy = float(p.get("y", 0))
            w  = float(p.get("width", 0))
            h  = float(p.get("height", 0))

            detections.append({
                "class_name": p.get("class", "Unknown"),
                "confidence": round(conf, 3),
                "x1": int(cx - w / 2),
                "y1": int(cy - h / 2),
                "x2": int(cx + w / 2),
                "y2": int(cy + h / 2),
            })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


# Colour map for drawing
_COLORS = {
    "Blast":     (255, 68,  68),
    "Blight":    (255, 140, 0),
    "Brownspot": (255, 215, 0),
    "Healthy":   (0,   200, 81),
}
_DEFAULT_COLOR = (160, 160, 255)


def _draw_detections(img: Image.Image, detections: list[dict]) -> Image.Image:
    """Draw bounding boxes + labels on a PIL image."""
    draw = ImageDraw.Draw(img)

    for det in detections:
        name  = det["class_name"]
        conf  = det["confidence"]
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        color = _COLORS.get(name, _DEFAULT_COLOR)

        # Box (2 px border)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Label background + text
        label = f"{name} {conf:.0%}"
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        bbox = draw.textbbox((x1, y1 - 18), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 18), label, fill="white", font=font)

    return img
