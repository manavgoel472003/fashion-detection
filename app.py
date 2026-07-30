from __future__ import annotations

import base64
import io
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("FASHION_MODEL_PATH", ROOT / "models" / "yolov8n-clothing-detection.pt"))
DEFAULT_CONFIDENCE = float(os.getenv("FASHION_CONFIDENCE", "0.35"))
DEFAULT_IMAGE_SIZE = int(os.getenv("FASHION_IMAGE_SIZE", "640"))

app = FastAPI(
    title="Fashion Detection Demo",
    description="Detect clothing, shoes, bags, and accessories in images.",
    version="1.0.0",
)

_model: YOLO | None = None
_model_lock = threading.RLock()


class DetectionRequest(BaseModel):
    image: str = Field(description="Base64 image or a data URL")
    confidence: float = Field(default=DEFAULT_CONFIDENCE, ge=0.05, le=0.95)


def get_model() -> YOLO:
    global _model
    with _model_lock:
        if _model is None:
            if not MODEL_PATH.is_file():
                raise RuntimeError(f"Model not found: {MODEL_PATH}")
            _model = YOLO(str(MODEL_PATH))
        return _model


def decode_image(value: str) -> np.ndarray:
    try:
        encoded = value.split(",", 1)[-1]
        raw = base64.b64decode(encoded, validate=True)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.asarray(image)[:, :, ::-1].copy()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image") from exc


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": MODEL_PATH.name,
        "model_available": MODEL_PATH.is_file(),
        "classes": ["accessories", "bags", "clothing", "shoes"],
    }


@app.post("/api/detect")
def detect(payload: DetectionRequest) -> dict[str, Any]:
    image = decode_image(payload.image)
    height, width = image.shape[:2]

    try:
        with _model_lock:
            result = get_model().predict(
                image,
                conf=payload.confidence,
                imgsz=DEFAULT_IMAGE_SIZE,
                verbose=False,
            )[0]
    except RuntimeError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    detections: list[dict[str, Any]] = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            detections.append(
                {
                    "label": str(result.names[class_id]),
                    "confidence": round(float(box.conf[0].item()), 4),
                    "box": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                    "class_id": class_id,
                }
            )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return {
        "ok": True,
        "image_width": width,
        "image_height": height,
        "count": len(detections),
        "detections": detections,
    }
