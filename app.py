from __future__ import annotations

import base64
import io
import json
import os
import re
import socket
import threading
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from ultralytics import YOLO
import websocket


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
MODEL_PATH = Path(os.getenv("FASHION_MODEL_PATH", ROOT / "models" / "yolov8n-clothing-detection.pt"))
DEFAULT_CONFIDENCE = float(os.getenv("FASHION_CONFIDENCE", "0.35"))
DEFAULT_IMAGE_SIZE = int(os.getenv("FASHION_IMAGE_SIZE", "640"))
DEFAULT_RECAMERA_HOST = os.getenv("RECAMERA_HOST", "192.168.42.1")
DEFAULT_RECAMERA_PORT = int(os.getenv("RECAMERA_PORT", "8090"))
DEFAULT_RECAMERA_TIMEOUT = float(os.getenv("RECAMERA_TIMEOUT", "5"))

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


class ReCameraDetectionRequest(BaseModel):
    host: str = Field(default=DEFAULT_RECAMERA_HOST, min_length=1, max_length=253)
    port: int = Field(default=DEFAULT_RECAMERA_PORT, ge=1, le=65535)
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


def validate_recamera_host(host: str) -> str:
    value = host.strip()
    if not re.fullmatch(r"[A-Za-z0-9.-]+", value) or value.startswith(".") or value.endswith("."):
        raise HTTPException(status_code=400, detail="Invalid reCamera host")
    return value


def parse_recamera_payload(raw: str | bytes, source: str) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="strict")
    try:
        payload = json.loads(raw)
        data = payload.get("data", payload)
        image_b64 = data.get("image")
    except (json.JSONDecodeError, AttributeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Invalid reCamera payload from {source}") from exc
    if not image_b64:
        raise RuntimeError(f"reCamera payload from {source} did not contain data.image")
    return {
        "image": image_b64,
        "resolution": data.get("resolution"),
        "onboard_labels": data.get("labels") or [],
        "onboard_boxes": data.get("boxes") or [],
        "source": source,
    }


def capture_recamera_frame(host: str, port: int, timeout: float = DEFAULT_RECAMERA_TIMEOUT) -> dict[str, Any]:
    host = validate_recamera_host(host)
    source = f"ws://{host}:{port}"
    connection = None
    try:
        connection = websocket.create_connection(source, timeout=timeout)
        raw = connection.recv()
        return parse_recamera_payload(raw, source)
    except (OSError, websocket.WebSocketException, RuntimeError) as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Could not read the reCamera stream at {source}: {exc}. "
                "Make sure the reCamera preview/Node-RED flow is deployed and port 8090 is open."
            ),
        ) from exc
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def run_detection(image: np.ndarray, confidence: float) -> dict[str, Any]:
    height, width = image.shape[:2]
    try:
        with _model_lock:
            result = get_model().predict(
                image,
                conf=confidence,
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


def port_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


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
        "recamera_default": f"ws://{DEFAULT_RECAMERA_HOST}:{DEFAULT_RECAMERA_PORT}",
    }


@app.post("/api/detect")
def detect(payload: DetectionRequest) -> dict[str, Any]:
    image = decode_image(payload.image)
    return run_detection(image, payload.confidence)


@app.get("/api/recamera/status")
def recamera_status(host: str = DEFAULT_RECAMERA_HOST, port: int = DEFAULT_RECAMERA_PORT) -> dict[str, Any]:
    host = validate_recamera_host(host)
    if not 1 <= port <= 65535:
        raise HTTPException(status_code=400, detail="Invalid reCamera port")
    dashboard_reachable = port_reachable(host, 80)
    stream_reachable = port_reachable(host, port)
    return {
        "ok": dashboard_reachable and stream_reachable,
        "host": host,
        "port": port,
        "dashboard_reachable": dashboard_reachable,
        "stream_reachable": stream_reachable,
        "stream_url": f"ws://{host}:{port}",
        "hint": None
        if stream_reachable
        else "Open the reCamera dashboard and deploy/start the preview Node-RED flow that publishes port 8090.",
    }


@app.post("/api/recamera/detect")
def detect_recamera(payload: ReCameraDetectionRequest) -> dict[str, Any]:
    frame = capture_recamera_frame(payload.host, payload.port)
    result = run_detection(decode_image(frame["image"]), payload.confidence)
    image_b64 = frame["image"].split(",", 1)[-1].strip()
    return {
        **result,
        "image": image_b64,
        "recamera": {
            "source": frame["source"],
            "resolution": frame["resolution"],
            "onboard_labels": frame["onboard_labels"],
            "onboard_boxes": frame["onboard_boxes"],
        },
    }
