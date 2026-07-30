# Fashion Detection Demo

A browser-based clothing detector powered by YOLOv8. Use a webcam or upload a
photo to find four fashion categories:

- clothing
- shoes
- bags
- accessories

The repository includes the trained model weights, a FastAPI inference API, and
a responsive web interface that draws detection boxes in the browser.

## Run locally

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. Webcam access works on `localhost` or behind
HTTPS. The model runs on CPU by default and automatically uses CUDA when
available.

See [INSTALL.md](INSTALL.md) for complete Linux, macOS, Windows, GPU, and
environment configuration instructions.

## API

`POST /api/detect` accepts a base64-encoded image or data URL:

```json
{
  "image": "data:image/jpeg;base64,...",
  "confidence": 0.35
}
```

The response includes the original image dimensions and a list of detections:

```json
{
  "ok": true,
  "image_width": 1280,
  "image_height": 720,
  "count": 1,
  "detections": [
    {
      "label": "clothing",
      "confidence": 0.91,
      "box": [210.4, 86.2, 792.8, 684.5],
      "class_id": 2
    }
  ]
}
```

Interactive API documentation is available at `/docs`, and `GET /health`
reports model availability.

## Configuration

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `FASHION_MODEL_PATH` | `models/yolov8n-clothing-detection.pt` | Path to the YOLO weights |
| `FASHION_CONFIDENCE` | `0.35` | Default confidence threshold |
| `FASHION_IMAGE_SIZE` | `640` | Inference image size |

## Project layout

```text
.
├── app.py
├── .env.example
├── INSTALL.md
├── models/
│   └── yolov8n-clothing-detection.pt
├── static/
│   └── index.html
└── requirements.txt
```

Do not commit API keys or tokens. Local `.env` files are ignored by Git.

