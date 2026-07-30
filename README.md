# Fashion Detection Demo

A browser-based clothing detector powered by YOLOv8. Use a browser webcam, a
Seeed reCamera, or upload a photo to find four fashion categories:

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

## Use a Seeed reCamera

Connect the reCamera over USB or the same network. The default USB-network
address is `192.168.42.1`.

1. Confirm its dashboard opens at <http://192.168.42.1>.
2. In the reCamera dashboard/Node-RED interface, deploy or start the preview
   flow so its JSON WebSocket is available on port `8090`.
3. Start this application, enter the reCamera IP in the sidebar, and select
   **Start reCamera**.

The application reads frames from `ws://<host>:8090`, runs the bundled fashion
model on the host computer, and displays the annotated live result. A Reachy
Mini is optional.

Check connectivity from the API:

```bash
curl "http://localhost:8000/api/recamera/status?host=192.168.42.1&port=8090"
```

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
| `RECAMERA_HOST` | `192.168.42.1` | Default reCamera host |
| `RECAMERA_PORT` | `8090` | reCamera JSON preview WebSocket port |
| `RECAMERA_TIMEOUT` | `5` | Connection timeout in seconds |

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
