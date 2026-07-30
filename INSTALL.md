# Installation and environment setup

This guide sets up the Fashion Detection Demo from a fresh clone. Python
3.10–3.12 is recommended.

## 1. Clone the repository

```bash
git clone https://github.com/manavgoel472003/fashion-detection.git
cd fashion-detection
```

## 2. Create a virtual environment

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Ubuntu or Debian, install the virtual-environment package first if needed:

```bash
sudo apt update
sudo apt install python3-venv
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

The default installation supports CPU inference. Ultralytics and PyTorch will
automatically use a compatible CUDA GPU when a CUDA-enabled PyTorch build and
NVIDIA driver are available.

To use a specific CUDA build, install the appropriate PyTorch package for your
machine before running `pip install -r requirements.txt`.

## 4. Configure the environment

Create a local `.env` file from the tracked template.

### Linux or macOS

```bash
cp .env.example .env
```

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

Available settings:

| Variable | Default | Description |
| --- | --- | --- |
| `FASHION_MODEL_PATH` | `models/yolov8n-clothing-detection.pt` | Model weights path |
| `FASHION_CONFIDENCE` | `0.35` | Default detection confidence, from `0.05` to `0.95` |
| `FASHION_IMAGE_SIZE` | `640` | YOLO inference resolution |
| `RECAMERA_HOST` | `192.168.42.1` | Default Seeed reCamera IP or hostname |
| `RECAMERA_PORT` | `8090` | JSON preview WebSocket port |
| `RECAMERA_TIMEOUT` | `5` | Stream connection timeout in seconds |

The application automatically loads `.env` at startup. Do not commit `.env`;
it is already excluded by `.gitignore`.

## 5. Start the demo

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open:

- Demo: <http://localhost:8000>
- Interactive API docs: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>

Webcam access is allowed on `localhost`. A remote deployment should use HTTPS
for browser camera permission.

## Verify the API

With the server running:

```bash
curl http://localhost:8000/health
```

The response should report `"ok": true` and `"model_available": true`.

## Troubleshooting

### Model not found

Confirm that the bundled model exists:

```bash
ls -lh models/yolov8n-clothing-detection.pt
```

If the model is elsewhere, set its path in `.env`:

```dotenv
FASHION_MODEL_PATH=/absolute/path/to/model.pt
```

### Camera button does not work

- Allow camera permission in the browser.
- Use `localhost` when running locally.
- Use HTTPS when accessing the demo from another computer.
- Image upload remains available when no camera is present.

### reCamera dashboard works but the stream does not

The reCamera dashboard uses port `80`, while this demo receives JSON preview
frames from port `8090`. Check both from the host running the demo:

```bash
curl -I http://192.168.42.1
curl "http://localhost:8000/api/recamera/status?host=192.168.42.1&port=8090"
```

If `dashboard_reachable` is true but `stream_reachable` is false, open the
reCamera dashboard or Node-RED editor and deploy/start the preview flow that
publishes the WebSocket on port `8090`. If the camera is connected over Wi-Fi
or Ethernet, replace `192.168.42.1` with its assigned address.

### Slow first detection

The first request loads the model into memory and can take longer than later
requests. CPU inference is supported; a compatible CUDA setup improves speed.

## Deactivate the environment

```bash
deactivate
```
