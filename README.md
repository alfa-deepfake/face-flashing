# Face Flashing Liveness MVP

Standalone MVP for **active face liveness** detection: the browser shows random fullscreen color flashes, streams webcam frames to a local Python server, and returns `live_probability` / `spoof_probability`.

Not a certified anti-spoofing product — a heuristic prototype for experiments.

## Structure

```text
face-flashing-mvp/
├── frontend/
│   └── index.html      # UI: camera, flashes, WebSocket client
├── server/
│   └── app.py          # FastAPI: session API, WebSocket, scoring
├── models/             # MediaPipe model (downloaded on first run)
├── requirements.txt
├── run.ps1             # Windows quick start
└── run.sh              # Linux / macOS quick start
```

## Requirements

- **Python 3.11 or 3.12** (recommended; 3.13+ may have OpenCV wheel issues)
- Webcam + modern browser (Chrome / Edge / Firefox)
- ~200 MB disk for Python deps + MediaPipe model

## Quick start

### Windows (PowerShell)

```powershell
cd face-flashing-mvp
.\run.ps1
```

Or manually:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python server\app.py
```

### Linux / macOS

```bash
cd face-flashing-mvp
chmod +x run.sh
./run.sh
```

### Conda (optional)

```powershell
conda create -y -n face-flashing python=3.12
conda activate face-flashing
pip install -r requirements.txt
python server/app.py
```

Open **http://localhost:8095**, allow camera access, click **Начать проверку**.

On first start you should see:

```text
Face detector: mediapipe
Open http://localhost:8095
```

MediaPipe face model downloads automatically into `models/` (~1 MB).

## How it works

1. Server creates a session with 5 random color flashes (400–650 ms each).
2. Browser captures ~16 fps and sends JPEG frames over WebSocket.
3. Server detects face (MediaPipe → Haar cascade fallback → center crop).
4. Scoring checks: face stability, flash color response, latency, skin texture, spatial variation.
5. Result: `live` / `spoof` / `uncertain` + debug metrics.

## API

| Endpoint | Description |
|----------|-------------|
| `GET /` | Web UI |
| `POST /api/session` | Create liveness challenge |
| `WS /ws/{session_id}` | Stream frames, receive result |

`.gitignore` excludes `.venv/`, `__pycache__/`, and downloaded `models/*.tflite`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `OpenCV face cascade not found` | Old server process — stop it (`Ctrl+C`) and restart |
| Black camera preview | Allow camera in browser; close other apps using webcam |
| False SPOOF on real face | Check `detector_kind` in debug metrics — should be `mediapipe` |
| `clearcut` / TensorFlow log spam | Harmless MediaPipe telemetry — ignore |
| Port 8095 busy | Kill old process or change port in `server/app.py` |

## License

MIT — use at your own risk for research and prototyping.
