from __future__ import annotations

import base64
import json
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
BASELINE_MS = 1400
FRAME_RATE = 16
WIDTH = 320
HEIGHT = 240

COLORS = [
    ("#ff1744", (255, 23, 68)),
    ("#00e676", (0, 230, 118)),
    ("#2979ff", (41, 121, 255)),
    ("#ffea00", (255, 234, 0)),
    ("#d500f9", (213, 0, 249)),
    ("#ffffff", (255, 255, 255)),
]

app = FastAPI(title="Face Flashing Standalone MVP")
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@dataclass
class Frame:
    index: int
    elapsed_ms: float
    face_found: bool
    face_area_ratio: float
    mean_bgr: tuple[float, float, float] | None
    mean_hsv: tuple[float, float, float] | None
    zones: list[float]
    texture_var: float | None
    laplacian_var: float | None
    global_brightness: float


@dataclass
class Session:
    session_id: str
    baseline_ms: int
    frame_rate: int
    width: int
    height: int
    challenge: list[dict]
    frames: list[Frame] = field(default_factory=list)


sessions: dict[str, Session] = {}

# Helps debugging: which face bbox provider is currently used.
DETECTOR_KIND = "unknown"


def try_load_haar_cascade() -> cv2.CascadeClassifier | None:
    if not hasattr(cv2, "CascadeClassifier"):
        return None

    haar_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not haar_path.is_file():
        return None

    cascade = cv2.CascadeClassifier(str(haar_path))
    if cascade.empty():
        return None

    return cascade


CASCADE: cv2.CascadeClassifier | None = None


MP_DETECTOR = None
FACE_MODEL_PATH = ROOT / "models" / "blaze_face_short_range.tflite"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


def ensure_face_model() -> Path:
    if FACE_MODEL_PATH.is_file():
        return FACE_MODEL_PATH

    FACE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
    return FACE_MODEL_PATH


def get_mediapipe_detector():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = ensure_face_model()
    options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path.resolve())),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )
    return vision.FaceDetector.create_from_options(options)


def detect_face_bbox(frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Returns (x, y, w, h) for the largest detected face.
    """
    global CASCADE, MP_DETECTOR, DETECTOR_KIND

    h_img, w_img = frame_bgr.shape[:2]

    if CASCADE is None:
        CASCADE = try_load_haar_cascade()

    if CASCADE is not None:
        DETECTOR_KIND = "haar"
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.12,
            minNeighbors=4,
            minSize=(45, 45),
        )

        if len(faces) == 0:
            return None

        x, y, w, h = max(
            (tuple(face) for face in faces),
            key=lambda face: face[2] * face[3],
        )
        return int(x), int(y), int(w), int(h)

    # Fallback: mediapipe face detector
    if MP_DETECTOR is None:
        try:
            MP_DETECTOR = get_mediapipe_detector()
        except Exception:
            # If mediapipe can't be loaded, fall back to a center crop.
            DETECTOR_KIND = "center_crop"
            return (
                int(w_img * 0.25),
                int(h_img * 0.2),
                int(w_img * 0.5),
                int(h_img * 0.6),
            )

    DETECTOR_KIND = "mediapipe"
    import mediapipe as mp

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = MP_DETECTOR.detect(mp_image)

    detections = detection_result.detections or []
    if not detections:
        return None

    best = None
    best_area = 0

    for det in detections:
        bbox = det.bounding_box
        x_min = int(max(0, bbox.origin_x))
        y_min = int(max(0, bbox.origin_y))
        bw = int(max(0, bbox.width))
        bh = int(max(0, bbox.height))
        area = bw * bh

        if area > best_area:
            best_area = area
            best = (x_min, y_min, bw, bh)

    return best


def init_face_detector() -> str:
    """Warm up face detector at startup and return active detector kind."""
    global CASCADE, MP_DETECTOR, DETECTOR_KIND

    CASCADE = try_load_haar_cascade()
    if CASCADE is not None:
        DETECTOR_KIND = "haar"
        return DETECTOR_KIND

    try:
        MP_DETECTOR = get_mediapipe_detector()
        DETECTOR_KIND = "mediapipe"
    except Exception as exc:
        DETECTOR_KIND = "center_crop"
        print(f"WARNING: MediaPipe face detector unavailable ({exc}); using center_crop fallback")

    return DETECTOR_KIND


@app.on_event("startup")
def on_startup() -> None:
    kind = init_face_detector()
    print(f"Face detector: {kind}")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.post("/api/session")
def create_session() -> dict:
    rng = random.SystemRandom()
    offset = BASELINE_MS
    challenge = []

    for idx, (color, rgb) in enumerate(rng.sample(COLORS, k=5)):
        offset += rng.randint(240, 460)
        duration = rng.randint(400, 650)
        challenge.append(
            {
                "id": f"flash_{idx + 1}",
                "color": color,
                "rgb": rgb,
                "offset_ms": offset,
                "duration_ms": duration,
            }
        )
        offset += duration

    session = Session(
        session_id=str(uuid.uuid4()),
        baseline_ms=BASELINE_MS,
        frame_rate=FRAME_RATE,
        width=WIDTH,
        height=HEIGHT,
        challenge=challenge,
    )
    sessions[session.session_id] = session

    return {
        "session_id": session.session_id,
        "baseline_ms": session.baseline_ms,
        "frame_rate": session.frame_rate,
        "width": session.width,
        "height": session.height,
        "challenge": session.challenge,
    }


def decode_image(data_url: str) -> np.ndarray:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image")
    return frame


def crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = frame.shape[:2]
    return frame[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]


def zones(face: np.ndarray) -> list[np.ndarray]:
    h, w = face.shape[:2]
    parts = [
        crop(face, int(w * 0.2), int(h * 0.1), int(w * 0.8), int(h * 0.32)),
        crop(face, int(w * 0.08), int(h * 0.35), int(w * 0.42), int(h * 0.68)),
        crop(face, int(w * 0.58), int(h * 0.35), int(w * 0.92), int(h * 0.68)),
        crop(face, int(w * 0.36), int(h * 0.3), int(w * 0.64), int(h * 0.72)),
        crop(face, int(w * 0.25), int(h * 0.7), int(w * 0.75), int(h * 0.95)),
    ]
    return [part for part in parts if part.size > 0]


def extract(frame_index: int, elapsed_ms: float, image: str) -> Frame:
    frame = decode_image(image)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    global_brightness = float(gray.mean())

    bbox = detect_face_bbox(frame)
    if bbox is None:
        return Frame(
            index=frame_index,
            elapsed_ms=elapsed_ms,
            face_found=False,
            face_area_ratio=0.0,
            mean_bgr=None,
            mean_hsv=None,
            zones=[],
            texture_var=None,
            laplacian_var=None,
            global_brightness=global_brightness,
        )

    x, y, w, h = bbox
    roi = frame[y : y + h, x : x + w]
    roi_gray = gray[y : y + h, x : x + w]
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    zone_values = [float(cv2.cvtColor(part, cv2.COLOR_BGR2GRAY).mean()) for part in zones(roi)]

    return Frame(
        index=frame_index,
        elapsed_ms=elapsed_ms,
        face_found=True,
        face_area_ratio=float((w * h) / (frame.shape[0] * frame.shape[1])),
        mean_bgr=tuple(float(v) for v in roi.mean(axis=(0, 1))),
        mean_hsv=tuple(float(v) for v in roi_hsv.mean(axis=(0, 1))),
        zones=zone_values,
        texture_var=float(roi_gray.var()),
        laplacian_var=float(cv2.Laplacian(roi_gray, cv2.CV_64F).var()),
        global_brightness=global_brightness,
    )


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def avg(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def mean_bgr(frames: list[Frame]) -> np.ndarray | None:
    values = [frame.mean_bgr for frame in frames if frame.mean_bgr is not None]
    if not values:
        return None
    return np.mean(np.asarray(values, dtype=np.float64), axis=0)


def unit_vector(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 1e-9 else np.zeros_like(values)


def analyze_flash_step(face_frames: list[Frame], step: dict) -> dict:
    """
    Match a face color change to one exact challenge flash.

    A local pre-flash baseline avoids treating slow auto-exposure drift as a
    response. Only frames captured while the flash is visible are eligible.
    """
    offset = float(step["offset_ms"])
    duration = float(step["duration_ms"])
    pre_frames = [
        frame
        for frame in face_frames
        if offset - 230.0 <= frame.elapsed_ms <= offset - 30.0
    ]
    flash_frames = [
        frame
        for frame in face_frames
        if offset + 35.0 <= frame.elapsed_ms <= offset + duration
    ]
    baseline_bgr = mean_bgr(pre_frames)

    if baseline_bgr is None or len(pre_frames) < 2 or len(flash_frames) < 2:
        return {
            "id": step["id"],
            "passed": False,
            "reason": "insufficient_frames",
            "strength": 0.0,
            "color_match": 0.0,
            "latency_ms": None,
        }

    # Frames and OpenCV are BGR; browser challenge colors are RGB.
    flash_rgb = np.asarray(step["rgb"], dtype=np.float64)
    expected_bgr = flash_rgb[::-1]
    expected_chroma = expected_bgr / max(float(expected_bgr.sum()), 1.0)
    baseline_chroma = baseline_bgr / max(float(baseline_bgr.sum()), 1.0)
    expected_axis = expected_chroma - baseline_chroma
    is_white = float(np.ptp(expected_bgr)) < 25.0

    candidates = []
    for frame in flash_frames:
        if frame.mean_bgr is None:
            continue

        current_bgr = np.asarray(frame.mean_bgr, dtype=np.float64)
        current_chroma = current_bgr / max(float(current_bgr.sum()), 1.0)
        observed_axis = current_chroma - baseline_chroma
        channel_delta = current_bgr - baseline_bgr
        brightness_rise = float(current_bgr.mean() - baseline_bgr.mean())

        if is_white:
            color_match = 1.0 if brightness_rise > 0.0 else 0.0
            raw_strength = max(0.0, brightness_rise)
            strength = clamp(raw_strength / 12.0)
        else:
            color_match = clamp(
                (float(np.dot(unit_vector(observed_axis), unit_vector(expected_axis))) + 1.0)
                / 2.0
            )
            # Remove common-mode brightness: a true colored flash must alter
            # the channel balance, not merely trigger camera auto-exposure.
            chromatic_delta = channel_delta - channel_delta.mean()
            raw_strength = float(np.linalg.norm(chromatic_delta))
            strength = clamp(raw_strength / 14.0)

        evidence = strength * color_match
        candidates.append(
            {
                "frame": frame,
                "strength": strength,
                "raw_strength": raw_strength,
                "color_match": color_match,
                "evidence": evidence,
            }
        )

    if not candidates:
        return {
            "id": step["id"],
            "passed": False,
            "reason": "insufficient_frames",
            "strength": 0.0,
            "color_match": 0.0,
            "latency_ms": None,
        }

    peak = max(candidates, key=lambda item: item["evidence"])
    peak_evidence = float(peak["evidence"])
    onset = next(
        (
            item
            for item in candidates
            if item["evidence"] >= max(0.12, peak_evidence * 0.55)
        ),
        peak,
    )
    latency_ms = max(0.0, float(onset["frame"].elapsed_ms) - offset)
    passed = (
        peak["strength"] >= (0.18 if is_white else 0.16)
        and peak["color_match"] >= (0.8 if is_white else 0.72)
        and latency_ms <= min(duration, 520.0)
    )

    return {
        "id": step["id"],
        "color": step["color"],
        "passed": passed,
        "reason": "matched" if passed else "weak_or_wrong_color",
        "strength": round(float(peak["strength"]), 4),
        "raw_strength": round(float(peak["raw_strength"]), 4),
        "color_match": round(float(peak["color_match"]), 4),
        "latency_ms": round(latency_ms, 1),
        "is_white": is_white,
    }


def score(session: Session) -> dict:
    frames = sorted(session.frames, key=lambda item: item.elapsed_ms)
    total = len(frames)
    face_frames = [frame for frame in frames if frame.face_found]

    if total < 20:
        return result(session, "uncertain", 0.2, ["too_few_frames"], {"total_frames": total})

    face_ratio = len(face_frames) / total
    area_std = pstdev([frame.face_area_ratio for frame in face_frames]) if len(face_frames) > 1 else 0.0
    face_stability = clamp(1.0 - area_std / 0.08)
    texture_score = clamp(
        avg([frame.texture_var or 0.0 for frame in face_frames]) / 900.0 * 0.55
        + avg([frame.laplacian_var or 0.0 for frame in face_frames]) / 140.0 * 0.45
    )

    flash_results = [analyze_flash_step(face_frames, step) for step in session.challenge]
    measurable = [item for item in flash_results if item["reason"] != "insufficient_frames"]
    passed = [item for item in measurable if item["passed"]]
    valid_flash_ratio = len(passed) / len(session.challenge)
    response_score = avg([item["strength"] for item in measurable])
    color_match_score = avg([item["color_match"] for item in measurable])
    challenge_score = clamp(
        0.55 * valid_flash_ratio
        + 0.25 * response_score
        + 0.20 * color_match_score
    )
    valid_latencies = [item["latency_ms"] for item in passed if item["latency_ms"] is not None]
    mean_latency = avg(valid_latencies) if valid_latencies else None

    # Face/texture are supporting quality signals only. The randomized
    # color challenge is mandatory and dominates the probability.
    quality_score = clamp(
        0.55 * face_ratio
        + 0.25 * face_stability
        + 0.20 * texture_score
    )
    live = clamp(0.08 + 0.78 * challenge_score + 0.14 * quality_score)
    reasons = []

    if face_ratio >= 0.8:
        reasons.append("stable_face")
    else:
        reasons.append("face_not_stable")

    if len(measurable) < 4:
        live = min(live, 0.49)
        reasons.append("insufficient_flash_samples")
    elif len(passed) >= 3 and valid_flash_ratio >= 0.6:
        reasons.append("color_challenge_passed")
    else:
        live = min(live, 0.39)
        reasons.append("color_challenge_failed")

    if DETECTOR_KIND == "center_crop":
        reasons.append("fallback_detector_center_crop")
        live = min(live, 0.49)

    if texture_score >= 0.18:
        reasons.append("skin_texture_present")
    else:
        reasons.append("low_texture_detail")

    live = clamp(live)
    challenge_passed = (
        len(measurable) >= 4
        and len(passed) >= 3
        and valid_flash_ratio >= 0.6
        and color_match_score >= 0.70
    )
    verdict = (
        "live"
        if challenge_passed and face_ratio >= 0.8 and live >= 0.68
        else "spoof"
        if live <= 0.40
        else "uncertain"
    )

    return result(
        session,
        verdict,
        live,
        reasons,
        {
            "total_frames": total,
            "face_found_ratio": face_ratio,
            "face_stability": face_stability,
            "mean_latency_ms": mean_latency,
            "response_score": response_score,
            "color_match_score": color_match_score,
            "valid_flash_ratio": valid_flash_ratio,
            "passed_flashes": len(passed),
            "measurable_flashes": len(measurable),
            "challenge_score": challenge_score,
            "challenge_passed": challenge_passed,
            "texture_score": texture_score,
            "flash_results": flash_results,
        },
    )


def result(session: Session, verdict: str, live: float, reasons: list[str], metrics: dict) -> dict:
    # Include face detector kind for quick debugging.
    metrics = dict(metrics)
    metrics["detector_kind"] = DETECTOR_KIND

    return {
        "session_id": session.session_id,
        "verdict": verdict,
        "live_probability": live,
        "spoof_probability": 1.0 - live,
        "reasons": reasons,
        "metrics": metrics,
    }


@app.websocket("/ws/preview")
async def preview_endpoint(websocket: WebSocket) -> None:
    """Lightweight face bbox stream for camera preview overlay."""
    await websocket.accept()
    try:
        while True:
            message = json.loads(await websocket.receive_text())
            if message.get("type") != "preview_frame":
                continue

            frame = decode_image(message["image"])
            frame_h, frame_w = frame.shape[:2]
            bbox = detect_face_bbox(frame)

            if bbox is None:
                await websocket.send_json(
                    {
                        "type": "face_bbox",
                        "found": False,
                        "frame_w": frame_w,
                        "frame_h": frame_h,
                        "detector_kind": DETECTOR_KIND,
                    }
                )
                continue

            x, y, w, h = bbox
            await websocket.send_json(
                {
                    "type": "face_bbox",
                    "found": True,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "frame_w": frame_w,
                    "frame_h": frame_h,
                    "detector_kind": DETECTOR_KIND,
                }
            )
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()


@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = sessions.get(session_id)

    if session is None:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return

    try:
        while True:
            message = json.loads(await websocket.receive_text())
            if message.get("type") == "frame":
                frame_index = int(message["frame_index"])

                features = extract(
                    frame_index=frame_index,
                    elapsed_ms=float(message["elapsed_ms"]),
                    image=message["image"],
                )
                session.frames.append(features)
            elif message.get("type") == "complete":
                await websocket.send_json({"type": "result", "payload": score(session)})
                await websocket.close()
                return
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()


if __name__ == "__main__":
    print("Open http://localhost:8095")
    uvicorn.run(app, host="0.0.0.0", port=8095)
