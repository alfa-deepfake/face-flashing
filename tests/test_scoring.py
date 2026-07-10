import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

import app  # noqa: E402


BASE_BGR = np.asarray((90.0, 110.0, 130.0))


def make_frame(index: int, elapsed_ms: float, bgr: np.ndarray) -> app.Frame:
    return app.Frame(
        index=index,
        elapsed_ms=elapsed_ms,
        face_found=True,
        face_area_ratio=0.24,
        mean_bgr=tuple(float(value) for value in bgr),
        mean_hsv=(10.0, 70.0, float(max(bgr))),
        zones=[100.0, 103.0, 98.0, 106.0, 101.0],
        texture_var=1000.0,
        laplacian_var=180.0,
        global_brightness=float(np.mean(bgr)),
    )


def build_session(with_flash_response: bool) -> app.Session:
    colors = [
        ("#ff1744", (255, 23, 68)),
        ("#00e676", (0, 230, 118)),
        ("#2979ff", (41, 121, 255)),
        ("#ffea00", (255, 234, 0)),
        ("#d500f9", (213, 0, 249)),
    ]
    challenge = []
    frames = []
    index = 0

    for step_index, (color, rgb) in enumerate(colors):
        offset = 1000.0 + step_index * 1000.0
        challenge.append(
            {
                "id": f"flash_{step_index + 1}",
                "color": color,
                "rgb": rgb,
                "offset_ms": offset,
                "duration_ms": 500,
            }
        )
        for elapsed in (offset - 210, offset - 140, offset - 70):
            frames.append(make_frame(index, elapsed, BASE_BGR))
            index += 1
        for elapsed in (offset + 60, offset + 130, offset + 210):
            current = BASE_BGR
            if with_flash_response:
                expected_bgr = np.asarray(rgb[::-1], dtype=np.float64)
                current = BASE_BGR * 0.78 + expected_bgr * 0.22
            frames.append(make_frame(index, elapsed, current))
            index += 1

    return app.Session(
        session_id="test",
        baseline_ms=800,
        frame_rate=16,
        width=320,
        height=240,
        challenge=challenge,
        frames=frames,
    )


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        app.DETECTOR_KIND = "mediapipe"

    def test_matching_color_sequence_can_pass(self) -> None:
        scored = app.score(build_session(with_flash_response=True))
        self.assertEqual(scored["verdict"], "live")
        self.assertTrue(scored["metrics"]["challenge_passed"])
        self.assertGreaterEqual(scored["metrics"]["passed_flashes"], 3)

    def test_stable_face_without_flash_response_cannot_pass(self) -> None:
        scored = app.score(build_session(with_flash_response=False))
        self.assertEqual(scored["verdict"], "spoof")
        self.assertFalse(scored["metrics"]["challenge_passed"])
        self.assertEqual(scored["metrics"]["passed_flashes"], 0)


if __name__ == "__main__":
    unittest.main()
