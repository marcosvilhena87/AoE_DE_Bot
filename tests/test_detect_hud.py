import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np

# Stub modules requiring GUI
dummy_pg = types.SimpleNamespace(
    PAUSE=0,
    FAILSAFE=False,
    size=lambda: (200, 200),
    click=lambda *a, **k: None,
    moveTo=lambda *a, **k: None,
    press=lambda *a, **k: None,
)


class DummyMSS:
    monitors = [{}, {"left": 0, "top": 0, "width": 200, "height": 200}]

    def grab(self, region):
        h, w = region["height"], region["width"]
        return np.zeros((h, w, 4), dtype=np.uint8)


sys.modules.setdefault("pyautogui", dummy_pg)
sys.modules.setdefault("mss", types.SimpleNamespace(mss=lambda: DummyMSS()))
pytesseract_stub = types.SimpleNamespace(
    pytesseract=types.SimpleNamespace(tesseract_cmd=""),
    image_to_data=lambda *a, **k: {"text": [], "conf": []},
)
sys.modules.setdefault("pytesseract", pytesseract_stub)
os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")
cv2_stub = types.SimpleNamespace(
    cvtColor=lambda src, code: src,
    resize=lambda img, *a, **k: img,
    matchTemplate=lambda *a, **k: np.zeros((1, 1), dtype=np.float32),
    minMaxLoc=lambda *a, **k: (0, 0, (0, 0), (0, 0)),
    imread=lambda *a, **k: np.zeros((1, 1), dtype=np.uint8),
    imwrite=lambda *a, **k: True,
    medianBlur=lambda src, k: src,
    bitwise_not=lambda src: src,
    rectangle=lambda img, pt1, pt2, color, thickness: img,
    threshold=lambda src, *a, **k: (None, src),
    bilateralFilter=lambda src, d, sigmaColor, sigmaSpace: src,
    adaptiveThreshold=lambda src, maxValue, adaptiveMethod, thresholdType, blockSize, C: src,
    dilate=lambda src, kernel, iterations=1: src,
    equalizeHist=lambda src: src,
    countNonZero=lambda src: int(np.count_nonzero(src)),
    normalize=lambda src, *a, **k: src,
    NORM_MINMAX=0,
    IMREAD_GRAYSCALE=0,
    COLOR_BGR2GRAY=0,
    TM_CCOEFF_NORMED=0,
)
sys.modules.setdefault("cv2", cv2_stub)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.resources as resources


class TestDetectHud(TestCase):
    def test_returns_box_and_score(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_template = np.zeros((10, 10, 3), dtype=np.uint8)
        with patch.object(resources.screen_utils, "HUD_TEMPLATE", dummy_template), \
             patch("script.resources.find_template", return_value=((1, 2, 3, 4), 0.9, None)):
            box, score = resources.detect_hud(frame)
        self.assertEqual(box, (1, 2, 3, 4))
        self.assertEqual(score, 0.9)

