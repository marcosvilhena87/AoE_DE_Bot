import os
import sys
import types
from unittest import TestCase

import numpy as np


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

dummy_cv2 = types.SimpleNamespace(
    cvtColor=lambda src, code: src,
    resize=lambda img, *a, **k: img,
    matchTemplate=lambda *a, **k: np.zeros((1, 1), dtype=np.float32),
    minMaxLoc=lambda *a, **k: (0, 0, (0, 0), (0, 0)),
    imread=lambda *a, **k: np.zeros((1, 1), dtype=np.uint8),
    imwrite=lambda *a, **k: True,
    medianBlur=lambda src, k: src,
    bitwise_not=lambda src: src,
    threshold=lambda src, *a, **k: (None, src),
    rectangle=lambda img, pt1, pt2, color, thickness: img,
    IMREAD_GRAYSCALE=0,
    COLOR_BGR2GRAY=0,
    INTER_LINEAR=0,
    THRESH_BINARY=0,
    THRESH_OTSU=0,
    TM_CCOEFF_NORMED=0,
)
sys.modules.setdefault("cv2", dummy_cv2)
sys.modules.setdefault(
    "pytesseract",
    types.SimpleNamespace(
        image_to_data=lambda *a, **k: {"text": [""], "conf": ["0"]},
        image_to_string=lambda *a, **k: "",
        Output=types.SimpleNamespace(DICT="dict"),
        pytesseract=types.SimpleNamespace(tesseract_cmd=""),
    ),
)

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from script.resources.ocr import _sanitize_digits


class TestSanitizeDigits(TestCase):
    def test_1400(self):
        self.assertEqual(_sanitize_digits("1400"), "140")

    def test_800(self):
        self.assertEqual(_sanitize_digits("800"), "800")

    def test_1000(self):
        self.assertEqual(_sanitize_digits("1000"), "100")
