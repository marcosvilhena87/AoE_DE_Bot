import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

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


dummy_cv2 = types.SimpleNamespace(
    cvtColor=lambda src, code: src,
    resize=lambda img, *a, **k: img,
    threshold=lambda img, *a, **k: (None, img),
    imread=lambda *a, **k: np.zeros((1, 1), dtype=np.uint8),
    imwrite=lambda *a, **k: True,
    IMREAD_GRAYSCALE=0,
    COLOR_BGR2GRAY=0,
    INTER_LINEAR=0,
    THRESH_BINARY=0,
    THRESH_OTSU=0,
)

sys.modules.setdefault("pyautogui", dummy_pg)
sys.modules.setdefault("mss", types.SimpleNamespace(mss=lambda: DummyMSS()))
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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from script.resources.ocr.executor import execute_ocr


class TestZeroConfidenceHandling(TestCase):
    def test_digits_returned_when_confidences_zero(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        fake_data = {"text": ["123"], "conf": ["0", "0", "0"]}
        with patch(
            "script.resources.ocr.masks._ocr_digits_better",
            return_value=("123", fake_data, None),
        ):
            digits, data, _mask, low_conf = execute_ocr(gray)
        self.assertEqual(digits, "123")
        self.assertFalse(low_conf)
        self.assertTrue(data.get("zero_conf"))
