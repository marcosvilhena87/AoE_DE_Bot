import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np

# Stub external dependencies before importing the module under test
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

# Make sure the script package is importable
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from script.resources import CFG
from script.resources.ocr.executor import execute_ocr


class TestOcrStopsWhenConfidenceStalls(TestCase):
    def test_calls_stop_on_non_increasing_metric(self):
        gray = np.zeros((10, 10), dtype=np.uint8)

        call_count = {"count": 0}

        def fake_ocr(*args, **kwargs):
            call_count["count"] += 1
            return "1", {"text": ["1"], "conf": ["5"]}, None

        with patch("script.resources.ocr.masks._ocr_digits_better", side_effect=fake_ocr):
            with patch.dict(
                CFG,
                {
                    "ocr_conf_threshold": 60,
                    "ocr_conf_decay": 0.5,
                    "ocr_conf_min": 1,
                    "ocr_conf_max_attempts": 10,
                },
                clear=False,
            ):
                execute_ocr(gray)

        self.assertEqual(call_count["count"], 2)

