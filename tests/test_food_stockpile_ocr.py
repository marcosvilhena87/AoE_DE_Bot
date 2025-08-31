import os
import sys
import types
from unittest import TestCase

import cv2
import numpy as np

# Stub modules requiring a display

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

_OLD_TESS = os.environ.get("TESSERACT_CMD")
os.environ["TESSERACT_CMD"] = "/usr/bin/tesseract"

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from script.resources import CFG
from script.resources.ocr.preprocess import preprocess_roi
from script.resources.ocr.executor import execute_ocr
from script.resources.ocr.confidence import parse_confidences


class TestFoodStockpileOCR(TestCase):
    def setUp(self):
        self._old_cmd = _OLD_TESS

    def tearDown(self):
        if self._old_cmd is None:
            os.environ.pop("TESSERACT_CMD", None)
        else:
            os.environ["TESSERACT_CMD"] = self._old_cmd

    def test_food_stockpile_detects_140_high_confidence(self):
        roi = np.full((60, 150, 3), (50, 50, 50), dtype=np.uint8)
        # Anti-aliased rendering introduces gray edges around the white digits
        cv2.putText(
            roi,
            "140",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        gray = preprocess_roi(roi)
        digits, data, _mask, low_conf = execute_ocr(
            gray, color=roi, resource="food_stockpile"
        )
        confs = parse_confidences(data)
        threshold = CFG.get(
            "food_stockpile_ocr_conf_threshold", CFG.get("ocr_conf_threshold", 60)
        )
        self.assertEqual(digits, "140")
        self.assertGreaterEqual(max(confs), threshold)
        self.assertFalse(low_conf)
