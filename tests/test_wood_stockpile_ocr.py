import os
import sys
import types
import shutil
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

_TESS_PATH = shutil.which("tesseract") or "/usr/bin/tesseract"

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.resources.ocr as ocr


class TestWoodStockpileOCR(TestCase):
    def setUp(self):
        self._old_cmd = os.environ.get("TESSERACT_CMD")
        os.environ["TESSERACT_CMD"] = _TESS_PATH

    def tearDown(self):
        if self._old_cmd is None:
            os.environ.pop("TESSERACT_CMD", None)
        else:
            os.environ["TESSERACT_CMD"] = self._old_cmd
    def test_wood_stockpile_high_confidence(self):
        roi = np.full((60, 120, 3), (19, 69, 139), dtype=np.uint8)
        cv2.putText(roi, "123", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        gray = ocr.preprocess_roi(roi)
        digits, data, _mask, low_conf = ocr.execute_ocr(gray, resource="wood_stockpile")
        confs = ocr.parse_confidences(data)
        threshold = ocr.CFG.get("ocr_conf_threshold", 60)
        self.assertTrue(digits.isdigit())
        self.assertGreaterEqual(np.median(confs), threshold)
        self.assertFalse(low_conf)
