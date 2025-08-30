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

import script.resources.ocr as ocr


class TestOcrConfDecayLogging(TestCase):
    def test_logs_old_and_new_threshold_after_decay(self):
        gray = np.zeros((10, 10), dtype=np.uint8)

        def fake_ocr(_gray):
            return "1", {"text": ["1"], "conf": ["50"]}, None

        with patch.object(ocr, "_ocr_digits_better", side_effect=fake_ocr):
            with patch.dict(
                ocr.CFG,
                {"ocr_conf_threshold": 60, "ocr_conf_decay": 0.5},
                clear=False,
            ):
                with self.assertLogs(ocr.logger, level="DEBUG") as cm:
                    ocr.execute_ocr(gray)

        record = next(
            r for r in cm.records if "OCR confidence threshold" in r.getMessage()
        )
        old, new = record.args
        self.assertNotEqual(old, new)
        self.assertEqual(old, 60)
        self.assertEqual(new, 30)

    def test_loop_runs_until_min_conf(self):
        gray = np.zeros((10, 10), dtype=np.uint8)

        call_count = {
            "count": 0,
        }

        def fake_ocr(_gray):
            call_count["count"] += 1
            return "1", {"text": ["1"], "conf": ["5"]}, None

        with patch.object(ocr, "_ocr_digits_better", side_effect=fake_ocr):
            with patch.dict(
                ocr.CFG,
                {
                    "ocr_conf_threshold": 60,
                    "ocr_conf_decay": 0.5,
                    "ocr_conf_min": 10,
                    "ocr_conf_max_attempts": 1,
                },
                clear=False,
            ):
                with self.assertLogs(ocr.logger, level="DEBUG") as cm:
                    digits, _data, _mask, low_conf = ocr.execute_ocr(gray)

        self.assertEqual(digits, "1")
        self.assertTrue(low_conf)
        self.assertEqual(call_count["count"], 2)
        threshold_logs = [
            r.args for r in cm.records if "OCR confidence threshold" in r.getMessage()
        ]
        self.assertEqual(threshold_logs[-1][1], 10)

    def test_stops_after_confidence_sufficient(self):
        gray = np.zeros((10, 10), dtype=np.uint8)

        call_count = {"count": 0}

        def fake_ocr(_gray):
            call_count["count"] += 1
            return "1", {"text": ["1"], "conf": ["50"]}, None

        with patch.object(ocr, "_ocr_digits_better", side_effect=fake_ocr):
            with patch.dict(
                ocr.CFG,
                {
                    "ocr_conf_threshold": 60,
                    "ocr_conf_decay": 0.5,
                    "ocr_conf_min": 10,
                    "ocr_conf_max_attempts": 5,
                },
                clear=False,
            ):
                digits, _data, _mask, low_conf = ocr.execute_ocr(gray)

        self.assertEqual(digits, "1")
        self.assertFalse(low_conf)
        self.assertEqual(call_count["count"], 2)
