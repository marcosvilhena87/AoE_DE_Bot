import os
import sys
import types
import shutil
from unittest import TestCase
from unittest.mock import patch

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
from script.resources.reader.core import _ocr_resource
from config import Config
from script.resources.reader.roi import prepare_roi
from script.resources.cache import ResourceCache
from script.resources.ocr.masks import _ocr_digits_better


class TestWoodStockpileOCR(TestCase):
    def setUp(self):
        self._old_cmd = _OLD_TESS

    def tearDown(self):
        if self._old_cmd is None:
            os.environ.pop("TESSERACT_CMD", None)
        else:
            os.environ["TESSERACT_CMD"] = self._old_cmd
    def test_wood_stockpile_high_confidence(self):
        roi = np.full((60, 120, 3), (19, 69, 139), dtype=np.uint8)
        cv2.putText(roi, "123", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        gray = preprocess_roi(roi)
        digits, data, _mask, low_conf = execute_ocr(
            gray, color=roi, resource="wood_stockpile"
        )
        confs = parse_confidences(data)
        threshold = CFG.get("ocr_conf_threshold", 60)
        self.assertTrue(digits.isdigit())
        self.assertGreaterEqual(max(confs), threshold)

    def test_wood_stockpile_detects_80(self):
        roi = np.full((60, 120, 3), (19, 69, 139), dtype=np.uint8)
        cv2.putText(roi, "80", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        gray = preprocess_roi(roi)
        digits, data, _mask, low_conf = execute_ocr(
            gray, color=roi, resource="wood_stockpile"
        )
        self.assertEqual(digits, "80")

    def test_wood_stockpile_no_dilate_reads_80(self):
        roi = np.full((60, 120, 3), (19, 69, 139), dtype=np.uint8)
        cv2.putText(roi, "80", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        gray = preprocess_roi(roi)
        calls = []
        orig_dilate = cv2.dilate

        def fake_dilate(src, kernel, iterations=1):
            calls.append(kernel.shape)
            return orig_dilate(src, kernel, iterations=iterations)

        with patch.dict(
            CFG, {"ocr_ws_dilate_var": 0, "wood_stockpile_dilate": False}, clear=False
        ), patch("cv2.dilate", side_effect=fake_dilate):
            digits, data, _mask, low_conf = execute_ocr(
                gray, color=roi, resource="wood_stockpile"
            )
        self.assertEqual(digits, "80")
        self.assertNotIn((3, 3), calls)

    def test_wood_stockpile_detects_300(self):
        roi = np.full((60, 150, 3), (19, 69, 139), dtype=np.uint8)
        cv2.putText(roi, "300", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        gray = preprocess_roi(roi)
        digits, data, _mask, low_conf = execute_ocr(
            gray, color=roi, resource="wood_stockpile"
        )
        self.assertEqual(digits, "300")

    def test_wood_stockpile_thin_strokes_detects_80(self):
        """Regression test ensuring thin segments survive preprocessing."""
        roi = np.full((60, 120, 3), (19, 69, 139), dtype=np.uint8)
        cv2.putText(roi, "80", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1)
        gray = preprocess_roi(roi)
        with patch.dict(CFG, {"wood_stockpile_dilate": True}, clear=False):
            digits, data, _mask, low_conf = execute_ocr(
                gray, color=roi, resource="wood_stockpile"
            )
        self.assertEqual(digits, "80")

    def test_wood_stockpile_zero_conf_rejected(self):
        roi = np.full((60, 120, 3), (19, 69, 139), dtype=np.uint8)
        cv2.putText(roi, "80", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        gray = preprocess_roi(roi)
        cache_obj = ResourceCache()

        def fake_ocr_digits_better(*args, **kwargs):
            data = {"text": ["2", "0"], "conf": ["0", "0"], "zero_conf": True}
            return "20", data, None

        with patch(
            "script.resources.ocr.executor.masks._ocr_digits_better",
            side_effect=fake_ocr_digits_better,
        ):
            cfg = Config(dict(CFG))
            cfg.update({"treat_low_conf_as_failure": True, "allow_zero_confidence_digits": False})
            digits, data, _mask, low_conf = _ocr_resource(
                "wood_stockpile",
                roi,
                gray,
                cfg.get("ocr_conf_threshold", 60),
                (0, 0, roi.shape[1], roi.shape[0]),
                cache_obj,
                cfg,
            )
        self.assertTrue(low_conf)
        self.assertEqual(parse_confidences(data), [0.0, 0.0])
        self.assertIsNone(digits)

    def test_wood_stockpile_roi_expansion_captures_80(self):
        frame = np.full((40, 80, 3), (19, 69, 139), dtype=np.uint8)
        cv2.putText(frame, "80", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        regions = {"wood_stockpile": (0, 0, 34, 40)}
        cache_obj = ResourceCache()
        roi_info = prepare_roi(frame, regions, "wood_stockpile", {"wood_stockpile"}, cache_obj)
        self.assertIsNotNone(roi_info)
        x, y, w, h, roi, gray, _top, _fail = roi_info
        self.assertGreater(w, 34)
        cfg = Config(dict(CFG))
        cfg.update({"treat_low_conf_as_failure": False})
        digits, data, _mask, low_conf = _ocr_resource(
            "wood_stockpile",
            roi,
            gray,
            cfg.get("ocr_conf_threshold", 60),
            (x, y, w, h),
            cache_obj,
            cfg,
        )
        self.assertEqual(digits, "80")

    def test_single_digit_low_confidence_triggers_color_pass(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        gray[0, 0] = 255
        color = np.zeros((10, 10, 3), dtype=np.uint8)
        call_indices = []

        def fake_run_masks(masks, psms, debug, debug_dir, ts, start_idx=0, whitelist="0123456789", resource=None):
            call_indices.append(start_idx)
            data = {"text": ["5"], "conf": ["50"]}
            return "5", data, masks[0]

        with patch("script.resources.ocr.masks._run_masks", side_effect=fake_run_masks):
            _ocr_digits_better(gray, color=color, resource="wood_stockpile")
        self.assertIn(4, call_indices)
