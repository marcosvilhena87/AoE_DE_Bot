import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np

# Stub modules that require a GUI/display before importing the bot modules

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

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.common as common
import script.resources as resources


class TestDetectResourceRegions(TestCase):
    def test_missing_icons_raises_error(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        required = ["wood_stockpile", "food_stockpile"]
        with patch("script.resources.locate_resource_panel", return_value={"wood_stockpile": (0, 0, 10, 10)}), \
             patch.object(common, "HUD_ANCHOR", None), \
             patch("script.resources._auto_calibrate_from_icons", return_value=None), \
             patch.dict(resources.CFG, {"food_stockpile_roi": None}, clear=False):
            with self.assertRaises(common.ResourceReadError):
                resources.detect_resource_regions(frame, required)


class TestPreprocessRoi(TestCase):
    def test_preprocess_returns_grayscale(self):
        roi = np.zeros((5, 5, 3), dtype=np.uint8)
        gray = resources.preprocess_roi(roi)
        self.assertEqual(gray.shape, (5, 5))
        self.assertEqual(gray.dtype, np.uint8)


class TestExecuteOcr(TestCase):
    def test_execute_ocr_fallback(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        with patch("script.resources._ocr_digits_better", return_value=("", {}, None)), \
             patch("script.resources.pytesseract.image_to_string", return_value="456"):
            digits, data, mask = resources.execute_ocr(gray)
        self.assertEqual(digits, "456")
        self.assertEqual(data["text"], ["456"])
        np.testing.assert_array_equal(mask, gray)

    def test_execute_ocr_rejects_low_confidence(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        data = {"text": ["123"], "conf": ["10", "20", "30"]}
        with patch("script.resources._ocr_digits_better", return_value=("123", data, None)), \
             patch("script.resources.pytesseract.image_to_string", return_value="") as img2str_mock:
            digits, _, _ = resources.execute_ocr(gray, conf_threshold=60)
        self.assertEqual(digits, "")
        img2str_mock.assert_called_once()

    def test_execute_ocr_rejects_low_mean_confidence(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        data = {"text": ["12"], "conf": ["80", "20"]}
        with patch("script.resources._ocr_digits_better", return_value=("12", data, None)), \
             patch("script.resources.pytesseract.image_to_string", return_value="") as img2str_mock:
            digits, _, _ = resources.execute_ocr(gray, conf_threshold=60)
        self.assertEqual(digits, "")
        img2str_mock.assert_called_once()

    def test_execute_ocr_accepts_low_conf_single_digit(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        data = {"text": ["0"], "conf": ["10"]}
        with patch("script.resources._ocr_digits_better", return_value=("0", data, None)), \
             patch("script.resources.pytesseract.image_to_string", return_value="") as img2str_mock, \
             patch("script.resources.logger.warning") as warn_mock:
            digits, _, _ = resources.execute_ocr(gray, conf_threshold=60)
        self.assertEqual(digits, "0")
        img2str_mock.assert_not_called()
        warn_mock.assert_called_once()

    def test_execute_ocr_second_attempt_success(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        data1 = {"text": ["123"], "conf": ["10", "20", "30"]}
        data2 = {"text": ["789"], "conf": ["80", "90", "100"]}
        with patch(
            "script.resources._ocr_digits_better",
            side_effect=[("123", data1, None), ("789", data2, None)],
        ), patch("script.resources.pytesseract.image_to_string") as img2str_mock:
            digits, _, _ = resources.execute_ocr(gray, conf_threshold=60)
        self.assertEqual(digits, "789")
        img2str_mock.assert_not_called()


class TestHandleOcrFailure(TestCase):
    def test_handle_ocr_failure_raises(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        regions = {"wood_stockpile": (0, 0, 10, 10)}
        results = {"wood_stockpile": None}
        with patch("script.resources.cv2.imwrite"), \
             patch("script.resources.logger.error"), \
             patch("script.resources.pytesseract.pytesseract.tesseract_cmd", "/usr/bin/true"):
            with self.assertRaises(common.ResourceReadError):
                resources.handle_ocr_failure(frame, regions, results, ["wood_stockpile"])

    def test_handle_ocr_failure_noop_when_success(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        regions = {"wood_stockpile": (0, 0, 10, 10)}
        results = {"wood_stockpile": 1}
        with patch("script.resources.cv2.imwrite") as imwrite_mock:
            resources.handle_ocr_failure(frame, regions, results, ["wood_stockpile"])
        imwrite_mock.assert_not_called()
