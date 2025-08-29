import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch
import cv2

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

    def test_blur_kernel_from_config(self):
        roi = np.zeros((40, 5, 3), dtype=np.uint8)
        with patch("script.resources.cv2.medianBlur", return_value=np.zeros((40, 5), dtype=np.uint8)) as blur_mock, \
             patch.dict(resources.CFG, {"ocr_blur_kernel": 5}, clear=False):
            resources.preprocess_roi(roi)
        args, _ = blur_mock.call_args
        self.assertEqual(args[1], 5)

    def test_blur_kernel_zero_disables(self):
        roi = np.zeros((40, 5, 3), dtype=np.uint8)
        with patch("script.resources.cv2.medianBlur") as blur_mock, \
             patch.dict(resources.CFG, {"ocr_blur_kernel": 0}, clear=False):
            gray = resources.preprocess_roi(roi)
        blur_mock.assert_not_called()
        expected = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(gray, expected)

    def test_small_roi_disables_blur(self):
        roi = np.zeros((20, 5, 3), dtype=np.uint8)
        with patch("script.resources.cv2.medianBlur") as blur_mock, \
             patch.dict(resources.CFG, {"ocr_blur_kernel": 5}, clear=False):
            gray = resources.preprocess_roi(roi)
        blur_mock.assert_not_called()
        expected = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(gray, expected)


class TestExecuteOcr(TestCase):
    def test_execute_ocr_fallback(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        with patch("script.resources._ocr_digits_better", return_value=("", {}, None)), \
             patch("script.resources.pytesseract.image_to_string", return_value="456"):
            digits, data, mask = resources.execute_ocr(gray)
        self.assertEqual(digits, "456")
        self.assertEqual(data["text"], ["456"])
        np.testing.assert_array_equal(mask, gray)

    def test_execute_ocr_warns_low_confidence(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        data = {"text": ["123"], "conf": ["10", "20", "30"]}
        with patch("script.resources._ocr_digits_better", return_value=("123", data, None)), \
             patch("script.resources.pytesseract.image_to_string", return_value="") as img2str_mock, \
             patch("script.resources.logger.warning") as warn_mock:
            digits, data_out, _ = resources.execute_ocr(gray, conf_threshold=60)
        self.assertEqual(digits, "")
        self.assertTrue(data_out.get("low_conf_multi"))
        img2str_mock.assert_called_once()
        self.assertGreaterEqual(warn_mock.call_count, 1)

    def test_execute_ocr_warns_low_mean_confidence(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        data = {"text": ["12"], "conf": ["80", "20"]}
        with patch("script.resources._ocr_digits_better", return_value=("12", data, None)), \
             patch("script.resources.pytesseract.image_to_string", return_value="") as img2str_mock, \
             patch("script.resources.logger.warning") as warn_mock, \
             patch.dict(resources.CFG, {"ocr_conf_decay": 1.0}, clear=False):
            digits, data_out, _ = resources.execute_ocr(gray, conf_threshold=60)
        self.assertEqual(digits, "")
        self.assertTrue(data_out.get("low_conf_multi"))
        img2str_mock.assert_called_once()
        self.assertGreaterEqual(warn_mock.call_count, 1)

    def test_execute_ocr_accepts_low_conf_single_digit(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        data = {"text": ["0"], "conf": ["10"]}
        with patch("script.resources._ocr_digits_better", return_value=("0", data, None)), \
             patch("script.resources.pytesseract.image_to_string", return_value="") as img2str_mock, \
             patch("script.resources.logger.warning") as warn_mock:
            digits, data_out, _ = resources.execute_ocr(gray, conf_threshold=60)
        self.assertEqual(digits, "")
        self.assertTrue(data_out.get("low_conf_single"))
        img2str_mock.assert_called_once()
        self.assertGreaterEqual(warn_mock.call_count, 1)

    def test_execute_ocr_second_attempt_success(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        data1 = {"text": ["123"], "conf": ["0", "0", "0"]}
        data2 = {"text": ["789"], "conf": ["80", "90", "100"]}
        with patch(
            "script.resources._ocr_digits_better",
            side_effect=[("123", data1, None), ("789", data2, None)],
        ), patch("script.resources.pytesseract.image_to_string") as img2str_mock:
            digits, _, _ = resources.execute_ocr(gray, conf_threshold=60)
        self.assertEqual(digits, "789")
        img2str_mock.assert_not_called()

    def test_execute_ocr_zero_variance_shortcut(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        with patch(
            "script.resources._ocr_digits_better",
            return_value=("0", {"zero_variance": True}, None),
        ), patch("script.resources.pytesseract.image_to_string") as img2str_mock, patch(
            "script.resources.logger.warning"
        ) as warn_mock:
            digits, data_out, mask = resources.execute_ocr(gray, conf_threshold=60)
        self.assertEqual(digits, "0")
        self.assertFalse(data_out.get("low_conf_single"))
        self.assertFalse(data_out.get("low_conf_multi"))
        self.assertIsNone(mask)
        img2str_mock.assert_not_called()
        warn_mock.assert_not_called()

    def test_execute_ocr_accepts_after_threshold_decay(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        side_effect = [
            ("12", {"text": ["12"], "conf": ["40", "40"]}, None)
            for _ in range(4)
        ]
        with patch("script.resources._ocr_digits_better", side_effect=side_effect) as ocr_mock, \
             patch("script.resources.pytesseract.image_to_string", return_value="") as img2str_mock, \
             patch.dict(resources.CFG, {"ocr_conf_min": 30, "ocr_conf_decay": 0.5}, clear=False):
            digits, data_out, _ = resources.execute_ocr(gray, conf_threshold=60)
        self.assertEqual(digits, "12")
        self.assertNotIn("low_conf_multi", data_out)
        img2str_mock.assert_not_called()
        self.assertGreaterEqual(ocr_mock.call_count, 3)


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

    def test_handle_ocr_failure_low_confidence_fallback(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        regions = {"wood_stockpile": (0, 0, 10, 10)}
        results = {"wood_stockpile": 50}
        cache = resources.ResourceCache()
        cache.last_resource_values["wood_stockpile"] = 99
        with patch("script.resources.cv2.imwrite"), \
             patch("script.resources.logger.warning") as warn_mock:
            resources.handle_ocr_failure(
                frame,
                regions,
                results,
                ["wood_stockpile"],
                cache=cache,
                retry_limit=2,
                low_confidence={"wood_stockpile"},
            )
            self.assertEqual(results["wood_stockpile"], 50)
            self.assertEqual(cache.resource_failure_counts["wood_stockpile"], 1)
            resources.handle_ocr_failure(
                frame,
                regions,
                results,
                ["wood_stockpile"],
                cache=cache,
                retry_limit=2,
                low_confidence={"wood_stockpile"},
            )
        self.assertEqual(results["wood_stockpile"], 99)
        self.assertEqual(warn_mock.call_count, 1)


class TestValidateStartingResources(TestCase):
    def test_deviation_raises_error(self):
        with self.assertRaises(ValueError):
            resources.validate_starting_resources(
                {"wood_stockpile": 50},
                {"wood_stockpile": 80},
                tolerance=10,
                raise_on_error=True,
            )

    def test_logs_warning_without_raise(self):
        with patch("script.resources.logger.warning") as warn_mock:
            resources.validate_starting_resources(
                {"wood_stockpile": 50},
                {"wood_stockpile": 80},
                tolerance=10,
                raise_on_error=False,
            )
            warn_mock.assert_called_once()

    def test_deviation_saves_roi_image(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        rois = {"wood_stockpile": (0, 0, 5, 5)}
        ts = 1.234
        expected_ts = int(ts * 1000)
        expected_path = resources.ROOT / "debug" / f"resource_roi_wood_stockpile_{expected_ts}.png"
        with patch("script.resources.cv2.imwrite") as imwrite_mock, \
             patch("script.resources.time.time", return_value=ts), \
             patch("script.resources.logger.error"):
            with self.assertRaises(ValueError) as ctx:
                resources.validate_starting_resources(
                    {"wood_stockpile": 50},
                    {"wood_stockpile": 80},
                    tolerance=10,
                    raise_on_error=True,
                    frame=frame,
                    rois=rois,
                )
        imwrite_mock.assert_called_once()
        self.assertIn(str(expected_path), str(ctx.exception))
