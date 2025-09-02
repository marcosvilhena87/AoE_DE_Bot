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
try:  # pragma: no cover - used for environments without OpenCV
    import cv2  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - fallback stub
    sys.modules.setdefault(
        "cv2",
        types.SimpleNamespace(
            cvtColor=lambda src, code: src,
            resize=lambda img, *a, **k: img,
            matchTemplate=lambda *a, **k: np.zeros((1, 1), dtype=np.float32),
            minMaxLoc=lambda *a, **k: (0, 0, (0, 0), (0, 0)),
            imread=lambda *a, **k: np.zeros((1, 1), dtype=np.uint8),
            imwrite=lambda *a, **k: True,
            medianBlur=lambda src, k: src,
            bitwise_not=lambda src: src,
            bitwise_or=lambda a, b: a,
            morphologyEx=lambda src, op, kernel, iterations=1: src,
            threshold=lambda src, *a, **k: (None, src),
            rectangle=lambda img, pt1, pt2, color, thickness: img,
            bilateralFilter=lambda src, d, sigmaColor, sigmaSpace: src,
            adaptiveThreshold=lambda src, maxValue, adaptiveMethod, thresholdType, blockSize, C: src,
            dilate=lambda src, kernel, iterations=1: src,
            equalizeHist=lambda src: src,
            inRange=lambda src, lower, upper: np.zeros(src.shape[:2], dtype=np.uint8),
            countNonZero=lambda src: int(np.count_nonzero(src)),
            ADAPTIVE_THRESH_GAUSSIAN_C=0,
            ADAPTIVE_THRESH_MEAN_C=0,
            IMREAD_GRAYSCALE=0,
            COLOR_BGR2GRAY=0,
            COLOR_GRAY2BGR=0,
            COLOR_BGR2HSV=0,
            INTER_LINEAR=0,
            THRESH_BINARY=0,
            THRESH_OTSU=0,
            MORPH_CLOSE=0,
            TM_CCOEFF_NORMED=0,
        ),
    )

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import script.resources.reader as resources


class TestResourceLowConfidence(TestCase):
    def setUp(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()
        resources._LAST_REGION_SPANS.clear()

    def tearDown(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()
        resources._LAST_REGION_SPANS.clear()

    def test_low_confidence_returns_none(self):
        def fake_grab_frame(bbox=None):
            if bbox:
                return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
            return np.zeros((600, 600, 3), dtype=np.uint8)

        def fake_detect(frame, required_icons, cache=None):
            return {"wood_stockpile": (0, 0, 50, 50)}

        def fake_ocr(gray):
            data = {"text": ["123"], "conf": ["10", "20", "30"]}
            return "123", data, np.zeros((1, 1), dtype=np.uint8)

        resources.RESOURCE_CACHE.last_resource_values["wood_stockpile"] = 0
        with patch.dict(resources.CFG, {"wood_stockpile_low_conf_fallback": False}, clear=False), \
            patch("script.resources.reader.detect_resource_regions", side_effect=fake_detect), \
            patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
            patch("script.resources.ocr.masks._ocr_digits_better", side_effect=fake_ocr) as ocr_mock, \
            patch("script.resources.reader.pytesseract.image_to_string", return_value="") as img2str_mock, \
            patch("script.resources.reader.cv2.imwrite"):
            result, _ = resources.read_resources_from_hud(["wood_stockpile"])

        self.assertIsNone(result["wood_stockpile"])
        self.assertGreaterEqual(ocr_mock.call_count, 1)
        img2str_mock.assert_not_called()

    def test_low_confidence_logs_warning(self):
        def fake_grab_frame(bbox=None):
            if bbox:
                return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
            return np.zeros((600, 600, 3), dtype=np.uint8)

        def fake_detect(frame, required_icons, cache=None):
            return {"wood_stockpile": (0, 0, 50, 50)}

        def fake_ocr(gray):
            data = {"text": ["123"], "conf": ["10", "20", "30"]}
            return "123", data, np.zeros((1, 1), dtype=np.uint8)

        resources.RESOURCE_CACHE.last_resource_values["wood_stockpile"] = 0
        with patch.dict(resources.CFG, {"wood_stockpile_low_conf_fallback": False}, clear=False), \
            patch("script.resources.reader.detect_resource_regions", side_effect=fake_detect), \
            patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
            patch("script.resources.ocr.masks._ocr_digits_better", side_effect=fake_ocr), \
            patch("script.resources.reader.pytesseract.image_to_string", return_value=""), \
            patch("script.resources.reader.cv2.imwrite"):
            with self.assertLogs(resources.logger, level="INFO") as cm:
                result, _ = resources.read_resources_from_hud(["wood_stockpile"])

        self.assertIsNone(result["wood_stockpile"])
        logs = "\n".join(cm.output)
        self.assertIn(
            "Discarding wood_stockpile=123 due to low-confidence OCR",
            logs,
        )
        self.assertNotIn("Detected wood_stockpile=123", logs)

    def test_low_confidence_single_digit_returns_none(self):
        def fake_grab_frame(bbox=None):
            if bbox:
                return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
            return np.zeros((600, 600, 3), dtype=np.uint8)

        def fake_detect(frame, required_icons, cache=None):
            return {"wood_stockpile": (0, 0, 50, 50)}

        ocr_seq = [
            ("7", {"text": ["7"], "conf": ["10"]}, np.zeros((1, 1), dtype=np.uint8)),
            ("", {"text": [""], "conf": [""]}, np.zeros((1, 1), dtype=np.uint8)),
            ("", {"text": [""], "conf": [""]}, np.zeros((1, 1), dtype=np.uint8)),
        ]

        def fake_ocr(gray):
            return ocr_seq.pop(0) if ocr_seq else ("", {"text": [""], "conf": [""]}, np.zeros((1, 1), dtype=np.uint8))

        resources.RESOURCE_CACHE.last_resource_values["wood_stockpile"] = 0
        with patch.dict(resources.CFG, {"wood_stockpile_low_conf_fallback": False}, clear=False), \
            patch("script.resources.reader.detect_resource_regions", side_effect=fake_detect), \
            patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
            patch("script.resources.ocr.masks._ocr_digits_better", side_effect=fake_ocr) as ocr_mock, \
            patch("script.resources.reader.pytesseract.image_to_string", return_value="") as img2str_mock, \
            patch("script.resources.reader.cv2.imwrite"):
            result, _ = resources.read_resources_from_hud(["wood_stockpile"])

        self.assertIsNone(result["wood_stockpile"])
        self.assertGreaterEqual(ocr_mock.call_count, 1)
        img2str_mock.assert_called()

    def test_zero_confidence_triggers_failure(self):
        def fake_grab_frame(bbox=None):
            if bbox:
                return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
            return np.zeros((600, 600, 3), dtype=np.uint8)

        def fake_detect(frame, required_icons, cache=None):
            return {"wood_stockpile": (0, 0, 50, 50)}

        def fake_ocr(gray):
            data = {"text": ["7"], "conf": ["0"]}
            return "7", data, np.zeros((1, 1), dtype=np.uint8)

        resources.RESOURCE_CACHE.last_resource_values["wood_stockpile"] = 0
        with patch.dict(resources.CFG, {"wood_stockpile_low_conf_fallback": False}, clear=False), \
            patch("script.resources.reader.detect_resource_regions", side_effect=fake_detect), \
            patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
            patch("script.resources.ocr.masks._ocr_digits_better", side_effect=fake_ocr), \
            patch("script.resources.reader.pytesseract.image_to_string", return_value="") as img2str_mock, \
            patch("script.resources.reader.cv2.imwrite"):
            result, _ = resources.read_resources_from_hud(["wood_stockpile"])

        self.assertIsNone(result["wood_stockpile"])
        img2str_mock.assert_not_called()


class TestWoodStockpileLowConfRetry(TestCase):
    def setUp(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources._LAST_READ_FROM_CACHE.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()

    def tearDown(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources._LAST_READ_FROM_CACHE.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()

    def test_low_conf_retry_accepted(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        def fake_detect(frame, required_icons, cache=None):
            return {"wood_stockpile": (0, 0, 10, 10)}

        calls = []

        def fake_execute(gray, conf_threshold=None, allow_fallback=True, roi=None, resource=None):
            calls.append(conf_threshold)
            return "999", {"conf": [conf_threshold or 0]}, None, True

        with patch("script.resources.reader.detect_resource_regions", side_effect=fake_detect), \
            patch("script.screen_utils._grab_frame", return_value=frame), \
            patch("script.resources.reader.preprocess_roi", side_effect=lambda roi: roi[..., 0]), \
            patch("script.resources.reader.execute_ocr", side_effect=fake_execute), \
            patch("script.resources.reader.cv2.imwrite"), \
            patch.dict(resources.CFG, {"treat_low_conf_as_failure": True}, clear=False):
            result, _ = resources.read_resources_from_hud(["wood_stockpile"])

        self.assertEqual(result["wood_stockpile"], 999)
        self.assertIn(
            "wood_stockpile", resources.RESOURCE_CACHE.last_low_confidence
        )
        self.assertEqual(
            calls,
            [
                resources.CFG.get(
                    "wood_stockpile_ocr_conf_threshold",
                    resources.CFG.get("ocr_conf_threshold", 60),
                ),
                resources.CFG.get("ocr_conf_min", 0),
            ],
        )

    def test_initial_low_conf_returns_none_and_triggers_retry(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        roi = np.zeros((1, 1, 3), dtype=np.uint8)
        gray = np.zeros((1, 1), dtype=np.uint8)
        roi_info = (0, 0, 1, 1, roi, gray, 0, 0)
        cache = resources.ResourceCache()

        with patch(
            "script.resources.reader.core._ocr_resource",
            return_value=("123", {}, None, True),
        ), patch(
            "script.resources.reader.core._retry_ocr",
            return_value=(
                "123",
                {},
                None,
                roi,
                gray,
                0,
                0,
                1,
                1,
                True,
            ),
        ), patch.dict(
            resources.CFG,
            {"allow_low_conf_digits": True},
            clear=False,
        ):
            value, _, _, _, _ = resources.core._process_resource(
                frame,
                "wood_stockpile",
                roi_info,
                cache_obj=cache,
                res_conf_threshold=60,
                max_cache_age=None,
                low_conf_counts={},
            )
        self.assertIsNone(value)
        self.assertEqual(cache.resource_failure_counts["wood_stockpile"], 1)

        roi_info2 = (0, 0, 1, 1, roi, gray, 0, cache.resource_failure_counts["wood_stockpile"])
        with patch(
            "script.resources.reader.core._ocr_resource",
            return_value=(None, {}, None, True),
        ), patch(
            "script.resources.reader.core.expand_roi_after_failure",
            return_value=(
                "456",
                {},
                None,
                roi,
                gray,
                0,
                0,
                1,
                1,
                False,
            ),
        ) as expand_mock, patch.dict(
            resources.CFG,
            {"allow_low_conf_digits": True},
            clear=False,
        ):
            value2, _, _, _, _ = resources.core._process_resource(
                frame,
                "wood_stockpile",
                roi_info2,
                cache_obj=cache,
                res_conf_threshold=60,
                max_cache_age=None,
                low_conf_counts={},
            )
        self.assertEqual(value2, 456)
        expand_mock.assert_called_once()
        self.assertEqual(expand_mock.call_args[0][9], 1)

