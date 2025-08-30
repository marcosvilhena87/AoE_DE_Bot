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

class TestNoDigits(TestCase):
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

    def test_read_resources_fallback(self):
        def fake_ocr(gray):
            return "", {"text": [""]}, np.zeros((1, 1), dtype=np.uint8)

        frame = np.zeros((600, 600, 3), dtype=np.uint8)

        resources.RESOURCE_CACHE.last_resource_values["wood_stockpile"] = 0
        with patch.dict(resources.CFG, {"wood_stockpile_low_conf_fallback": False}, clear=False), \
            patch(
                "script.resources.reader.detect_resource_regions",
                return_value={
                    "wood_stockpile": (0, 0, 50, 50),
                    "food_stockpile": (50, 0, 50, 50),
                    "gold_stockpile": (100, 0, 50, 50),
                    "stone_stockpile": (150, 0, 50, 50),
                    "population_limit": (200, 0, 50, 50),
                    "idle_villager": (250, 0, 50, 50),
                },
            ), patch("script.resources.ocr.masks._ocr_digits_better", side_effect=fake_ocr), patch(
            "script.resources.reader.pytesseract.image_to_data",
            return_value={"text": [""], "conf": ["0"]},
        ), patch(
            "script.resources.reader.pytesseract.image_to_string", return_value="123"
        ), patch("script.resources.ocr.executor._read_population_from_roi", return_value=(0, 0)):
            icons = resources.RESOURCE_ICON_ORDER[:-1]
            result, _ = resources._read_resources(
                frame,
                icons,
                icons,
            )
        self.assertIsNone(result["wood_stockpile"])

    def test_optional_icon_failure_does_not_raise(self):
        def fake_grab_frame(bbox=None):
            if bbox:
                return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
            return np.zeros((600, 600, 3), dtype=np.uint8)

        def fake_detect(frame, required_icons, cache=None):
            return {
                "wood_stockpile": (0, 0, 50, 50),
                "food_stockpile": (50, 0, 50, 50),
            }

        ocr_seq = [
            ("123", {"text": ["123"]}, np.zeros((1, 1), dtype=np.uint8)),
            ("", {"text": [""]}, np.zeros((1, 1), dtype=np.uint8)),
        ]

        def fake_ocr(gray):
            return ocr_seq.pop(0)

        with patch("script.resources.reader.detect_resource_regions", side_effect=fake_detect), \
             patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
             patch("script.resources.ocr.masks._ocr_digits_better", side_effect=fake_ocr), \
             patch("script.resources.reader.pytesseract.image_to_string", return_value=""), \
             patch("script.resources.reader.cv2.imwrite"):
            result, _ = resources.read_resources_from_hud(["wood_stockpile"])
        self.assertEqual(result.get("wood_stockpile"), 123)
        self.assertIsNone(result.get("food_stockpile"))

    def test_zero_roi_returns_zero(self):
        gray = np.full((20, 20), 128, dtype=np.uint8)
        with patch(
            "script.resources.reader.pytesseract.image_to_data",
            return_value={"text": [""], "conf": ["-1"]},
        ):
            digits, data, _ = resources._ocr_digits_better(gray)
        self.assertEqual(digits, "0")
        self.assertTrue(data.get("zero_variance"))

    def test_gold_and_stone_zero_digits_return_zero(self):
        def make_gold_roi():
            roi = np.full((10, 10), 210, dtype=np.uint8)
            roi[2:-2, 2] = 200
            roi[2:-2, -3] = 200
            roi[2, 2:-2] = 200
            roi[-3, 2:-2] = 200
            return roi

        def make_stone_roi():
            roi = np.full((10, 10), 180, dtype=np.uint8)
            roi[2:-2, 2] = 170
            roi[2:-2, -3] = 170
            roi[2, 2:-2] = 170
            roi[-3, 2:-2] = 170
            return roi

        with patch(
            "script.resources.reader.pytesseract.image_to_data",
            return_value={"text": [""], "conf": ["-1"]},
        ), patch.dict(resources.CFG, {"ocr_zero_variance": 50}, clear=False):
            gold, _, _ = resources._ocr_digits_better(make_gold_roi())
            stone, _, _ = resources._ocr_digits_better(make_stone_roi())
        self.assertEqual(gold, "0")
        self.assertEqual(stone, "0")
