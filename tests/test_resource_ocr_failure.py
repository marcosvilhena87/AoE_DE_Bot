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


class TestResourceOcrFailure(TestCase):
    def setUp(self):
        resources._LAST_RESOURCE_VALUES.clear()

    def tearDown(self):
        resources._LAST_RESOURCE_VALUES.clear()
    def test_read_resources_fallback(self):
        def fake_grab_frame(bbox=None):
            if bbox:
                return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
            return np.zeros((600, 600, 3), dtype=np.uint8)

        def fake_ocr(gray):
            return "", {"text": [""]}, np.zeros((1, 1), dtype=np.uint8)

        with patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
             patch(
                 "script.resources.locate_resource_panel",
                 return_value={
                     "wood_stockpile": (0, 0, 50, 50),
                     "food_stockpile": (50, 0, 50, 50),
                     "gold": (100, 0, 50, 50),
                     "stone": (150, 0, 50, 50),
                     "population": (200, 0, 50, 50),
                     "idle_villager": (250, 0, 50, 50),
                 },
             ), \
             patch("script.resources._ocr_digits_better", side_effect=fake_ocr), \
             patch("script.resources.pytesseract.image_to_string", return_value="123"):
            result = resources.read_resources_from_hud()
            self.assertEqual(result["wood_stockpile"], 123)

    def test_optional_icon_failure_does_not_raise(self):
        def fake_grab_frame(bbox=None):
            if bbox:
                return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
            return np.zeros((600, 600, 3), dtype=np.uint8)

        def fake_detect(frame, required_icons):
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

        with patch("script.resources.detect_resource_regions", side_effect=fake_detect), \
             patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
             patch("script.resources._ocr_digits_better", side_effect=fake_ocr), \
             patch("script.resources.pytesseract.image_to_string", return_value=""), \
             patch("script.resources.cv2.imwrite"):
            result = resources.read_resources_from_hud(["wood_stockpile"])
        self.assertEqual(result.get("wood_stockpile"), 123)
        self.assertIsNone(result.get("food_stockpile"))

    def test_low_confidence_triggers_failure(self):
        def fake_grab_frame(bbox=None):
            if bbox:
                return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
            return np.zeros((600, 600, 3), dtype=np.uint8)

        def fake_detect(frame, required_icons):
            return {"wood_stockpile": (0, 0, 50, 50)}

        def fake_ocr(gray):
            data = {"text": ["123"], "conf": ["10", "20", "30"]}
            return "123", data, np.zeros((1, 1), dtype=np.uint8)

        with patch("script.resources.detect_resource_regions", side_effect=fake_detect), \
             patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
             patch("script.resources._ocr_digits_better", side_effect=fake_ocr), \
             patch("script.resources.pytesseract.image_to_string", return_value="") as img2str_mock, \
             patch("script.resources.cv2.imwrite"), \
             self.assertRaises(common.ResourceReadError):
            resources.read_resources_from_hud(["wood_stockpile"])
        img2str_mock.assert_called_once()

    def test_cached_value_used_for_optional_failure(self):
        def fake_detect(frame, required_icons):
            return {
                "wood_stockpile": (0, 0, 50, 50),
                "food_stockpile": (50, 0, 50, 50),
            }

        ocr_seq = [
            ("123", {"text": ["123"]}, np.zeros((1, 1), dtype=np.uint8)),
            ("234", {"text": ["234"]}, np.zeros((1, 1), dtype=np.uint8)),
            ("345", {"text": ["345"]}, np.zeros((1, 1), dtype=np.uint8)),
            ("", {"text": [""]}, np.zeros((1, 1), dtype=np.uint8)),
        ]

        def fake_ocr(gray):
            return ocr_seq.pop(0)

        frame = np.zeros((600, 600, 3), dtype=np.uint8)

        with patch("script.resources.detect_resource_regions", side_effect=fake_detect), \
             patch("script.screen_utils._grab_frame", return_value=frame), \
             patch("script.resources._ocr_digits_better", side_effect=fake_ocr), \
             patch("script.resources.pytesseract.image_to_string", return_value=""), \
             patch("script.resources.cv2.imwrite"):
            first = resources.read_resources_from_hud(["wood_stockpile"])
            second = resources.read_resources_from_hud(["wood_stockpile"])

        self.assertNotIn("food_stockpile", first)
        self.assertNotIn("food_stockpile", second)
