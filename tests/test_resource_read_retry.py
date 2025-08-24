import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import time

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


class TestResourceReadRetry(TestCase):
    def setUp(self):
        resources._LAST_RESOURCE_VALUES.clear()
        resources._LAST_RESOURCE_TS.clear()
        resources._LAST_READ_FROM_CACHE.clear()
        resources._RESOURCE_FAILURE_COUNTS.clear()

    def tearDown(self):
        resources._LAST_RESOURCE_VALUES.clear()
        resources._LAST_RESOURCE_TS.clear()
        resources._LAST_READ_FROM_CACHE.clear()
        resources._RESOURCE_FAILURE_COUNTS.clear()

    def test_required_icon_fallback(self):
        def fake_detect(frame, required_icons):
            return {"wood_stockpile": (0, 0, 50, 50)}

        ocr_seq = [
            ("123", {"text": ["123"]}, np.zeros((1, 1), dtype=np.uint8)),
            ("", {"text": [""]}, np.zeros((1, 1), dtype=np.uint8)),
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

        self.assertEqual(first["wood_stockpile"], 123)
        self.assertEqual(second["wood_stockpile"], 123)
        self.assertIn("wood_stockpile", resources._LAST_READ_FROM_CACHE)

    def test_retry_succeeds_after_expansion(self):
        def fake_detect(frame, required_icons):
            return {"wood_stockpile": (0, 0, 50, 50)}

        ocr_seq = [
            ("", {"text": [""]}, np.zeros((1, 1), dtype=np.uint8)),
            ("456", {"text": ["456"]}, np.zeros((1, 1), dtype=np.uint8)),
        ]

        def fake_ocr(gray):
            return ocr_seq.pop(0)

        frame = np.zeros((600, 600, 3), dtype=np.uint8)

        with patch("script.resources.detect_resource_regions", side_effect=fake_detect), \
             patch("script.screen_utils._grab_frame", return_value=frame), \
             patch("script.resources._ocr_digits_better", side_effect=fake_ocr) as ocr_mock, \
             patch("script.resources.pytesseract.image_to_string", return_value=""), \
             patch("script.resources.cv2.imwrite"):
            result = resources.read_resources_from_hud(["wood_stockpile"])

        self.assertEqual(result["wood_stockpile"], 456)
        self.assertEqual(ocr_mock.call_count, 2)

    def test_expired_cache_used_after_consecutive_failures(self):
        def fake_detect(frame, required_icons):
            return {"wood_stockpile": (0, 0, 50, 50)}

        def fake_ocr(gray):
            return "", {"text": [""]}, np.zeros((1, 1), dtype=np.uint8)

        frame = np.zeros((600, 600, 3), dtype=np.uint8)

        resources._LAST_RESOURCE_VALUES["wood_stockpile"] = 999
        resources._LAST_RESOURCE_TS["wood_stockpile"] = (
            time.time() - (resources._RESOURCE_CACHE_TTL + 10)
        )

        with patch("script.resources.detect_resource_regions", side_effect=fake_detect), \
             patch("script.screen_utils._grab_frame", return_value=frame), \
             patch("script.resources._ocr_digits_better", side_effect=fake_ocr), \
             patch("script.resources.pytesseract.image_to_string", return_value=""), \
             patch("script.resources.cv2.imwrite"):
            with self.assertRaises(common.ResourceReadError):
                resources.read_resources_from_hud(["wood_stockpile"])
            result = resources.read_resources_from_hud(["wood_stockpile"])

        self.assertEqual(result["wood_stockpile"], 999)
        self.assertIn("wood_stockpile", resources._LAST_READ_FROM_CACHE)
