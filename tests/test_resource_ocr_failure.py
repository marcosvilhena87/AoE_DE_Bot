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


class TestResourceOcrFailure(TestCase):
    def test_read_resources_fallback(self):
        def fake_grab_frame(bbox=None):
            if bbox:
                return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
            return np.zeros((600, 600, 3), dtype=np.uint8)

        def fake_ocr(gray):
            return "", {"text": [""]}, np.zeros((1, 1), dtype=np.uint8)

        with patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
             patch(
                 "script.common.locate_resource_panel",
                 return_value={
                     "wood_stockpile": (0, 0, 50, 50),
                     "food_stockpile": (50, 0, 50, 50),
                     "gold": (100, 0, 50, 50),
                     "stone": (150, 0, 50, 50),
                     "population": (200, 0, 50, 50),
                     "idle_villager": (250, 0, 50, 50),
                 },
             ), \
             patch("script.common._ocr_digits_better", side_effect=fake_ocr), \
             patch("script.common.pytesseract.image_to_string", return_value="123"):
            result = common.read_resources_from_hud()
            self.assertEqual(result["wood_stockpile"], 123)
