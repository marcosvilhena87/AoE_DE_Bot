import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np

# Stub modules that require a GUI/display

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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.common as common


class TestHudAnchor(TestCase):
    def test_wait_hud_sets_asset(self):
        common.HUD_ANCHOR = None
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch("script.common._grab_frame", return_value=fake_frame), \
             patch("script.common.find_template", return_value=((10, 20, 30, 40), 0.9, None)):
            anchor, asset = common.wait_hud(timeout=1)
        self.assertEqual(asset, "assets/resources.png")
        self.assertEqual(anchor["asset"], "assets/resources.png")
        self.assertEqual(common.HUD_ANCHOR["asset"], "assets/resources.png")

    def test_read_resources_uses_anchor_slices(self):
        anchor = {"left": 10, "top": 20, "width": 600, "height": 60, "asset": "assets/resources.png"}
        common.HUD_ANCHOR = anchor.copy()

        digits_iter = iter(["100", "200", "300", "400", "500", "600"])
        grab_calls = []

        def fake_grab_frame(bbox=None):
            if bbox:
                grab_calls.append(bbox.copy())
                h, w = bbox["height"], bbox["width"]
                return np.zeros((h, w, 3), dtype=np.uint8)
            return np.zeros((200, 200, 3), dtype=np.uint8)

        def fake_ocr(gray):
            d = next(digits_iter)
            return d, {"text": [d]}, np.zeros((1, 1), dtype=np.uint8)

        with patch("script.common.locate_resource_panel", return_value={}), \
             patch("script.common._grab_frame", side_effect=fake_grab_frame), \
             patch("script.common._ocr_digits_better", side_effect=fake_ocr):
            result = common.read_resources_from_hud()

        expected = {
            "food": 100,
            "wood": 200,
            "gold": 300,
            "stone": 400,
            "population": 500,
            "idle_villager": 600,
        }
        self.assertEqual(result, expected)

        expected_boxes = [
            {"left": 28, "top": 24, "width": 80, "height": 50},
            {"left": 128, "top": 24, "width": 80, "height": 50},
            {"left": 228, "top": 24, "width": 80, "height": 50},
            {"left": 328, "top": 24, "width": 80, "height": 50},
            {"left": 428, "top": 24, "width": 80, "height": 50},
            {"left": 528, "top": 24, "width": 80, "height": 50},
        ]
        self.assertEqual(grab_calls, expected_boxes)
