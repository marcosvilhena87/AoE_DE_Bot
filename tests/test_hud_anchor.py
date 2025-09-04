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

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.common as common
import script.hud as hud
import script.resources.reader as resources
import tools.campaign as cb

ASSET = "assets/resources.png"


class TestHudAnchor(TestCase):
    def setUp(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()

    def test_wait_hud_raises_without_template(self):
        with patch.object(hud.screen_utils, "HUD_TEMPLATE", None), \
            patch("script.screen_utils.grab_frame") as grab_mock:
            with self.assertRaises(RuntimeError) as ctx:
                hud.wait_hud(timeout=1)
        grab_mock.assert_not_called()
        self.assertIn(ASSET, str(ctx.exception))

    def test_wait_hud_sets_asset(self):
        common.HUD_ANCHOR = None
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch("script.screen_utils.grab_frame", return_value=fake_frame), \
             patch("script.hud.find_template", return_value=((10, 20, 30, 40), 0.9, None)):
            anchor, asset = hud.wait_hud(timeout=1)
        self.assertEqual(asset, ASSET)
        self.assertEqual(anchor["asset"], ASSET)
        self.assertEqual(common.HUD_ANCHOR["asset"], ASSET)

    def test_read_resources_uses_anchor_slices(self):
        anchor = {"left": 10, "top": 20, "width": 600, "height": 60, "asset": ASSET}
        common.HUD_ANCHOR = anchor.copy()

        digits_iter = iter(["100", "200", "300", "400", "500", "600"])
        grab_calls = []
        roi_shapes = []

        def fake_grab_frame(bbox=None):
            grab_calls.append(bbox)
            return np.zeros((200, 800, 3), dtype=np.uint8)

        def fake_ocr(gray):
            roi_shapes.append(gray.shape[:2])
            d = next(digits_iter)
            return d, {"text": [d]}, np.zeros((1, 1), dtype=np.uint8)

        with patch("script.resources.reader.locate_resource_panel", return_value={}), \
            patch("script.screen_utils.grab_frame", side_effect=fake_grab_frame), \
             patch("script.resources.reader._ocr_digits_better", side_effect=fake_ocr), \
             patch("script.resources.reader.preprocess_roi", side_effect=lambda roi: np.zeros((52, 90), dtype=np.uint8)), \
             patch(
                 "script.resources.reader.pytesseract.image_to_data",
                 return_value={"text": ["600"], "conf": ["90"]},
             ), \
             patch("script.resources.reader._read_population_from_roi", return_value=(500, 500, False)), \
             patch("script.input_utils._screen_size", return_value=(1920, 1080)), \
             patch("script.resources.reader.cv2.imwrite"):
            result, _ = resources.read_resources_from_hud()

        expected = {
            "wood_stockpile": 100,
            "food_stockpile": 200,
            "gold_stockpile": 300,
            "stone_stockpile": 400,
            "population_limit": 500,
            "idle_villager": 600,
        }
        self.assertEqual(result, expected)

        expected_boxes = [
            {"left": 35, "top": 24, "width": 73, "height": 50},
            {"left": 130, "top": 24, "width": 78, "height": 50},
            {"left": 230, "top": 24, "width": 78, "height": 50},
            {"left": 330, "top": 24, "width": 78, "height": 50},
            {"left": 430, "top": 24, "width": 78, "height": 50},
            {"left": 530, "top": 24, "width": 78, "height": 50},
        ]
        expected_shapes = [
            (52, 90),
            (51, 90),
            (51, 90),
            (51, 90),
        ]
        self.assertEqual(roi_shapes, expected_shapes)
        self.assertEqual(grab_calls, [None])


class TestHudAnchorTools(TestCase):
    def test_wait_hud_sets_asset(self):
        cb.HUD_ANCHOR = None
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch("tools.campaign.grab_frame", return_value=fake_frame), \
             patch("tools.campaign.find_template", return_value=((10, 20, 30, 40), 0.9, None)):
            anchor, asset = cb.wait_hud(timeout=1)
        self.assertEqual(asset, ASSET)
        self.assertEqual(anchor["asset"], ASSET)
        self.assertEqual(cb.HUD_ANCHOR["asset"], ASSET)

    def test_read_resources_uses_anchor_slices(self):
        anchor = {"left": 10, "top": 20, "width": 600, "height": 60, "asset": ASSET}
        cb.HUD_ANCHOR = anchor.copy()

        digits_iter = iter(["100", "200", "300", "400", "500", "600"])
        grab_calls = []
        roi_shapes = []

        def fake_grab_frame(bbox=None):
            grab_calls.append(bbox)
            return np.zeros((200, 800, 3), dtype=np.uint8)

        def fake_ocr(gray):
            roi_shapes.append(gray.shape[:2])
            d = next(digits_iter)
            return d, {"text": [d]}

        with patch("tools.campaign.locate_resource_panel", return_value={}), \
            patch("tools.campaign.grab_frame", side_effect=fake_grab_frame), \
             patch("tools.campaign._ocr_digits_better", side_effect=fake_ocr), \
             patch("script.resources.reader._ocr_digits_better", side_effect=fake_ocr), \
             patch("script.resources.reader.preprocess_roi", side_effect=lambda roi: np.zeros((52, 90), dtype=np.uint8)), \
             patch(
                 "script.resources.reader.pytesseract.image_to_data",
                 return_value={"text": ["600"], "conf": ["90"]},
             ), \
             patch("script.resources.reader._read_population_from_roi", return_value=(500, 500, False)), \
             patch("script.input_utils._screen_size", return_value=(1920, 1080)), \
             patch("script.resources.reader.cv2.imwrite"):
            result, _ = cb.read_resources_from_hud([
                "wood_stockpile",
                "food_stockpile",
                "gold_stockpile",
                "stone_stockpile",
                "population_limit",
                "idle_villager",
            ])

        expected = {
            "wood_stockpile": 100,
            "food_stockpile": 200,
            "gold_stockpile": 300,
            "stone_stockpile": 400,
            "population_limit": 500,
            "idle_villager": 600,
        }
        self.assertEqual(result, expected)

        expected_boxes = [
            {"left": 35, "top": 24, "width": 73, "height": 50},
            {"left": 130, "top": 24, "width": 78, "height": 50},
            {"left": 230, "top": 24, "width": 78, "height": 50},
            {"left": 330, "top": 24, "width": 78, "height": 50},
            {"left": 430, "top": 24, "width": 78, "height": 50},
            {"left": 530, "top": 24, "width": 78, "height": 50},
        ]
        expected_shapes = [
            (52, 90),
            (51, 90),
            (51, 90),
            (51, 90),
        ]
        self.assertEqual(roi_shapes, expected_shapes)
        self.assertEqual(grab_calls, [None])
