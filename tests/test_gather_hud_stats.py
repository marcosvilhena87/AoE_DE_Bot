import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import os
import sys

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
        threshold=lambda src, *a, **k: (None, src),
        rectangle=lambda img, pt1, pt2, color, thickness: img,
        IMREAD_GRAYSCALE=0,
        COLOR_BGR2GRAY=0,
    ),
)
os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

import tools.campaign_bot as cb


class TestGatherHudStats(TestCase):
    def setUp(self):
        # Clear caches in resources module used by campaign bot
        cb.resources.RESOURCE_CACHE.last_resource_values.clear()
        cb.resources.RESOURCE_CACHE.last_resource_ts.clear()
        cb.resources.RESOURCE_CACHE.resource_failure_counts.clear()

    def test_gather_reads_resources_and_population(self):
        anchor = {"left": 10, "top": 20, "width": 600, "height": 60, "asset": "assets/resources.png"}
        cb.HUD_ANCHOR = anchor.copy()

        digits_iter = iter(["100", "200", "300", "400", "600"])
        grab_calls = []
        roi_shapes = []
        pop_shapes = []

        def fake_grab_frame(bbox=None):
            grab_calls.append(bbox)
            return np.zeros((200, 800, 3), dtype=np.uint8)

        def fake_ocr(gray):
            roi_shapes.append(gray.shape[:2])
            d = next(digits_iter)
            return d, {"text": [d]}

        def fake_pop(roi, conf_threshold=None):
            pop_shapes.append(roi.shape[:2])
            return 123, 200

        with patch("tools.campaign_bot.locate_resource_panel", return_value={}), \
             patch("tools.campaign_bot._grab_frame", side_effect=fake_grab_frame), \
             patch("tools.campaign_bot._ocr_digits_better", side_effect=fake_ocr), \
             patch("tools.campaign_bot._read_population_from_roi", side_effect=fake_pop), \
             patch("tools.campaign_bot.resources.detect_resource_regions", return_value={
                 "wood_stockpile": (0, 0, 90, 52),
                 "food_stockpile": (90, 0, 90, 52),
                 "gold_stockpile": (180, 0, 90, 52),
                 "stone_stockpile": (270, 0, 90, 52),
                 "population_limit": (360, 0, 90, 52),
                 "idle_villager": (450, 0, 98, 52),
             }), \
             patch(
                 "tools.campaign_bot.resources.pytesseract.image_to_data",
                 return_value={"text": ["600"], "conf": ["90"]},
             ):
            res, pop = cb.gather_hud_stats()

        expected_res = {
            "wood_stockpile": 100,
            "food_stockpile": 200,
            "gold_stockpile": 300,
            "stone_stockpile": 400,
            "idle_villager": 600,
            "population_limit": 123,
        }
        self.assertEqual(res, expected_res)
        self.assertEqual(pop, (123, 200))
        expected_shapes = [(52, 90), (52, 90), (52, 90), (52, 90)]
        self.assertEqual(roi_shapes, expected_shapes)
        self.assertEqual(pop_shapes, [(52, 90)])
        self.assertEqual(grab_calls, [None])

    def test_missing_icons_become_optional(self):
        anchor = {"left": 10, "top": 20, "width": 600, "height": 60, "asset": "assets/resources.png"}
        cb.HUD_ANCHOR = anchor.copy()

        digits_iter = iter(["100", "200"])

        def fake_grab_frame(bbox=None):
            return np.zeros((200, 800, 3), dtype=np.uint8)

        def fake_ocr(gray):
            d = next(digits_iter)
            return d, {"text": [d]}

        with patch("tools.campaign_bot.locate_resource_panel", return_value={}), \
             patch("tools.campaign_bot._grab_frame", side_effect=fake_grab_frame), \
             patch("tools.campaign_bot._ocr_digits_better", side_effect=fake_ocr), \
             patch(
                 "tools.campaign_bot.resources.detect_resource_regions",
                 return_value={
                     "wood_stockpile": (0, 0, 50, 50),
                     "food_stockpile": (50, 0, 50, 50),
                 },
             ):
            res, pop = cb.gather_hud_stats()

        self.assertEqual(res, {"wood_stockpile": 100, "food_stockpile": 200})
        self.assertEqual(pop, (None, None))

    def test_scenario_defined_icons(self):
        anchor = {"left": 10, "top": 20, "width": 600, "height": 60, "asset": "assets/resources.png"}
        cb.HUD_ANCHOR = anchor.copy()

        def fake_grab_frame(bbox=None):
            return np.zeros((200, 800, 3), dtype=np.uint8)

        def fake_ocr(gray):
            return "100", {"text": ["100"]}

        with patch("tools.campaign_bot.locate_resource_panel", return_value={}), \
             patch("tools.campaign_bot._grab_frame", side_effect=fake_grab_frame), \
             patch("tools.campaign_bot._ocr_digits_better", side_effect=fake_ocr), \
             patch(
                 "tools.campaign_bot.resources.detect_resource_regions",
                 return_value={"wood_stockpile": (0, 0, 50, 50)},
             ):
            res, pop = cb.gather_hud_stats(
                required_icons=["wood_stockpile"],
                optional_icons=["food_stockpile"],
            )

        self.assertEqual(res, {"wood_stockpile": 100})
        self.assertEqual(pop, (None, None))
