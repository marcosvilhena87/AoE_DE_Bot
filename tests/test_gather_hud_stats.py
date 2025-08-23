import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np

import tools.campaign_bot as cb


class TestGatherHudStats(TestCase):
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
             patch("tools.campaign_bot._read_population_from_roi", side_effect=fake_pop):
            res, pop = cb.gather_hud_stats()

        expected_res = {
            "wood_stockpile": 100,
            "food_stockpile": 200,
            "gold": 300,
            "stone": 400,
            "idle_villager": 600,
            "population": 123,
        }
        self.assertEqual(res, expected_res)
        self.assertEqual(pop, (123, 200))
        expected_shapes = [(50, 90)] * 5
        self.assertEqual(roi_shapes, expected_shapes)
        self.assertEqual(pop_shapes, [(50, 90)])
        self.assertEqual(grab_calls, [None])
