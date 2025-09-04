import numpy as np
from unittest import TestCase
from unittest.mock import patch

import script.resources as resources
from script.resources.panel import _apply_custom_rois


class TestResourceCustomROIs(TestCase):
    def test_roi_matches_configured_percentages_when_missing(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        cfg = {
            "left_pct": 0.2,
            "top_pct": 0.3,
            "width_pct": 0.25,
            "height_pct": 0.25,
        }
        expected = (40, 60, 50, 50)
        names = [
            "wood_stockpile",
            "food_stockpile",
            "gold_stockpile",
            "stone_stockpile",
            "population_limit",
        ]
        with patch("script.screen_utils.get_screen_size", return_value=(200, 200)):
            for name in names:
                with self.subTest(name=name):
                    key = f"{name}_roi"
                    with patch.dict(resources.CFG, {key: cfg}, clear=False):
                        regions = _apply_custom_rois(frame, {})
                    self.assertEqual(regions[name], expected)
