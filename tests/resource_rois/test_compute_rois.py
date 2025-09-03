import numpy as np
from unittest import TestCase
from unittest.mock import patch

import script.common as common
import script.resources as resources
import script.resources.reader as reader
import script.screen_utils as screen_utils


class TestComputeResourceROIs(TestCase):
    def test_build_rois_reduce_width_when_gap_below_min(self):
        detected = {
            "wood_stockpile": (0, 0, 5, 5),
            "food_stockpile": (15, 0, 5, 5),
        }
        regions, spans, narrow = resources.compute_resource_rois(
            0,
            100,
            0,
            10,
            [2] * 6,
            [2] * 6,
            [0] * 6,
            [999] * 6,
            [20] * 6,
            0,
            0,
            detected=detected,
        )
        roi = regions["wood_stockpile"]
        span_left = 6  # icon_right + pad_left after adjustment
        span_right = 15  # next_icon_left - pad_right after adjustment
        pad_extra = int(round(detected["wood_stockpile"][2] * 0.25))
        self.assertEqual(roi[0], span_left - pad_extra)
        self.assertEqual(roi[0] + roi[2], span_right + pad_extra)
        self.assertEqual(roi[2], span_right - span_left + 2 * pad_extra)
        self.assertIn("wood_stockpile", narrow)
        self.assertEqual(narrow["wood_stockpile"], 11)

    def test_ignores_non_positive_span(self):
        detected = {
            "wood_stockpile": (0, 0, 5, 5),
            "food_stockpile": (6, 0, 5, 5),
        }
        regions, spans, narrow = resources.compute_resource_rois(
            0,
            100,
            0,
            10,
            [2] * 6,
            [2] * 6,
            [0] * 6,
            [999] * 6,
            [0] * 6,
            0,
            0,
            detected=detected,
        )
        self.assertNotIn("wood_stockpile", regions)
        self.assertNotIn("wood_stockpile", spans)
        self.assertNotIn("wood_stockpile", narrow)
        self.assertIn("food_stockpile", spans)

    def test_per_icon_min_width(self):
        frame = np.zeros((50, 100, 3), dtype=np.uint8)
        panel_box = (0, 0, 100, 20)

        icons = ["wood_stockpile", "food_stockpile", "gold_stockpile"]
        positions = [0, 39, 78]
        pad_left = 2
        pad_right = 2
        min_widths = [40, 10, 0, 0, 0, 0]
        loc_iter = iter([(x, 0) for x in positions])

        def fake_minmax(res):
            xi, yi = next(loc_iter)
            return 0.0, 0.95, (0, 0), (xi, yi)

        with patch("script.resources.find_template", return_value=(panel_box, 0.9, None)), \
            patch("script.resources.cv2.cvtColor", lambda src, code: np.zeros(src.shape[:2], dtype=np.uint8)), \
            patch("script.resources.cv2.resize", lambda img, *a, **k: img), \
            patch("script.resources.cv2.matchTemplate", lambda *a, **k: np.zeros((100, 200), dtype=np.float32)), \
            patch("script.resources.cv2.minMaxLoc", side_effect=fake_minmax), \
            patch.object(screen_utils, "_load_icon_templates", lambda: None), \
            patch.object(screen_utils, "HUD_TEMPLATE", np.zeros((1, 1), dtype=np.uint8)), \
            patch.dict(screen_utils.ICON_TEMPLATES, {name: np.zeros((5, 5), dtype=np.uint8) for name in icons}, clear=True), \
            patch.dict(common.CFG["resource_panel"], {
                "roi_padding_left": [pad_left] * 6,
                "roi_padding_right": [pad_right] * 6,
                "icon_trim_pct": [0] * 6,
                "scales": [1.0],
                "match_threshold": 0.5,
                "max_width": 999,
                "min_width": min_widths,
                "top_pct": 0.0,
                "height_pct": 1.0,
            }), patch.dict(
                common.CFG["profiles"]["aoe1de"]["resource_panel"],
                {"icon_trim_pct": [0] * 6},
            ):
            resources.cache._NARROW_ROIS.clear()
            resources.cache._NARROW_ROI_DEFICITS.clear()
            resources.RESOURCE_CACHE.resource_failure_counts.clear()
            resources.RESOURCE_CACHE.last_resource_values.clear()
            resources.RESOURCE_CACHE.last_resource_ts.clear()
            resources.locate_resource_panel(frame)
            result = resources.cache._NARROW_ROIS.copy()
            resources.cache._NARROW_ROIS.clear()
            resources.cache._NARROW_ROI_DEFICITS.clear()
            resources.RESOURCE_CACHE.resource_failure_counts.clear()
            resources.RESOURCE_CACHE.last_resource_values.clear()
            resources.RESOURCE_CACHE.last_resource_ts.clear()

        self.assertEqual(result, {"wood_stockpile"})

    def test_food_roi_accommodates_three_digits(self):
        detected = {
            "wood_stockpile": (0, 0, 10, 10),
            "food_stockpile": (50, 0, 10, 10),
            "gold_stockpile": (100, 0, 10, 10),
            "stone_stockpile": (150, 0, 10, 10),
            "population_limit": (200, 0, 10, 10),
            "idle_villager": (250, 0, 10, 10),
        }
        regions, spans, narrow = resources.compute_resource_rois(
            0,
            320,
            0,
            20,
            [0, 3, 0, 0, 0, 3],
            [4, 3, 0, 0, 0, 3],
            [0] * 6,
            [999] * 6,
            [30] * 6,
            0,
            0,
            detected=detected,
        )
        self.assertGreaterEqual(regions["food_stockpile"][2], 30)
        self.assertNotIn("food_stockpile", narrow)

    def test_food_roi_width_can_exceed_60_when_configured(self):
        detected = {
            "food_stockpile": (0, 0, 10, 10),
            "gold_stockpile": (200, 0, 10, 10),
        }
        with patch.dict(resources.CFG, {"food_stockpile_max_width": 80}, clear=False):
            regions, _spans, _narrow = resources.compute_resource_rois(
                0,
                300,
                0,
                20,
                [0] * 6,
                [0] * 6,
                [0] * 6,
                [999] * 6,
                [0] * 6,
                0,
                0,
                detected=detected,
            )
        self.assertGreater(regions["food_stockpile"][2], 60)

    def test_population_roi_respects_min_width(self):
        detected = {
            "population_limit": (0, 0, 5, 5),
            "idle_villager": (200, 0, 5, 5),
        }
        min_pop_width = 50
        regions, _spans, _narrow = resources.compute_resource_rois(
            0,
            300,
            0,
            10,
            [2] * 6,
            [2] * 6,
            [0] * 6,
            [999] * 6,
            [0] * 6,
            min_pop_width,
            0,
            [0] * 6,
            detected=detected,
        )
        self.assertGreaterEqual(regions["population_limit"][2], min_pop_width)

    def test_population_roi_excludes_nearby_idle_villager(self):
        detected = {
            "population_limit": (0, 0, 5, 5),
            "idle_villager": (15, 0, 5, 5),
        }
        min_pop_width = 10
        with patch.dict(resources.CFG, {"population_idle_padding": 6}, clear=False):
            regions, _spans, _narrow = resources.compute_resource_rois(
                0,
                40,
                0,
                10,
                [2] * 6,
                [2] * 6,
                [0] * 6,
                [999] * 6,
                [0] * 6,
                min_pop_width,
                0,
                [0] * 6,
                detected=detected,
            )
        roi = regions["population_limit"]
        left, _, width, _ = roi
        right = left + width
        idle_left = detected["idle_villager"][0]
        self.assertLessEqual(right, idle_left - 6)
        self.assertLess(width, min_pop_width)

    def test_population_roi_captures_three_four_digits(self):
        detected = {
            "population_limit": (0, 0, 10, 10),
            "idle_villager": (100, 0, 10, 10),
        }
        regions, _spans, _narrow = resources.compute_resource_rois(
            0,
            200,
            0,
            20,
            [0] * 6,
            [0] * 6,
            [0] * 6,
            [999] * 6,
            [0] * 6,
            0,
            0,
            detected=detected,
        )
        roi = regions["population_limit"]
        left, _, width, _ = roi
        right = left + width
        idle_left = detected["idle_villager"][0]
        self.assertGreaterEqual(width, 30)
        self.assertLessEqual(
            right, idle_left - common.CFG.get("population_idle_padding", 6)
        )

    def test_idle_roi_uses_icon_bounds_when_span_non_positive(self):
        detected = {
            "population_limit": (0, 0, 5, 5),
            "idle_villager": (10, 0, 5, 5),
        }
        regions, spans, _narrow = resources.compute_resource_rois(
            0,
            40,
            0,
            10,
            [0, 0, 0, 0, 0, 70],
            [0] * 6,
            [0] * 6,
            [999] * 6,
            [0] * 6,
            0,
            10,
            detected=detected,
        )
        self.assertIn("idle_villager", regions)
        self.assertEqual(regions["idle_villager"], (10, 0, 5, 10))
        self.assertEqual(spans["idle_villager"], (10, 15))


class TestNarrowROIExpansion(TestCase):
    def setUp(self):
        reader.RESOURCE_CACHE.last_resource_values.clear()
        reader.RESOURCE_CACHE.last_resource_ts.clear()
        reader.RESOURCE_CACHE.resource_failure_counts.clear()
        reader._LAST_REGION_SPANS.clear()
        reader._NARROW_ROI_DEFICITS.clear()
        reader._NARROW_ROIS.clear()

    def tearDown(self):
        reader.RESOURCE_CACHE.last_resource_values.clear()
        reader.RESOURCE_CACHE.last_resource_ts.clear()
        reader.RESOURCE_CACHE.resource_failure_counts.clear()
        reader._LAST_REGION_SPANS.clear()
        reader._NARROW_ROI_DEFICITS.clear()
        reader._NARROW_ROIS.clear()

    def test_stone_stockpile_expands_narrow_roi(self):
        frame = np.zeros((40, 200, 3), dtype=np.uint8)
        regions = {"stone_stockpile": (50, 10, 40, 20)}

        def fake_detect(frame, required_icons, cache=None):
            reader._NARROW_ROIS.clear()
            reader._NARROW_ROIS.add("stone_stockpile")
            reader._NARROW_ROI_DEFICITS.clear()
            reader._NARROW_ROI_DEFICITS["stone_stockpile"] = 20
            reader._LAST_REGION_SPANS = {"stone_stockpile": (50, 90)}
            return regions

        with patch("script.resources.reader.core.detect_resource_regions", side_effect=fake_detect), \
            patch("script.resources.reader.core.execute_ocr", return_value=("123", {"text": ["123"]}, np.zeros((1, 1), dtype=np.uint8), False)) as ocr_mock:
            results, _ = reader._read_resources(
                frame, ["stone_stockpile"], ["stone_stockpile"]
            )

        self.assertEqual(ocr_mock.call_args[1]["roi"], (40, 10, 60, 20))
        self.assertEqual(results["stone_stockpile"], 123)

