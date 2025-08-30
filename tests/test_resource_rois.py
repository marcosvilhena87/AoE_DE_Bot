import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np

# Stub modules requiring GUI

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
        rectangle=lambda img, pt1, pt2, color, thickness: img,
        threshold=lambda src, *a, **k: (None, src),
        IMREAD_GRAYSCALE=0,
        COLOR_BGR2GRAY=0,
        INTER_LINEAR=0,
        THRESH_BINARY=0,
        THRESH_OTSU=0,
        TM_CCOEFF_NORMED=0,
    ),
)

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.common as common
import script.resources as resources
import script.screen_utils as screen_utils


class TestResourceROIs(TestCase):
    icons = [
        "wood_stockpile",
        "food_stockpile",
        "gold_stockpile",
        "stone_stockpile",
        "population_limit",
        "idle_villager",
    ]
    positions = [0, 30, 60, 90, 120, 150]
    pad_left = 2
    pad_right = 2
    panel_box = (0, 0, 200, 20)
    frame = np.zeros((50, 200, 3), dtype=np.uint8)

    def _build_synthetic_frame(self):
        icon_w = 5
        icon_h = 5
        icon_color = 255
        frame = np.zeros((20, 200, 3), dtype=np.uint8)
        for pos in self.positions:
            frame[0:icon_h, pos : pos + icon_w] = icon_color
        digits = [1, 2, 3, 4, 5]
        for idx in range(4):
            start = self.positions[idx] + icon_w
            end = self.positions[idx + 1]
            frame[:, start:end] = digits[idx]
        pop_start = self.positions[4] + icon_w
        pop_end = self.positions[5]
        frame[:, pop_start:pop_end] = digits[4]
        self.frame = frame
        self.icon_color = icon_color
        self.digit_values = {self.icons[i]: digits[i] for i in range(5)}

    def _locate_regions(self, icon_trim_pct=None):
        loc_iter = iter([(x, 0) for x in self.positions])

        def fake_minmax(res):
            xi, yi = next(loc_iter)
            return 0.0, 0.95, (0, 0), (xi, yi)

        trim = icon_trim_pct if icon_trim_pct is not None else [0] * 6
        self.trim = trim

        with patch(
            "script.resources.find_template", return_value=(self.panel_box, 0.9, None)
        ), patch(
            "script.resources.cv2.cvtColor",
            lambda src, code: np.zeros(src.shape[:2], dtype=np.uint8),
        ), patch(
            "script.resources.cv2.resize", lambda img, *a, **k: img
        ), patch(
            "script.resources.cv2.matchTemplate",
            lambda *a, **k: np.zeros((100, 200), dtype=np.float32),
        ), patch(
            "script.resources.cv2.minMaxLoc", side_effect=fake_minmax
        ), patch.object(
            screen_utils, "_load_icon_templates", lambda: None
        ), patch.object(
            screen_utils, "HUD_TEMPLATE", np.zeros((1, 1), dtype=np.uint8)
        ), patch.dict(
            screen_utils.ICON_TEMPLATES,
            {name: np.zeros((5, 5), dtype=np.uint8) for name in self.icons},
            clear=True,
        ), patch.dict(
            common.CFG["resource_panel"],
            {
                "roi_padding_left": [self.pad_left] * 6,
                "roi_padding_right": [self.pad_right] * 6,
                "icon_trim_pct": trim,
                "scales": [1.0],
                "match_threshold": 0.5,
                "max_width": 999,
                "min_width": 0,
                "top_pct": 0.0,
                "height_pct": 1.0,
            },
        ), patch.dict(
            common.CFG["profiles"]["aoe1de"]["resource_panel"],
            {
                "icon_trim_pct": trim,
            },
        ):
            regions = resources.locate_resource_panel(self.frame)
            icon_width = screen_utils.ICON_TEMPLATES[self.icons[0]].shape[1]

        return regions, icon_width

    def _assert_bounds(self, regions, icon_width, index):
        name = self.icons[index]
        roi = regions[name]
        left = roi[0]
        right = roi[0] + roi[2]
        icon_left = self.panel_box[0] + self.positions[index]
        cur_trim_val = self.trim[index] if index < len(self.trim) else self.trim[-1]
        if 0 <= cur_trim_val <= 1:
            cur_trim_px = int(round(cur_trim_val * icon_width))
        else:
            cur_trim_px = int(round(cur_trim_val))
        icon_right_trimmed = icon_left + icon_width - cur_trim_px
        next_icon_left = self.panel_box[0] + self.positions[index + 1]
        next_trim_val = (
            self.trim[index + 1] if index + 1 < len(self.trim) else self.trim[-1]
        )
        if 0 <= next_trim_val <= 1:
            next_trim_px = int(round(next_trim_val * icon_width))
        else:
            next_trim_px = int(round(next_trim_val))
        next_icon_left_trimmed = next_icon_left - next_trim_px

        self.assertGreaterEqual(
            left,
            icon_right_trimmed + self.pad_left,
            f"{name} left not ≥ icon_right + padding",
        )
        self.assertLessEqual(
            right,
            next_icon_left_trimmed - self.pad_right,
            f"{name} right not ≤ next_icon_left - padding",
        )

        if name == "population_limit":
            idle_left = regions["idle_villager"][0]
            self.assertLess(right, idle_left, "population right not < idle villager left")
            self.assertLessEqual(
                right,
                idle_left - self.pad_right,
                "population right not ≤ idle villager left - padding",
            )

    def test_wood_stockpile_roi_bounds(self):
        regions, icon_width = self._locate_regions()
        self._assert_bounds(regions, icon_width, 0)

    def test_food_stockpile_roi_bounds(self):
        regions, icon_width = self._locate_regions()
        self._assert_bounds(regions, icon_width, 1)

    def test_gold_stockpile_roi_bounds(self):
        regions, icon_width = self._locate_regions()
        self._assert_bounds(regions, icon_width, 2)

    def test_stone_stockpile_roi_bounds(self):
        regions, icon_width = self._locate_regions()
        self._assert_bounds(regions, icon_width, 3)

    def test_population_limit_roi_bounds(self):
        regions, icon_width = self._locate_regions()
        self._assert_bounds(regions, icon_width, 4)

    def test_synthetic_frame_rois_between_icons(self):
        self._build_synthetic_frame()
        regions, icon_width = self._locate_regions()
        for idx in range(5):
            name = self.icons[idx]
            self._assert_bounds(regions, icon_width, idx)
            x, y, w, h = regions[name]
            self.assertGreater(w, 0, f"{name} ROI width not positive")
            roi = self.frame[y : y + h, x : x + w]
            self.assertFalse(
                np.any(roi == self.icon_color),
                f"{name} ROI overlaps icon",
            )
            self.assertTrue(
                np.all(roi == self.digit_values[name]),
                f"{name} ROI missing digits",
            )

    def test_rois_trimmed_icon_bounds(self):
        trim = [0.2] * 6
        regions, icon_width = self._locate_regions(icon_trim_pct=trim)
        for idx in range(5):
            self._assert_bounds(regions, icon_width, idx)

    def test_rois_trimmed_icon_bounds_pixels(self):
        trim = [2] * 6
        regions, icon_width = self._locate_regions(icon_trim_pct=trim)
        for idx in range(5):
            self._assert_bounds(regions, icon_width, idx)

    def test_rois_use_available_width_with_close_icons(self):
        frame = np.zeros((50, 100, 3), dtype=np.uint8)
        panel_box = (0, 0, 100, 20)

        icons = ["wood_stockpile", "food_stockpile"]
        positions = [0, 20]
        pad_left = 2
        pad_right = 2
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
                "min_width": 0,
                "top_pct": 0.0,
                "height_pct": 1.0,
            }), patch.dict(
                common.CFG["profiles"]["aoe1de"]["resource_panel"],
                {"icon_trim_pct": [0] * 6},
            ):
                regions = resources.locate_resource_panel(frame)
                icon_width = screen_utils.ICON_TEMPLATES[icons[0]].shape[1]

        roi = regions[icons[0]]
        left, _, width, _ = roi
        right = left + width
        icon_right = positions[0] + icon_width
        next_icon_left = positions[1]
        available_left = icon_right + pad_left
        available_right = next_icon_left - pad_right
        expected_width = available_right - available_left
        self.assertEqual(width, expected_width)
        self.assertGreaterEqual(left, available_left)
        self.assertLessEqual(right, available_right)

    def test_rois_clamp_with_min_width(self):
        frame = np.zeros((50, 100, 3), dtype=np.uint8)
        panel_box = (0, 0, 100, 20)

        icons = ["wood_stockpile", "food_stockpile"]
        positions = [0, 20]
        pad_left = 2
        pad_right = 2
        min_width = 20
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
                "min_width": min_width,
                "top_pct": 0.0,
                "height_pct": 1.0,
            }), patch.dict(
                common.CFG["profiles"]["aoe1de"]["resource_panel"],
                {"icon_trim_pct": [0] * 6},
            ):
            regions = resources.locate_resource_panel(frame)
            icon_width = screen_utils.ICON_TEMPLATES[icons[0]].shape[1]

        roi = regions[icons[0]]
        left, _, width, _ = roi
        right = left + width
        icon_right = positions[0] + icon_width
        next_icon_left = positions[1]

        self.assertGreaterEqual(left, icon_right + pad_left)
        self.assertLessEqual(right, next_icon_left - pad_right)

    def test_roi_limited_by_max_width(self):
        frame = np.zeros((50, 200, 3), dtype=np.uint8)
        panel_box = (0, 0, 200, 20)

        icons = ["wood_stockpile", "food_stockpile"]
        positions = [0, 120]
        pad_left = 2
        pad_right = 2
        max_widths = [50, 30, 999, 999, 999, 999]
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
                "max_width": max_widths,
                "min_width": 0,
                "top_pct": 0.0,
                "height_pct": 1.0,
            }), patch.dict(
                common.CFG["profiles"]["aoe1de"]["resource_panel"],
                {"icon_trim_pct": [0] * 6},
            ):
                regions = resources.locate_resource_panel(frame)
                icon_width = screen_utils.ICON_TEMPLATES[icons[0]].shape[1]

        roi1 = regions[icons[0]]
        left1, _, width1, _ = roi1
        right1 = left1 + width1
        icon_right1 = positions[0] + icon_width
        next_icon_left1 = positions[1]
        avail_left1 = icon_right1 + pad_left
        avail_right1 = next_icon_left1 - pad_right
        avail_width1 = avail_right1 - avail_left1
        expected_width1 = min(avail_width1, max_widths[0])
        expected_left1 = avail_left1
        expected_right1 = expected_left1 + expected_width1
        if expected_right1 > avail_right1:
            expected_left1 = avail_right1 - expected_width1
            expected_right1 = avail_right1

        self.assertEqual(width1, expected_width1, "width not limited by max_width[0]")
        self.assertEqual(left1, expected_left1, "ROI not anchored within bounds")
        self.assertEqual(right1, expected_right1, "ROI exceeds available space")

        roi2 = regions[icons[1]]
        left2, _, width2, _ = roi2
        icon_right2 = positions[1] + icon_width
        avail_left2 = icon_right2 + pad_left
        avail_right2 = panel_box[2] - pad_right
        avail_width2 = avail_right2 - avail_left2
        expected_width2 = min(avail_width2, max_widths[1])

        self.assertEqual(width2, expected_width2, "width not limited by max_width[1]")
        self.assertEqual(left2, avail_left2)


    def test_cache_cleared_on_region_change(self):
        resources.RESOURCE_CACHE.last_icon_bounds.clear()
        resources._LAST_REGION_BOUNDS = None
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()

        self.positions = [0, 30, 60, 90, 120, 150]
        self._locate_regions()

        resources.RESOURCE_CACHE.last_resource_values["wood_stockpile"] = 111
        resources.RESOURCE_CACHE.last_resource_ts["wood_stockpile"] = 1.0
        self.assertTrue(resources.RESOURCE_CACHE.last_resource_values)
        self.assertTrue(resources.RESOURCE_CACHE.last_resource_ts)

        self.positions = [5, 35, 65, 95, 125, 155]
        self._locate_regions()

        self.assertEqual(resources.RESOURCE_CACHE.last_resource_values, {})
        self.assertEqual(resources.RESOURCE_CACHE.last_resource_ts, {})

    def test_detect_regions_scales_with_resolution(self):
        """Auto-calibrated ROIs should scale with the screen resolution."""

        def _run(scale):
            width = 200 * scale
            height = 20 * scale
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            positions = [p * scale for p in self.positions]
            loc_iter = iter([(x, 0) for x in positions])

            def fake_minmax(res):
                xi, yi = next(loc_iter)
                return 0.0, 0.95, (0, 0), (xi, yi)

            icon_size = 5 * scale

            with patch("script.resources.locate_resource_panel", return_value={}), \
                patch("script.resources.input_utils._screen_size", return_value=(width, height)), \
                patch("script.resources.cv2.cvtColor", lambda src, code: np.zeros(src.shape[:2], dtype=np.uint8)), \
                patch("script.resources.cv2.resize", lambda img, *a, **k: img), \
                patch("script.resources.cv2.matchTemplate", lambda *a, **k: np.zeros((100, 200), dtype=np.float32)), \
                patch("script.resources.cv2.minMaxLoc", side_effect=fake_minmax), \
                patch.object(screen_utils, "_load_icon_templates", lambda: None), \
                patch.dict(
                    screen_utils.ICON_TEMPLATES,
                    {name: np.zeros((icon_size, icon_size), dtype=np.uint8) for name in self.icons},
                    clear=True,
                ), patch.dict(
                    common.CFG["resource_panel"],
                    {
                        "roi_padding_left": [self.pad_left] * 6,
                        "roi_padding_right": [self.pad_right] * 6,
                        "icon_trim_pct": [0] * 6,
                        "scales": [1.0],
                        "match_threshold": 0.5,
                        "max_width": 999,
                        "min_width": 0,
                        "top_pct": 0.0,
                        "height_pct": 1.0,
                    },
                ), patch.dict(
                    common.CFG["profiles"]["aoe1de"]["resource_panel"],
                    {"icon_trim_pct": [0] * 6},
                ), patch.object(common, "HUD_ANCHOR", None):
                return resources.detect_resource_regions(frame, self.icons)

        regions1 = _run(1)
        regions2 = _run(2)

        for name in self.icons[:-1]:
            roi1 = regions1[name]
            roi2 = regions2[name]
            self.assertAlmostEqual(roi1[0] * 2, roi2[0], delta=1)
            self.assertAlmostEqual(roi1[2] * 2, roi2[2], delta=1)
