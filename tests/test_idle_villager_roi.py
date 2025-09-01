import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np

import cv2
cv2.imread = lambda *a, **k: np.zeros((1, 1), dtype=np.uint8)

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

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.common as common
import script.resources as resources
import script.screen_utils as screen_utils


class TestIdleVillagerROI(TestCase):
    def test_idle_villager_roi_uses_digit_span(self):
        frame = np.zeros((50, 100, 3), dtype=np.uint8)
        panel_box = (10, 15, 80, 20)  # x, y, w, h
        xi, yi = 5, 4
        icon_h, icon_w = 5, 5

        def fake_imread(path, flags=0):
            name = os.path.splitext(os.path.basename(path))[0]
            if name == "idle_villager":
                return np.ones((icon_h, icon_w), dtype=np.uint8)
            return np.zeros((icon_h, icon_w), dtype=np.uint8)

        def fake_match(img, templ, method):
            h = img.shape[0] - templ.shape[0] + 1
            w = img.shape[1] - templ.shape[1] + 1
            res = np.zeros((h, w), dtype=np.float32)
            if np.all(templ == 1):
                res[yi, xi] = 0.95
            return res

        def fake_minmax(res):
            max_val = float(res.max())
            max_loc = tuple(np.unravel_index(res.argmax(), res.shape)[::-1])
            return 0.0, max_val, (0, 0), max_loc

        def fake_cvtColor(src, code):
            return np.zeros(src.shape[:2], dtype=np.uint8)

        def fake_compute(pl, pr, top, height, *args, **kwargs):
            left = pl + xi + icon_w
            span = (left, left + 10)
            return {}, {"idle_villager": span}, {}

        with patch("script.resources.find_template", return_value=(panel_box, 0.9, None)), \
            patch("script.resources.cv2.cvtColor", side_effect=fake_cvtColor), \
            patch("script.resources.cv2.resize", side_effect=lambda img, *a, **k: img), \
            patch("script.resources.cv2.matchTemplate", side_effect=fake_match), \
            patch("script.resources.cv2.minMaxLoc", side_effect=fake_minmax), \
            patch("script.resources.cv2.imread", side_effect=fake_imread), \
            patch("script.resources.panel.detection.compute_resource_rois", side_effect=fake_compute), \
            patch.dict(screen_utils.ICON_TEMPLATES, {}, clear=True), \
            patch.dict(
                common.CFG["resource_panel"],
                {
                    "roi_padding_left": [0] * 6,
                    "roi_padding_right": [0] * 6,
                    "icon_trim_pct": [0] * 6,
                    "scales": [1.0],
                    "match_threshold": 0.5,
                    "max_width": 999,
                    "min_width": 0,
                },
            ), patch.dict(
                common.CFG["profiles"]["aoe1de"]["resource_panel"],
                {"icon_trim_pct": [0] * 6},
            ):
                regions = resources.locate_resource_panel(frame)

        self.assertIn("idle_villager", regions)
        roi = regions["idle_villager"]
        cfg_obj = resources.panel._get_resource_panel_cfg()
        top = panel_box[1] + int(cfg_obj.top_pct * panel_box[3])
        height = int(cfg_obj.height_pct * panel_box[3])
        expected = (panel_box[0] + xi + icon_w, top, 10, height)
        self.assertEqual(roi, expected)
        self.assertEqual(roi[2], 10)
        self.assertGreater(roi[3], 0)

    def test_idle_villager_roi_clamped_before_population(self):
        frame = np.zeros((50, 100, 3), dtype=np.uint8)
        panel_box = (10, 15, 80, 20)
        xi, yi = 5, 4
        icon_h, icon_w = 5, 5

        def fake_imread(path, flags=0):
            name = os.path.splitext(os.path.basename(path))[0]
            if name == "idle_villager":
                return np.ones((icon_h, icon_w), dtype=np.uint8)
            return np.zeros((icon_h, icon_w), dtype=np.uint8)

        def fake_match(img, templ, method):
            h = img.shape[0] - templ.shape[0] + 1
            w = img.shape[1] - templ.shape[1] + 1
            res = np.zeros((h, w), dtype=np.float32)
            if np.all(templ == 1):
                res[yi, xi] = 0.95
            return res

        def fake_minmax(res):
            max_val = float(res.max())
            max_loc = tuple(np.unravel_index(res.argmax(), res.shape)[::-1])
            return 0.0, max_val, (0, 0), max_loc

        def fake_cvtColor(src, code):
            return np.zeros(src.shape[:2], dtype=np.uint8)

        def fake_compute(pl, pr, top, height, *args, **kwargs):
            offset = 20
            pop_span = (pl + offset, pl + offset + 20)
            idle_left = pl + xi + icon_w
            idle_span = (idle_left, pop_span[0] + 10)
            regions = {"population_limit": (pl + offset, top, 10, height)}
            spans = {"population_limit": pop_span, "idle_villager": idle_span}
            return regions, spans, {}

        with patch("script.resources.find_template", return_value=(panel_box, 0.9, None)), \
            patch("script.resources.cv2.cvtColor", side_effect=fake_cvtColor), \
            patch("script.resources.cv2.resize", side_effect=lambda img, *a, **k: img), \
            patch("script.resources.cv2.matchTemplate", side_effect=fake_match), \
            patch("script.resources.cv2.minMaxLoc", side_effect=fake_minmax), \
            patch("script.resources.cv2.imread", side_effect=fake_imread), \
            patch("script.resources.panel.detection.compute_resource_rois", side_effect=fake_compute), \
            patch.dict(screen_utils.ICON_TEMPLATES, {}, clear=True), \
            patch.dict(
                common.CFG["resource_panel"],
                {
                    "roi_padding_left": [0] * 6,
                    "roi_padding_right": [0] * 6,
                    "icon_trim_pct": [0] * 6,
                    "scales": [1.0],
                    "match_threshold": 0.5,
                    "max_width": 999,
                    "min_width": 0,
                },
            ), patch.dict(
                common.CFG["profiles"]["aoe1de"]["resource_panel"],
                {"icon_trim_pct": [0] * 6},
            ):
                regions = resources.locate_resource_panel(frame)

        self.assertIn("idle_villager", regions)
        roi = regions["idle_villager"]
        pop_left = panel_box[0] + 20
        self.assertLessEqual(roi[0] + roi[2], pop_left)

    def test_idle_villager_roi_shrinks_near_population(self):
        frame = np.zeros((50, 100, 3), dtype=np.uint8)
        panel_box = (10, 15, 80, 20)
        xi, yi = 5, 4
        icon_h, icon_w = 5, 5

        def fake_imread(path, flags=0):
            name = os.path.splitext(os.path.basename(path))[0]
            if name == "idle_villager":
                return np.ones((icon_h, icon_w), dtype=np.uint8)
            return np.zeros((icon_h, icon_w), dtype=np.uint8)

        def fake_match(img, templ, method):
            h = img.shape[0] - templ.shape[0] + 1
            w = img.shape[1] - templ.shape[1] + 1
            res = np.zeros((h, w), dtype=np.float32)
            if np.all(templ == 1):
                res[yi, xi] = 0.95
            return res

        def fake_minmax(res):
            max_val = float(res.max())
            max_loc = tuple(np.unravel_index(res.argmax(), res.shape)[::-1])
            return 0.0, max_val, (0, 0), max_loc

        def fake_cvtColor(src, code):
            return np.zeros(src.shape[:2], dtype=np.uint8)

        def fake_compute(pl, pr, top, height, *args, **kwargs):
            offset = 25
            pop_span = (pl + offset, pl + offset + 20)
            idle_left = pl + xi + icon_w
            idle_span = (idle_left, pop_span[0] + 10)
            regions = {"population_limit": (pl + offset, top, 10, height)}
            spans = {"population_limit": pop_span, "idle_villager": idle_span}
            return regions, spans, {}

        with patch("script.resources.find_template", return_value=(panel_box, 0.9, None)), \
            patch("script.resources.cv2.cvtColor", side_effect=fake_cvtColor), \
            patch("script.resources.cv2.resize", side_effect=lambda img, *a, **k: img), \
            patch("script.resources.cv2.matchTemplate", side_effect=fake_match), \
            patch("script.resources.cv2.minMaxLoc", side_effect=fake_minmax), \
            patch("script.resources.cv2.imread", side_effect=fake_imread), \
            patch("script.resources.panel.detection.compute_resource_rois", side_effect=fake_compute), \
            patch.dict(screen_utils.ICON_TEMPLATES, {}, clear=True), \
            patch.dict(
                common.CFG["resource_panel"],
                {
                    "roi_padding_left": [0] * 6,
                    "roi_padding_right": [0] * 6,
                    "icon_trim_pct": [0] * 6,
                    "scales": [1.0],
                    "match_threshold": 0.5,
                    "max_width": 999,
                    "min_width": 0,
                },
            ), patch.dict(
                common.CFG["profiles"]["aoe1de"]["resource_panel"],
                {"icon_trim_pct": [0] * 6},
            ):
                regions = resources.locate_resource_panel(frame)

        self.assertIn("idle_villager", regions)
        roi = regions["idle_villager"]
        pop_left = panel_box[0] + 25
        self.assertLessEqual(roi[0] + roi[2], pop_left)
        expected_width = pop_left - (panel_box[0] + xi + icon_w)
        self.assertEqual(roi[2], expected_width)

    def test_detect_resource_regions_uses_configured_idle_roi_when_missing(self):
        frame = np.zeros((50, 100, 3), dtype=np.uint8)
        cfg = {
            "left_pct": 0.5,
            "top_pct": 0.25,
            "width_pct": 0.04,
            "height_pct": 0.05,
        }
        expected = (100, 50, 40, 20)
        with patch("script.resources.locate_resource_panel", return_value={}), \
            patch("script.resources.input_utils._screen_size", return_value=(200, 200)), \
            patch.dict(resources.CFG, {"idle_villager_roi": cfg}, clear=False), \
            patch.object(common, "HUD_ANCHOR", None):
            regions = resources.detect_resource_regions(frame, ["idle_villager"])

        self.assertEqual(regions["idle_villager"], expected)

    def test_detect_resource_regions_prefers_detected_idle_roi_over_config(self):
        frame = np.zeros((50, 100, 3), dtype=np.uint8)
        detected = {"idle_villager": (1, 2, 3, 4)}
        cfg = {
            "left_pct": 0.5,
            "top_pct": 0.25,
            "width_pct": 0.04,
            "height_pct": 0.05,
        }
        with patch("script.resources.panel.locate_resource_panel", return_value=detected), \
            patch.dict(resources.CFG, {"idle_villager_roi": cfg}, clear=False), \
            patch.object(common, "HUD_ANCHOR", None):
            regions = resources.detect_resource_regions(frame, ["idle_villager"])

        self.assertEqual(regions["idle_villager"], detected["idle_villager"])

