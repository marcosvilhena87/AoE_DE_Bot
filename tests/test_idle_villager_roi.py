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
    def test_idle_villager_roi_uses_icon_bounds(self):
        detected = {
            "idle_villager": (10, 0, 20, 10),
        }
        regions, spans, _narrow = resources.compute_resource_rois(
            0,
            100,
            0,
            10,
            [0] * 6,
            [0] * 6,
            [0] * 6,
            [999] * 6,
            [0] * 6,
            0,
            0,
            detected=detected,
        )
        self.assertIn("idle_villager", regions)
        self.assertEqual(regions["idle_villager"], (10, 0, 20, 10))
        self.assertEqual(spans["idle_villager"], (10, 30))

    def test_idle_villager_roi_respects_inner_trim_config(self):
        detected = {
            "idle_villager": (10, 0, 20, 10),
        }
        with patch.dict(resources.CFG, {"idle_icon_inner_trim": 3}, clear=False):
            regions, spans, _ = resources.compute_resource_rois(
                0,
                100,
                0,
                10,
                [0] * 6,
                [0] * 6,
                [0] * 6,
                [999] * 6,
                [0] * 6,
                0,
                0,
                detected=detected,
            )
        self.assertEqual(regions["idle_villager"], (13, 0, 14, 10))
        self.assertEqual(spans["idle_villager"], (13, 27))

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

