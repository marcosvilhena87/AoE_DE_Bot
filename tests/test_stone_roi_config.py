import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np

# Stub GUI-dependent modules

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


class TestStoneROIConfig(TestCase):
    def _detect(self, size):
        w, h = size
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        with patch("script.resources.locate_resource_panel", return_value={}), \
             patch("script.resources.input_utils._screen_size", return_value=(w, h)), \
             patch.object(common, "HUD_ANCHOR", None):
            regions = resources.detect_resource_regions(frame, ["stone_stockpile"])
        return regions["stone_stockpile"]

    def test_stone_roi_scales_1080p(self):
        roi = self._detect((1920, 1080))
        cfg = resources.CFG["stone_stockpile_roi"]
        expected = (
            int(cfg["left_pct"] * 1920),
            int(cfg["top_pct"] * 1080),
            int(cfg["width_pct"] * 1920),
            int(cfg["height_pct"] * 1080),
        )
        self.assertEqual(roi, expected)

    def test_stone_roi_scales_1440p(self):
        roi = self._detect((2560, 1440))
        cfg = resources.CFG["stone_stockpile_roi"]
        expected = (
            int(cfg["left_pct"] * 2560),
            int(cfg["top_pct"] * 1440),
            int(cfg["width_pct"] * 2560),
            int(cfg["height_pct"] * 1440),
        )
        self.assertEqual(roi, expected)
