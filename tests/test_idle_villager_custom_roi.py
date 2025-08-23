import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np

# Stub modules requiring a GUI/display

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


class TestIdleVillagerCustomROI(TestCase):
    def test_roi_matches_configured_percentages(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        cfg = {
            "left_pct": 0.2,
            "top_pct": 0.3,
            "width_pct": 0.25,
            "height_pct": 0.25,
        }
        expected = (40, 60, 50, 50)
        with patch("script.resources.locate_resource_panel", return_value={}), \
            patch("script.resources.input_utils._screen_size", return_value=(200, 200)), \
            patch.dict(resources.CFG, {"idle_villager_roi": cfg}, clear=False), \
            patch.object(common, "HUD_ANCHOR", None):
            regions = resources.detect_resource_regions(frame, ["idle_villager"])

        self.assertEqual(regions["idle_villager"], expected)
