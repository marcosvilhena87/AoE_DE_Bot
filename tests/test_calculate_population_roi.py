import os
import sys
import types
import numpy as np
from unittest import TestCase
from unittest.mock import patch

# Stub modules that require a GUI/display before importing bot modules

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
        threshold=lambda src, *a, **k: (None, src),
        imread=lambda *a, **k: np.zeros((1, 1), dtype=np.uint8),
        imwrite=lambda *a, **k: True,
        IMREAD_GRAYSCALE=0,
        COLOR_BGR2GRAY=0,
        INTER_LINEAR=0,
        THRESH_BINARY=0,
        THRESH_OTSU=0,
    ),
)

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

# Ensure project root is importable
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import script.hud as hud


class TestCalculatePopulationROI(TestCase):
    def test_clamps_against_idle_icon(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        regions = {
            "population_limit": (50, 0, 40, 20),
            "idle_villager": (80, 0, 10, 20),
        }
        with patch("script.resources.locate_resource_panel", return_value=regions), \
            patch.dict(hud.CFG, {"population_limit_roi": None, "population_idle_padding": 5}, clear=False):
            roi = hud.calculate_population_roi(frame)
        assert roi == {"left": 50, "top": 0, "width": 25, "height": 20}

    def test_custom_population_limit_roi_config(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        pop_cfg = {
            "left_pct": 0.1,
            "top_pct": 0.2,
            "width_pct": 0.3,
            "height_pct": 0.4,
        }
        with patch("script.resources.locate_resource_panel", return_value={}), \
            patch("script.screen_utils.get_screen_size", return_value=(200, 200)), \
            patch.dict(hud.CFG, {"population_limit_roi": pop_cfg}, clear=False):
            roi = hud.calculate_population_roi(frame)
        assert roi == {"left": 20, "top": 40, "width": 60, "height": 80}
