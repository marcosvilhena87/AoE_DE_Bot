import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np

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
import script.resources as resources


class TestGetResourcePanelCfg(TestCase):
    def test_basic_processing(self):
        cfg_data = {
            "resource_panel": {
                "roi_padding_left": 3,
                "roi_padding_right": 4,
                "icon_trim_pct": 0.2,
                "max_width": 150,
                "min_width": 80,
                "min_required_width": 20,
                "top_pct": 0.1,
                "height_pct": 0.9,
                "idle_roi_extra_width": 5,
                "match_threshold": 0.95,
            },
            "scales": [1.0, 1.5],
            "profile": None,
        }
        with patch.dict(resources.CFG, cfg_data, clear=False):
            cfg = resources._get_resource_panel_cfg()
        n = len(resources.RESOURCE_ICON_ORDER)
        self.assertEqual(cfg.match_threshold, 0.95)
        self.assertEqual(cfg.scales, [1.0, 1.5])
        self.assertEqual(cfg.pad_left, [3] * n)
        self.assertEqual(cfg.pad_right, [4] * n)
        self.assertEqual(cfg.icon_trims, [0.2] * n)
        self.assertEqual(cfg.max_width, 150)
        self.assertEqual(cfg.min_widths, [80] * n)
        self.assertEqual(cfg.min_requireds, [20] * n)
        self.assertEqual(cfg.top_pct, 0.1)
        self.assertEqual(cfg.height_pct, 0.9)
        self.assertEqual(cfg.idle_roi_extra_width, 5)
