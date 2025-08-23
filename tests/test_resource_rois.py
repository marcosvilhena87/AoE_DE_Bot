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

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.common as common
import script.resources as resources
import script.screen_utils as screen_utils


class TestResourceROIs(TestCase):
    def test_resource_rois_do_not_overlap(self):
        frame = np.zeros((50, 200, 3), dtype=np.uint8)
        panel_box = (0, 0, 200, 20)

        icons = [
            "wood_stockpile",
            "food_stockpile",
            "gold",
            "stone",
            "population",
        ]

        positions = [0, 30, 60, 90, 120]
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
            patch.dict(screen_utils.HUD_TEMPLATES, {"assets/resources.png": np.zeros((1, 1), dtype=np.uint8)}, clear=True), \
            patch.dict(screen_utils.ICON_TEMPLATES, {name: np.zeros((5, 5), dtype=np.uint8) for name in icons}, clear=True), \
            patch.dict(common.CFG["resource_panel"], {
                "roi_padding_left": pad_left,
                "roi_padding_right": pad_right,
                "scales": [1.0],
                "match_threshold": 0.5,
                "min_width": 0,
                "top_pct": 0.0,
                "height_pct": 1.0,
            }):
                regions = resources.locate_resource_panel(frame)
                icon_width = screen_utils.ICON_TEMPLATES[icons[0]].shape[1]

        for i, name in enumerate(icons):
            roi = regions[name]
            left = roi[0]
            right = roi[0] + roi[2]
            xi = positions[i]
            icon_right = panel_box[0] + xi + icon_width

            # Ensure ROI starts after the icon with padding
            self.assertGreaterEqual(
                left,
                icon_right + pad_left,
                f"{name} left not ≥ icon_right + padding",
            )

            if i > 0:
                prev = regions[icons[i - 1]]
                prev_right = prev[0] + prev[2]
                self.assertGreater(left, prev_right, f"{name} left not > previous right")

            if i < len(icons) - 1:
                next_left = regions[icons[i + 1]][0]
                next_icon_left = panel_box[0] + positions[i + 1]
                # Ensure ROI ends before the next icon with padding
                self.assertLessEqual(
                    right,
                    next_icon_left - pad_right,
                    f"{name} right not ≤ next_icon_left - padding",
                )
                self.assertLess(right, next_left, f"{name} right not < next left")
