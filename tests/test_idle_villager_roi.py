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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.common as common


class TestIdleVillagerROI(TestCase):
    def test_idle_villager_roi_matches_icon_bounds(self):
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

        with patch("script.common.find_template", return_value=(panel_box, 0.9, None)), \
            patch("script.common.cv2.cvtColor", side_effect=fake_cvtColor), \
            patch("script.common.cv2.resize", side_effect=lambda img, *a, **k: img), \
            patch("script.common.cv2.matchTemplate", side_effect=fake_match), \
            patch("script.common.cv2.minMaxLoc", side_effect=fake_minmax), \
            patch("script.common.cv2.imread", side_effect=fake_imread), \
            patch.dict(
                common.CFG["resource_panel"],
                {
                    "roi_padding_left": 0,
                    "roi_padding_right": 0,
                    "scales": [1.0],
                    "match_threshold": 0.5,
                },
            ):
            regions = common.locate_resource_panel(frame)

        self.assertIn("idle_villager", regions)
        roi = regions["idle_villager"]
        expected = (panel_box[0] + xi, panel_box[1] + yi, icon_w, icon_h)
        self.assertEqual(roi, expected)
        self.assertGreater(roi[2], 0)
        self.assertGreater(roi[3], 0)
