from unittest import TestCase
import sys
import types
import numpy as np
import os

cv2_stub = types.SimpleNamespace(
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
)
sys.modules.setdefault("cv2", cv2_stub)
sys.modules.setdefault("pytesseract", types.SimpleNamespace())
sys.modules.setdefault("pyautogui", types.SimpleNamespace())
sys.modules.setdefault("mss", types.SimpleNamespace(mss=lambda: None))
sys.modules.setdefault(
    "script.screen_utils",
    types.SimpleNamespace(ICON_TEMPLATES={}, HUD_TEMPLATE=None, _load_icon_templates=lambda: None),
)
sys.modules.setdefault(
    "script.common",
    types.SimpleNamespace(CFG={"resource_panel": {}}, HUD_ANCHOR={"left": 0, "width": 0}),
)
sys.modules.setdefault(
    "script.input_utils",
    types.SimpleNamespace(_screen_size=lambda: (0, 0)),
)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import script.resources as resources


class TestResourceMinRequiredWidth(TestCase):
    def test_forces_min_width_for_wide_values(self):
        """ROIs should expand to fit values with 3-4 digits."""
        detected = {
            "wood_stockpile": (0, 0, 5, 5),
            "food_stockpile": (100, 0, 5, 5),
        }
        regions, _spans, _narrow = resources.compute_resource_rois(
            0,
            200,
            0,
            10,
            [0] * 6,
            [0] * 6,
            [0] * 6,
            20,
            [0] * 6,
            [50] * 6,
            detected,
        )
        self.assertEqual(regions["wood_stockpile"][2], 50)
