import types
import sys
import numpy as np
import pytest
import os

# Stub external dependencies used during import of script.resources
try:  # pragma: no cover
    import cv2  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
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
        bilateralFilter=lambda src, d, sigmaColor, sigmaSpace: src,
        adaptiveThreshold=lambda src, maxValue, adaptiveMethod, thresholdType, blockSize, C: src,
        dilate=lambda src, kernel, iterations=1: src,
        equalizeHist=lambda src: src,
        countNonZero=lambda src: int(np.count_nonzero(src)),
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
    types.SimpleNamespace(ICON_TEMPLATES={}, HUD_TEMPLATE=None, load_icon_templates=lambda: None),
)
sys.modules.setdefault(
    "script.common",
    types.SimpleNamespace(
        STATE=types.SimpleNamespace(
            config={"resource_panel": {}}, current_pop=0, pop_cap=0, target_pop=0
        ),
        HUD_ANCHOR={"left": 0, "width": 0},
    ),
)
sys.modules.setdefault("script.input_utils", types.SimpleNamespace())
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import script.resources as resources

BASE_PARAMS = {
    "pad_left": [1],
    "pad_right": [1],
    "icon_trims": [0],
    "max_widths": [10],
    "min_widths": [1],
    "min_pop_width": 0,
    "idle_extra_width": 0,
}
ORDER = ["pad_left", "pad_right", "icon_trims", "max_widths", "min_widths"]


@pytest.mark.parametrize("missing", ORDER)
def test_empty_config_lists_do_not_raise(missing):
    params = [BASE_PARAMS[key] for key in ORDER] + [BASE_PARAMS["min_pop_width"], BASE_PARAMS["idle_extra_width"]]
    idx = ORDER.index(missing)
    params[idx] = []
    resources.compute_resource_rois(0, 100, 0, 10, *params)
