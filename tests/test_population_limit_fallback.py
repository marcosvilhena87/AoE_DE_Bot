from unittest.mock import patch

import os
import sys
import types
import numpy as np

sys.modules.setdefault(
    "cv2",
    types.SimpleNamespace(
        cvtColor=lambda src, code: src,
        resize=lambda img, *a, **k: img,
        matchTemplate=lambda *a, **k: np.zeros((1, 1), dtype=np.float32),
        minMaxLoc=lambda res: (0.0, 0.0, (0, 0), (0, 0)),
        imread=lambda *a, **k: np.zeros((1, 1), dtype=np.uint8),
        NORM_MINMAX=0,
        TM_CCOEFF_NORMED=0,
        IMREAD_GRAYSCALE=0,
        COLOR_BGR2GRAY=0,
    ),
)

class DummyMSS:
    monitors = [{}, {"left": 0, "top": 0, "width": 200, "height": 200}]

    def grab(self, region):
        h, w = region["height"], region["width"]
        return np.zeros((h, w, 4), dtype=np.uint8)

sys.modules.setdefault("mss", types.SimpleNamespace(mss=lambda: DummyMSS()))

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")
sys.modules.setdefault("pytesseract", types.SimpleNamespace(pytesseract=types.SimpleNamespace(tesseract_cmd="")))

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import script.resources as resources
from script.resources.panel import detection
from script.resources.panel import ResourcePanelCfg


def _make_cfg():
    return ResourcePanelCfg(
        match_threshold=0.5,
        scales=[1.0],
        pad_left=[0] * len(resources.RESOURCE_ICON_ORDER),
        pad_right=[0] * len(resources.RESOURCE_ICON_ORDER),
        icon_trims=[0] * len(resources.RESOURCE_ICON_ORDER),
        max_widths=[999] * len(resources.RESOURCE_ICON_ORDER),
        min_widths=[0] * len(resources.RESOURCE_ICON_ORDER),
        min_requireds=[0] * len(resources.RESOURCE_ICON_ORDER),
        top_pct=0.0,
        height_pct=1.0,
        idle_roi_extra_width=0,
        min_pop_width=30,
        pop_roi_extra_width=5,
    )


def test_population_limit_fallback_estimates_width_with_padding():
    frame = np.zeros((20, 200, 3), dtype=np.uint8)
    cache = resources.cache.ResourceCache()
    cache.last_icon_bounds["population_limit"] = (0, 0, 1000, 10)
    xi = 100
    yi = 0
    wi = 10
    hi = 10

    cfg = _make_cfg()

    with patch.object(detection, "detect_hud", return_value=((0, 0, 200, 20), 1.0)), \
        patch.object(resources.screen_utils, "_load_icon_templates", lambda: None), \
        patch.dict(resources.screen_utils.ICON_TEMPLATES, {"idle_villager": np.zeros((hi, wi), dtype=np.uint8)}, clear=True), \
        patch.object(resources.cv2, "cvtColor", lambda src, code: np.zeros(src.shape[:2], dtype=np.uint8)), \
        patch.object(resources.cv2, "resize", lambda img, *a, **k: img), \
        patch.object(resources.cv2, "matchTemplate", lambda *a, **k: np.zeros((1, 1), dtype=np.float32)), \
        patch.object(resources.cv2, "minMaxLoc", lambda res: (0.0, 0.95, (0, 0), (xi, yi))), \
        patch.object(detection, "_get_resource_panel_cfg", return_value=cfg):
        detection.locate_resource_panel(frame, cache)
        detection.locate_resource_panel(frame, cache)

    pl = cache.last_icon_bounds["population_limit"]
    iv = cache.last_icon_bounds["idle_villager"]
    base = max(2 * wi, cfg.min_pop_width)
    assert pl[2] == base + cfg.pop_roi_extra_width
    assert pl[0] == max(0, iv[0] - base)
    assert pl[0] + pl[2] == iv[0] + cfg.pop_roi_extra_width
