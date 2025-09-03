import os
import sys
import types
from unittest.mock import patch
import numpy as np

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
        patch.object(resources.cv2, "matchTemplate", lambda *a, **k: np.zeros((1, 1), dtype=np.float32), create=True), \
        patch.object(resources.cv2, "minMaxLoc", lambda res: (0.0, 0.95, (0, 0), (xi, yi)), create=True), \
        patch.object(resources.cv2, "TM_CCOEFF_NORMED", 0, create=True), \
        patch.object(detection, "_get_resource_panel_cfg", return_value=cfg):
        detection.locate_resource_panel(frame, cache)
        detection.locate_resource_panel(frame, cache)

    pl = cache.last_icon_bounds["population_limit"]
    iv = cache.last_icon_bounds["idle_villager"]
    base = max(2 * wi, cfg.min_pop_width)
    assert pl[2] == base + cfg.pop_roi_extra_width
    assert pl[0] == max(0, iv[0] - base)
    assert pl[0] + pl[2] == iv[0] + cfg.pop_roi_extra_width


def test_population_low_conf_cache_fallback():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    regions = {"population_limit": (0, 0, 5, 5)}
    results = {}
    cache = resources.cache.ResourceCache()
    cache.last_resource_values["population_limit"] = (80, 200)
    cache.resource_failure_counts["population_limit"] = 2
    cache.resource_low_conf_counts = {"population_limit": 2}

    def low_conf_error(roi, conf_threshold=None, roi_bbox=None, failure_count=0):
        err = resources.common.PopulationReadError("low")
        err.low_conf = True
        err.low_conf_digits = None
        raise err

    with patch.dict(
        resources.common.CFG,
        {"population_limit_low_conf_fallback": True, "ocr_retry_limit": 3},
        clear=False,
    ), patch(
        "script.resources.ocr.executor._read_population_from_roi",
        side_effect=low_conf_error,
    ), patch(
        "script.resources.reader.roi.expand_population_roi_after_failure",
        return_value=None,
    ):
        cur, cap = resources._extract_population(
            frame,
            regions,
            results,
            True,
            conf_threshold=60,
            cache_obj=cache,
        )

    assert (cur, cap) == (80, 200)
    assert results["population_limit"] == 80


def test_population_roi_expansion_respects_idle_villager_boundary():
    frame = np.zeros((20, 200, 3), dtype=np.uint8)
    regions = {
        "population_limit": (50, 0, 20, 10),
        "idle_villager": (90, 0, 10, 10),
    }
    results = {}
    cache = resources.cache.ResourceCache()

    side_effects = [
        resources.common.PopulationReadError("fail"),
        (80, 200, False),
    ]

    def fake_read(roi, conf_threshold=None, roi_bbox=None, failure_count=0):
        res = side_effects.pop(0)
        if isinstance(res, Exception):
            raise res
        return res

    expansion = {}
    orig_expand = resources.reader.roi.expand_population_roi_after_failure

    def wrapper(frame, x, y, w, h, r, failure_count, res_conf_threshold, max_right=None):
        res = orig_expand(frame, x, y, w, h, r, failure_count, res_conf_threshold, max_right=max_right)
        expansion["res"] = res
        return res

    mock_read_fn = patch("script.resources.ocr.executor._read_population_from_roi", side_effect=fake_read)
    with patch.dict(
        resources.common.CFG,
        {"population_ocr_roi_expand_base": 50, "population_idle_padding": 6},
        clear=False,
    ), mock_read_fn as mock_exec, patch(
        "script.resources.reader.roi._read_population_from_roi",
        mock_exec,
    ), patch(
        "script.resources.reader.roi.expand_population_roi_after_failure",
        new=wrapper,
    ):
        resources._extract_population(
            frame,
            regions,
            results,
            True,
            cache_obj=cache,
        )

    assert expansion["res"] is not None
    x0 = expansion["res"][3]
    y0 = expansion["res"][4]
    width = expansion["res"][5]
    height = expansion["res"][6]
    assert x0 + width <= regions["idle_villager"][0] - 6
    assert y0 == regions["population_limit"][1]
    assert height == regions["population_limit"][3]
