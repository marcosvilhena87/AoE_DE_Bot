import os
import sys
import types
import numpy as np
from unittest import TestCase
from unittest.mock import patch

# Stub modules requiring GUI/display

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

import script.resources as resources
import script.common as common


class TestPopulationROIBounds(TestCase):
    def test_expansion_respects_idle_padding(self):
        frame = np.zeros((20, 200, 3), dtype=np.uint8)
        regions = {
            "population_limit": (50, 0, 20, 10),
            "idle_villager": (90, 0, 10, 10),
        }
        results = {}
        cache = resources.cache.ResourceCache()

        side_effects = [common.PopulationReadError("fail"), (80, 200, False)]

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

        with patch.dict(
            resources.common.CFG,
            {"population_ocr_roi_expand_base": 50, "population_idle_padding": 6},
            clear=False,
        ), patch(
            "script.resources.ocr.executor._read_population_from_roi",
            side_effect=fake_read,
        ), patch(
            "script.resources.reader.roi._read_population_from_roi",
            side_effect=fake_read,
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
        width = expansion["res"][5]
        assert x0 + width <= regions["idle_villager"][0] - 6
