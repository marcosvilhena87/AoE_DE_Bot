import os
import sys
import types
from unittest import TestCase

import numpy as np

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
try:  # pragma: no cover - used for environments without OpenCV
    import cv2  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - fallback stub
    sys.modules.setdefault(
        "cv2",
        types.SimpleNamespace(
            cvtColor=lambda src, code: src,
            resize=lambda img, *a, **k: img,
            matchTemplate=lambda *a, **k: np.zeros((1, 1), dtype=np.float32),
            minMaxLoc=lambda *a, **k: (0, 0, (0, 0), (0, 0)),
            imread=lambda *a, **k: np.zeros((1, 1), dtype=np.uint8),
            imwrite=lambda *a, **k: True,
            medianBlur=lambda src, k: src,
            bitwise_not=lambda src: src,
            threshold=lambda src, *a, **k: (None, src),
            rectangle=lambda img, pt1, pt2, color, thickness: img,
            INTER_LINEAR=0,
            THRESH_BINARY=0,
            THRESH_OTSU=0,
            TM_CCOEFF_NORMED=0,
            IMREAD_GRAYSCALE=0,
            COLOR_BGR2GRAY=0,
        ),
    )

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.resources as resources


class TestFallbackROIsFromSlice(TestCase):
    def setUp(self):
        resources.cache._NARROW_ROIS.clear()
        resources.cache._NARROW_ROI_DEFICITS.clear()
        resources.cache._LAST_REGION_SPANS.clear()

    def tearDown(self):
        resources.cache._NARROW_ROIS.clear()
        resources.cache._NARROW_ROI_DEFICITS.clear()
        resources.cache._LAST_REGION_SPANS.clear()

    def test_fallback_rois_from_slice_updates_cache_and_regions(self):
        regions = resources._fallback_rois_from_slice(
            0,  # left
            120,  # width
            0,  # top
            20,  # height
            [0],  # icon_trims
            0,  # right_trim
            set(),  # required_icons
        )

        expected_icons = set(resources.RESOURCE_ICON_ORDER)
        self.assertEqual(set(regions.keys()), expected_icons)

        for name in resources.RESOURCE_ICON_ORDER:
            l, t, w, h = regions[name]
            self.assertEqual((t, h), (0, 20))
            self.assertEqual(resources.cache._LAST_REGION_SPANS[name], (l, l + w))

        self.assertEqual(resources.cache._NARROW_ROIS, set())

        # Ensure ROI spans are reported in non-decreasing order
        spans = resources.cache._LAST_REGION_SPANS
        prev_left = -1
        for name in resources.RESOURCE_ICON_ORDER:
            self.assertGreaterEqual(spans[name][0], prev_left)
            prev_left = spans[name][0]
