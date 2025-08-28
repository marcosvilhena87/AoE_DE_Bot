import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

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
            bilateralFilter=lambda src, d, sigmaColor, sigmaSpace: src,
            adaptiveThreshold=lambda src, maxValue, adaptiveMethod, thresholdType, blockSize, C: src,
            dilate=lambda src, kernel, iterations=1: src,
            equalizeHist=lambda src: src,
            countNonZero=lambda src: int(np.count_nonzero(src)),
            ADAPTIVE_THRESH_GAUSSIAN_C=0,
            ADAPTIVE_THRESH_MEAN_C=0,
            IMREAD_GRAYSCALE=0,
            COLOR_BGR2GRAY=0,
            INTER_LINEAR=0,
            THRESH_BINARY=0,
            THRESH_OTSU=0,
            TM_CCOEFF_NORMED=0,
        ),
    )

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.common as common
import script.resources as resources


class TestResourceRoiValidation(TestCase):
    def setUp(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()
        resources._LAST_REGION_SPANS.clear()

    def tearDown(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()
        resources._LAST_REGION_SPANS.clear()

    def test_zero_width_roi_raises(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch(
            "script.resources.detect_resource_regions",
            return_value={"wood_stockpile": (0, 0, 0, 10)},
        ):
            with self.assertRaises(common.ResourceReadError):
                resources._read_resources(
                    frame,
                    ["wood_stockpile"],
                    ["wood_stockpile"],
                )
