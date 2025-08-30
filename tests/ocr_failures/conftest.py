import os
import sys
import types

import numpy as np
import pytest

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Stub modules that require a GUI/display before importing bot modules

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
            bitwise_or=lambda a, b: a,
            morphologyEx=lambda src, op, kernel, iterations=1: src,
            threshold=lambda src, *a, **k: (None, src),
            rectangle=lambda img, pt1, pt2, color, thickness: img,
            bilateralFilter=lambda src, d, sigmaColor, sigmaSpace: src,
            adaptiveThreshold=lambda src, maxValue, adaptiveMethod, thresholdType, blockSize, C: src,
            dilate=lambda src, kernel, iterations=1: src,
            equalizeHist=lambda src: src,
            inRange=lambda src, lower, upper: np.zeros(src.shape[:2], dtype=np.uint8),
            countNonZero=lambda src: int(np.count_nonzero(src)),
            ADAPTIVE_THRESH_GAUSSIAN_C=0,
            ADAPTIVE_THRESH_MEAN_C=0,
            IMREAD_GRAYSCALE=0,
            COLOR_BGR2GRAY=0,
            COLOR_GRAY2BGR=0,
            COLOR_BGR2HSV=0,
            INTER_LINEAR=0,
            THRESH_BINARY=0,
            THRESH_OTSU=0,
            MORPH_CLOSE=0,
            TM_CCOEFF_NORMED=0,
        ),
    )

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

import pytesseract
import script.resources.reader as resources

pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]


@pytest.fixture(autouse=True)
def reset_resources_cache():
    resources.RESOURCE_CACHE.last_resource_values.clear()
    resources.RESOURCE_CACHE.last_resource_ts.clear()
    resources.RESOURCE_CACHE.resource_failure_counts.clear()
    resources._LAST_REGION_SPANS.clear()
    resources._LAST_READ_FROM_CACHE.clear()
    yield
    resources.RESOURCE_CACHE.last_resource_values.clear()
    resources.RESOURCE_CACHE.last_resource_ts.clear()
    resources.RESOURCE_CACHE.resource_failure_counts.clear()
    resources._LAST_REGION_SPANS.clear()
    resources._LAST_READ_FROM_CACHE.clear()
