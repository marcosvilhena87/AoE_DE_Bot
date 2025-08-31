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
import script.resources.reader as reader


class TestResourceRoiValidation(TestCase):
    def setUp(self):
        reader.RESOURCE_CACHE.last_resource_values.clear()
        reader.RESOURCE_CACHE.last_resource_ts.clear()
        reader.RESOURCE_CACHE.resource_failure_counts.clear()
        reader._LAST_REGION_SPANS.clear()

    def tearDown(self):
        reader.RESOURCE_CACHE.last_resource_values.clear()
        reader.RESOURCE_CACHE.last_resource_ts.clear()
        reader.RESOURCE_CACHE.resource_failure_counts.clear()
        reader._LAST_REGION_SPANS.clear()

    def test_zero_width_roi_raises(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        def fake_detect(frame, icons, cache_obj):
            return {"wood_stockpile": (0, 0, 0, 10)}

        with patch(
            "script.resources.reader.detect_resource_regions",
            side_effect=fake_detect,
        ), patch(
            "script.resources.reader.core.detect_resource_regions",
            side_effect=fake_detect,
        ):
            with self.assertRaises(common.ResourceReadError):
                reader._read_resources(
                    frame,
                    ["wood_stockpile"],
                    ["wood_stockpile"],
                )

    def test_widen_roi_within_span(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        attempts: list[int] = []

        def fake_execute_ocr(gray, color=None, conf_threshold=None, allow_fallback=True, roi=None, resource=None):
            attempts.append(roi[2])
            if roi[2] == 35:
                return "5", {"text": ["5"]}, None, False
            return None, {"text": []}, None, False

        def fake_prepare_roi(frame, regions, name, required_set, cache_obj):
            x, y, w, h = regions[name]
            roi = frame[y : y + h, x : x + w]
            gray = roi
            return x, y, w, h, roi, gray, 0, 0

        def fake_detect(frame, icons, cache_obj):
            reader.cache._LAST_REGION_SPANS = {"wood_stockpile": (0, 35)}
            return {"wood_stockpile": (0, 0, 30, 10)}

        with patch(
            "script.resources.reader.detect_resource_regions",
            side_effect=fake_detect,
        ), patch(
            "script.resources.reader.core.detect_resource_regions",
            side_effect=fake_detect,
        ), patch(
            "script.resources.reader.core.preprocess_roi",
            side_effect=lambda r: r,
        ), patch(
            "script.resources.reader.core.execute_ocr",
            side_effect=fake_execute_ocr,
        ), patch(
            "script.resources.reader.core.prepare_roi",
            side_effect=fake_prepare_roi,
        ), patch(
            "script.resources.reader.core.expand_roi_after_failure",
            return_value=None,
        ):
            results, _ = reader._read_resources(
                frame,
                ["wood_stockpile"],
                ["wood_stockpile"],
            )

        self.assertEqual(results["wood_stockpile"], 5)
        self.assertIn(35, attempts)
        self.assertEqual(max(attempts), 35)

