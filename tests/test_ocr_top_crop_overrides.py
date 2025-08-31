import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np

# Stub external dependencies before importing the module under test
os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

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


dummy_cv2 = types.SimpleNamespace(
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
    IMREAD_GRAYSCALE=0,
    COLOR_BGR2GRAY=0,
    INTER_LINEAR=0,
    THRESH_BINARY=0,
    THRESH_OTSU=0,
    ADAPTIVE_THRESH_GAUSSIAN_C=0,
    ADAPTIVE_THRESH_MEAN_C=0,
    TM_CCOEFF_NORMED=0,
)

sys.modules.setdefault("pyautogui", dummy_pg)
sys.modules.setdefault("mss", types.SimpleNamespace(mss=lambda: DummyMSS()))
sys.modules.setdefault("cv2", dummy_cv2)
sys.modules.setdefault(
    "pytesseract",
    types.SimpleNamespace(
        image_to_data=lambda *a, **k: {"text": [""], "conf": ["0"]},
        image_to_string=lambda *a, **k: "",
        Output=types.SimpleNamespace(DICT="dict"),
        pytesseract=types.SimpleNamespace(tesseract_cmd=""),
    ),
)

# Make sure the script package is importable
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.resources.reader.roi as roi


class TestOcrTopCropOverrides(TestCase):
    def test_prepare_roi_uses_overrides(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        regions = {
            "population_limit": (0, 0, 10, 10),
            "idle_villager": (0, 0, 10, 10),
        }

        gray = np.arange(100).reshape(10, 10)

        with patch.dict(
            roi.CFG,
            {
                "ocr_top_crop": 2,
                "ocr_top_crop_overrides": {
                    "population_limit": 0,
                    "idle_villager": 0,
                },
            },
            clear=False,
        ), patch(
            "script.resources.reader.roi.preprocess_roi", return_value=gray
        ), patch(
            "script.resources.reader.roi.get_narrow_roi_deficit", return_value=0
        ), patch(
            "script.resources.reader.roi.get_failure_count", return_value=0
        ):
            for name in ("population_limit", "idle_villager"):
                result = roi.prepare_roi(frame, regions, name, set(), object())
                self.assertIsNotNone(result)
                _x, _y, _w, _h, _roi, gray_out, top_crop, _fail = result
                self.assertEqual(top_crop, 0)
                self.assertEqual(gray_out.shape, gray.shape)

