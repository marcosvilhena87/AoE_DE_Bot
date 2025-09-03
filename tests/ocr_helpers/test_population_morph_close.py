import os
import sys
import types
from unittest.mock import patch

import numpy as np

# Ensure test modules can import without real dependencies
os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from script.resources.ocr import masks

# Augment dummy cv2 with functions required for _ocr_digits_better
cv2 = masks.cv2
cv2.bilateralFilter = lambda img, *a, **k: img
cv2.equalizeHist = lambda img: img
cv2.normalize = lambda src, *a, **k: src
cv2.adaptiveThreshold = lambda src, maxval, adaptiveMethod, thresholdType, blockSize, C: np.zeros_like(src)
cv2.countNonZero = np.count_nonzero
cv2.getStructuringElement = lambda shape, ksize: np.ones(ksize, np.uint8)
cv2.dilate = lambda src, kernel, iterations=1: src
cv2.inRange = lambda hsv, lower, upper: np.zeros_like(hsv)
cv2.cvtColor = lambda src, code: src
cv2.createCLAHE = lambda clipLimit, tileGridSize: types.SimpleNamespace(apply=lambda img: img)
cv2.morphologyEx = lambda src, op, kernel, iterations=1: src
cv2.Canny = lambda src, t1, t2: np.zeros_like(src)
cv2.MORPH_RECT = 0
cv2.MORPH_CLOSE = 0
cv2.MORPH_CROSS = 0
cv2.ADAPTIVE_THRESH_MEAN_C = 0
cv2.ADAPTIVE_THRESH_GAUSSIAN_C = 0
cv2.NORM_MINMAX = 0


def test_population_morph_close_skipped_for_high_variance_preserves_fraction():
    gray = np.zeros((10, 10), dtype=np.uint8)
    gray[:, :5] = 255  # high variance

    side_effects = [
        ("", {"text": [], "conf": []}, None),
        ("34", {"text": ["3/4"], "conf": ["90"]}, None),
    ]

    with patch("script.resources.ocr.masks.cv2.morphologyEx") as morph_mock, \
         patch("script.resources.ocr.masks._run_masks", side_effect=side_effects):
        with patch.dict(masks.CFG, {"population_morph_close": {"variance_threshold": 1}}, clear=False):
            digits, data, mask = masks._ocr_digits_better(gray, resource="population_limit")

    assert digits == "34"
    morph_mock.assert_not_called()


def test_population_morph_close_skipped_when_slash_detected():
    gray = np.zeros((10, 10), dtype=np.uint8)
    np.fill_diagonal(gray, 255)

    side_effects = [
        ("", {"text": [], "conf": []}, None),
        ("34", {"text": ["3/4"], "conf": ["95"]}, None),
    ]

    with patch("script.resources.ocr.masks.cv2.Canny", return_value=np.ones_like(gray)), \
         patch(
             "script.resources.ocr.masks.cv2.morphologyEx",
             side_effect=lambda src, op, kernel, iterations=1: src,
         ) as morph_mock, \
         patch("script.resources.ocr.masks._run_masks", side_effect=side_effects):
        digits, data, mask = masks._ocr_digits_better(gray, resource="population_limit")

    assert digits == "34"
    assert "/" in "".join(data["text"])
    assert morph_mock.call_count == 0
    assert int(data["conf"][0]) >= 90
