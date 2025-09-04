import os
import sys
import types
from unittest.mock import patch

import numpy as np

# Ensure test modules can import without real dependencies
os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from script.resources.ocr import masks


def test_population_fraction_high_confidence():
    gray = np.zeros((25, 25), dtype=np.uint8)
    np.fill_diagonal(gray, 255)

    k_sizes = []
    def get_struct_elem(shape, ksize):
        k_sizes.append(ksize)
        return np.ones(ksize, np.uint8)

    cv2_stub = types.SimpleNamespace(
        bilateralFilter=lambda img, *a, **k: img,
        equalizeHist=lambda img: img,
        normalize=lambda src, *a, **k: src,
        adaptiveThreshold=lambda src, *a, **k: np.zeros_like(src),
        countNonZero=np.count_nonzero,
        getStructuringElement=get_struct_elem,
        dilate=lambda src, kernel, iterations=1: src,
        inRange=lambda hsv, lower, upper: np.zeros_like(hsv),
        cvtColor=lambda src, code: src,
        createCLAHE=lambda clipLimit, tileGridSize: types.SimpleNamespace(apply=lambda img: img),
        morphologyEx=lambda src, op, kernel, iterations=1: src,
        Canny=lambda src, t1, t2: np.zeros_like(src),
        bitwise_not=lambda src: src,
        bitwise_or=lambda a, b: a,
        threshold=lambda src, thresh, maxval, type: (None, np.zeros_like(src)),
        resize=lambda img, *a, **k: img,
        INTER_LINEAR=0,
        THRESH_BINARY=0,
        THRESH_OTSU=0,
        MORPH_RECT=0,
        MORPH_CLOSE=0,
        MORPH_CROSS=0,
        ADAPTIVE_THRESH_MEAN_C=0,
        ADAPTIVE_THRESH_GAUSSIAN_C=0,
        NORM_MINMAX=0,
    )

    side_effects = [
        ("", {"text": [], "conf": []}, None),
        ("34", {"text": ["3/4"], "conf": ["95"]}, None),
    ]

    with patch.object(masks, "cv2", cv2_stub), \
         patch("script.resources.ocr.masks._run_masks", side_effect=side_effects), \
         patch.dict(masks.CFG, {"population_morph_close": {"kernel": [1, 2]}, "ocr_conf_threshold": 45}, clear=False):
        digits, data, _mask = masks._ocr_digits_better(gray, resource="population_limit")

    assert digits == "34"
    assert int(data["conf"][0]) >= masks.CFG["ocr_conf_threshold"]
    assert (1, 2) in k_sizes
