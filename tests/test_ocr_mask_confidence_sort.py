import os
import sys
import types
from unittest.mock import patch

import numpy as np

# Stub modules that require a GUI/display before importing bot modules

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
    monitors = [{}, {"left": 0, "top": 0, "width": 1, "height": 1}]

    def grab(self, region):
        h, w = region["height"], region["width"]
        return np.zeros((h, w, 4), dtype=np.uint8)


sys.modules.setdefault("pyautogui", dummy_pg)
sys.modules.setdefault("mss", types.SimpleNamespace(mss=lambda: DummyMSS()))

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from script.resources.ocr.masks import _run_masks


def test_longer_digits_preferred_over_shorter_even_if_confidence_lower():
    masks = [np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8)]
    psms = [6]
    outputs = [
        {"text": ["0"], "conf": ["91"]},
        {"text": ["140"], "conf": ["90", "90", "90"]},
    ]
    with patch("script.resources.ocr.masks.pytesseract.image_to_data", side_effect=outputs):
        digits, data, mask = _run_masks(masks, psms, False, None, 0)
    assert digits == "140"


def test_population_limit_prefers_candidates_with_slash_over_more_digits():
    masks = [np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8)]
    psms = [6]
    outputs = [
        {"text": ["775"], "conf": ["91", "91", "91"]},
        {"text": ["3/4"], "conf": ["89", "89", "89"]},
    ]
    with patch("script.resources.ocr.masks.pytesseract.image_to_data", side_effect=outputs):
        digits, data, mask = _run_masks(
            masks, psms, False, None, 0, resource="population_limit"
        )
    assert digits == "34"
