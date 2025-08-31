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


def test_high_confidence_shorter_digits_win():
    masks = [np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8)]
    psms = [6]
    outputs = [
        {"text": ["1234"], "conf": ["10", "10", "10", "10"]},
        {"text": ["99"], "conf": ["90", "90"]},
    ]
    with patch("script.resources.ocr.masks.pytesseract.image_to_data", side_effect=outputs):
        digits, data, mask = _run_masks(masks, psms, False, None, 0)
    assert digits == "99"
