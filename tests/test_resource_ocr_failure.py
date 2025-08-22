import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np

# Stub modules that require a GUI/display before importing the bot modules


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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.common as common


class TestResourceOcrFailure(TestCase):
    def test_read_resources_ocr_failure(self):
        def fake_grab_frame(bbox=None):
            if bbox:
                return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
            return np.zeros((600, 600, 3), dtype=np.uint8)

        def fake_ocr(gray):
            return "", {"text": [""]}, np.zeros((1, 1), dtype=np.uint8)

        with patch("script.common._grab_frame", side_effect=fake_grab_frame), \
             patch("script.common.locate_resource_panel", return_value={"wood": (0, 0, 50, 50)}), \
             patch("script.common._ocr_digits_better", side_effect=fake_ocr):
            with self.assertRaises(common.ResourceReadError):
                common.read_resources_from_hud()
