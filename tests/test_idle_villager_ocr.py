import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch, ANY

import numpy as np

# Stub modules requiring a GUI

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

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.resources as resources


class TestIdleVillagerOCR(TestCase):
    def setUp(self):
        resources._LAST_RESOURCE_VALUES.clear()
        resources._LAST_RESOURCE_TS.clear()
        resources._RESOURCE_FAILURE_COUNTS.clear()

    def test_idle_villager_uses_digit_only_tesseract(self):
        def fake_detect(frame, required_icons):
            return {"idle_villager": (0, 0, 50, 50)}

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch("script.resources.detect_resource_regions", side_effect=fake_detect), \
             patch("script.screen_utils._grab_frame", return_value=frame), \
             patch("script.resources.pytesseract.image_to_string", return_value="12") as img2str, \
             patch("script.resources._ocr_digits_better") as ocr_mock:
            result = resources.read_resources_from_hud(["idle_villager"])

        self.assertEqual(result["idle_villager"], 12)
        img2str.assert_called_once_with(ANY, config="--psm 7 -c tessedit_char_whitelist=0123456789")
        ocr_mock.assert_not_called()
