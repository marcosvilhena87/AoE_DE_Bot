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
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()

    def test_idle_villager_high_confidence_ocr(self):
        def fake_detect(frame, required_icons, cache=None):
            return {"idle_villager": (0, 0, 50, 50)}

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch("script.resources.detect_resource_regions", side_effect=fake_detect), \
             patch("script.screen_utils._grab_frame", return_value=frame), \
             patch(
                 "script.resources.pytesseract.image_to_data",
                 return_value={"text": ["1", "2"], "conf": ["90", "90"]},
             ) as img2data, \
             patch("script.resources.execute_ocr") as exec_mock:
            result, _ = resources.read_resources_from_hud(["idle_villager"])

        self.assertEqual(result["idle_villager"], 12)
        img2data.assert_called_once_with(
            ANY,
            config="--psm 7 -c tessedit_char_whitelist=0123456789",
            output_type=resources.pytesseract.Output.DICT,
        )
        exec_mock.assert_not_called()

    def test_idle_villager_ignores_non_positive_confidences(self):
        def fake_detect(frame, required_icons, cache=None):
            return {"idle_villager": (0, 0, 50, 50)}

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch("script.resources.detect_resource_regions", side_effect=fake_detect), \
             patch("script.screen_utils._grab_frame", return_value=frame), \
             patch(
                 "script.resources.pytesseract.image_to_data",
                 return_value={"text": ["1", "2"], "conf": [-1, "0", "95"]},
             ) as img2data, \
             patch("script.resources.execute_ocr") as exec_mock:
            result, _ = resources.read_resources_from_hud(["idle_villager"])

        self.assertEqual(result["idle_villager"], 12)
        img2data.assert_called_once_with(
            ANY,
            config="--psm 7 -c tessedit_char_whitelist=0123456789",
            output_type=resources.pytesseract.Output.DICT,
        )
        exec_mock.assert_not_called()

    def test_idle_villager_low_confidence_returns_none(self):
        def fake_detect(frame, required_icons, cache=None):
            return {"idle_villager": (0, 0, 50, 50)}

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch("script.resources.detect_resource_regions", side_effect=fake_detect), \
             patch("script.screen_utils._grab_frame", return_value=frame), \
             patch(
                 "script.resources.pytesseract.image_to_data",
                 return_value={"text": ["1"], "conf": ["30"]},
             ), \
             patch(
                 "script.resources.execute_ocr",
                 return_value=("", {"text": [""], "conf": []}, None, True),
             ):
            result, _ = resources.read_resources_from_hud(
                required_icons=[], icons_to_read=["idle_villager"]
            )

        self.assertIsNone(result.get("idle_villager"))
