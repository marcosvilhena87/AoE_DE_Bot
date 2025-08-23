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

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.common as common
import script.resources as resources
import script.villager as villager


class TestResourceReadRetry(TestCase):
    def setUp(self):
        resources._LAST_RESOURCE_VALUES.clear()
        resources._LAST_RESOURCE_TS.clear()
        resources._LAST_READ_FROM_CACHE.clear()

    def tearDown(self):
        resources._LAST_RESOURCE_VALUES.clear()
        resources._LAST_RESOURCE_TS.clear()
        resources._LAST_READ_FROM_CACHE.clear()

    def test_required_icon_fallback(self):
        def fake_detect(frame, required_icons):
            return {"wood_stockpile": (0, 0, 50, 50)}

        ocr_seq = [
            ("123", {"text": ["123"]}, np.zeros((1, 1), dtype=np.uint8)),
            ("", {"text": [""]}, np.zeros((1, 1), dtype=np.uint8)),
        ]

        def fake_ocr(gray):
            return ocr_seq.pop(0)

        frame = np.zeros((600, 600, 3), dtype=np.uint8)

        with patch("script.resources.detect_resource_regions", side_effect=fake_detect), \
             patch("script.screen_utils._grab_frame", return_value=frame), \
             patch("script.resources._ocr_digits_better", side_effect=fake_ocr), \
             patch("script.resources.pytesseract.image_to_string", return_value=""), \
             patch("script.resources.cv2.imwrite"):
            first = resources.read_resources_from_hud(["wood_stockpile"])
            second = resources.read_resources_from_hud(["wood_stockpile"])

        self.assertEqual(first["wood_stockpile"], 123)
        self.assertEqual(second["wood_stockpile"], 123)
        self.assertIn("wood_stockpile", resources._LAST_READ_FROM_CACHE)

    def test_econ_loop_uses_cached_value(self):
        common.CURRENT_POP = 8
        common.POP_CAP = 10
        common.CFG.setdefault("areas", {}).update({"food_spot": (0, 0), "wood_spot": (0, 0)})
        common.CFG.setdefault("timers", {}).update({"idle_gap": 0, "loop_sleep": 0})

        def fake_read(_, force_delay=None):
            resources._LAST_READ_FROM_CACHE = {"idle_villager"}
            return {"idle_villager": 0}

        def fake_time():
            fake_time.calls += 1
            if fake_time.calls < 4:
                return 0
            return 1

        fake_time.calls = 0

        with patch("script.villager.select_idle_villager", lambda: None), \
             patch("script.villager.build_granary", return_value=True), \
             patch("script.villager.build_storage_pit", return_value=True), \
             patch("script.villager.build_house", return_value=True), \
             patch("script.input_utils._click_norm", lambda *a, **k: None), \
             patch("script.villager.time.sleep", lambda s: None), \
             patch("script.resources.read_resources_from_hud", side_effect=fake_read), \
             patch("script.villager.time.time", side_effect=fake_time), \
             self.assertLogs(level="WARNING") as log:
            villager.econ_loop(minutes=1 / 60)

        self.assertTrue(any("cached idle villager" in msg for msg in log.output))
