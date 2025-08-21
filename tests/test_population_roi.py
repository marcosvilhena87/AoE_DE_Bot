import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np

# Stub modules that require a GUI/display before importing campaign_bot
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
import campaign_bot as cb


class TestPopulationROI(TestCase):
    def test_population_roi_outside_hud_does_not_raise(self):
        cb.HUD_REGION = {"left": 0, "top": 0, "width": 100, "height": 100}

        small_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        big_frame = np.zeros((200, 200, 3), dtype=np.uint8)

        def fake_grab_frame(bbox=None):
            if bbox is cb.MONITOR:
                return big_frame
            return small_frame

        with patch("campaign_bot._grab_frame", side_effect=fake_grab_frame), \
            patch("campaign_bot._screen_size", return_value=(200, 200)), \
            patch("campaign_bot.pytesseract.image_to_data", return_value={"text": [""], "conf": ["-1"]}):
            try:
                cb.read_population_from_hud(retries=1)
            except Exception as exc:  # pragma: no cover
                self.fail(f"read_population_from_hud raised {exc}")

