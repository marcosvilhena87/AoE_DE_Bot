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
    def test_population_roi_outside_screen_does_not_raise(self):
        big_frame = np.zeros((200, 200, 3), dtype=np.uint8)

        with patch("campaign_bot._grab_frame", return_value=big_frame), \
            patch("campaign_bot._screen_size", return_value=(200, 200)), \
            patch.dict(cb.CFG["areas"], {"pop_box": [2.0, 2.0, 0.1, 0.1]}), \
            patch(
                "campaign_bot.pytesseract.image_to_data",
                return_value={"text": [""], "conf": ["-1"]},
            ):
            try:
                cb.read_population_from_hud(retries=1)
            except Exception as exc:  # pragma: no cover
                self.fail(f"read_population_from_hud raised {exc}")

    def test_population_roi_respects_anchor_offsets(self):
        frame = np.arange(200 * 200 * 3, dtype=np.uint8).reshape(200, 200, 3)
        pop_box = [0.1, 0.2, 0.5, 0.4]

        anchors = [
            {"left": 10, "top": 20, "width": 50, "height": 50},
            {"left": 30, "top": 40, "width": 50, "height": 50},
        ]

        for anchor in anchors:
            recorded = {}

            def fake_cvtColor(src, code):
                recorded["roi"] = src
                return src

            with patch("campaign_bot._grab_frame", return_value=frame), \
                patch.object(cb, "HUD_ANCHOR", anchor), \
                patch.dict(cb.CFG["areas"], {"pop_box": pop_box}), \
                patch("campaign_bot.cv2.cvtColor", side_effect=fake_cvtColor), \
                patch("campaign_bot.cv2.resize", side_effect=lambda img, *a, **k: img), \
                patch("campaign_bot.cv2.threshold", side_effect=lambda img, *a, **k: (None, img)), \
                patch(
                    "campaign_bot.pytesseract.image_to_data",
                    return_value={"text": ["12/34"], "conf": ["70"]},
                ):
                cb.read_population_from_hud(retries=1)

            roi = recorded["roi"]
            expected_left = anchor["left"] + int(pop_box[0] * anchor["width"])
            expected_top = anchor["top"] + int(pop_box[1] * anchor["height"])
            expected_w = int(pop_box[2] * anchor["width"])
            expected_h = int(pop_box[3] * anchor["height"])
            expected_roi = frame[
                expected_top : expected_top + expected_h,
                expected_left : expected_left + expected_w,
            ]
            self.assertTrue(np.array_equal(roi, expected_roi))

