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
    def test_population_roi_outside_screen_raises_error(self):
        with patch("campaign_bot._screen_size", return_value=(200, 200)), \
            patch.dict(cb.CFG["areas"], {"pop_box": [2.0, 2.0, 0.1, 0.1]}), \
            patch("campaign_bot._grab_frame") as grab_mock, \
            patch("campaign_bot.pytesseract.image_to_data") as ocr_mock:
            with self.assertRaises(cb.PopulationReadError) as ctx:
                cb.read_population_from_hud(
                    retries=1, conf_threshold=cb.CFG["ocr_conf_threshold"]
                )
            msg = str(ctx.exception).lower()
            self.assertIn("recalibrate areas.pop_box", msg)
            self.assertIn("left=", msg)
            self.assertIn("top=", msg)
            self.assertIn("hud_anchor", msg)
            grab_mock.assert_not_called()
            ocr_mock.assert_not_called()

    def test_read_population_raises_when_no_digits(self):
        def fake_grab(bbox):
            h, w = bbox["height"], bbox["width"]
            return np.zeros((h, w, 3), dtype=np.uint8)

        with patch("campaign_bot._grab_frame", side_effect=fake_grab), \
            patch("campaign_bot._screen_size", return_value=(200, 200)), \
            patch.dict(cb.CFG["areas"], {"pop_box": [0.1, 0.1, 0.5, 0.5]}), \
            patch("campaign_bot.cv2.cvtColor", side_effect=lambda img, code: img), \
            patch("campaign_bot.cv2.resize", side_effect=lambda img, *a, **k: img), \
            patch("campaign_bot.cv2.threshold", side_effect=lambda img, *a, **k: (None, img)), \
            patch(
                "campaign_bot.pytesseract.image_to_data",
                return_value={"text": ["xx"], "conf": ["70"]},
            ):
            with self.assertRaises(cb.PopulationReadError):
                cb.read_population_from_hud(retries=1, conf_threshold=cb.CFG["ocr_conf_threshold"])

    def test_population_roi_respects_anchor_offsets(self):
        frame = np.arange(200 * 200 * 3, dtype=np.uint8).reshape(200, 200, 3)
        pop_box = [0.1, 0.2, 0.5, 0.4]

        anchors = [
            {"left": 10, "top": 20, "width": 50, "height": 50},
            {"left": 30, "top": 40, "width": 50, "height": 50},
        ]

        for anchor in anchors:
            recorded = {}

            def fake_grab(bbox):
                recorded["bbox"] = bbox
                l, t, w, h = (
                    bbox["left"],
                    bbox["top"],
                    bbox["width"],
                    bbox["height"],
                )
                return frame[t : t + h, l : l + w]

            def fake_cvtColor(src, code):
                recorded["roi"] = src
                return src

            with patch("campaign_bot._grab_frame", side_effect=fake_grab), \
                patch.object(cb, "HUD_ANCHOR", anchor), \
                patch.dict(cb.CFG["areas"], {"pop_box": pop_box}), \
                patch("campaign_bot.cv2.cvtColor", side_effect=fake_cvtColor), \
                patch("campaign_bot.cv2.resize", side_effect=lambda img, *a, **k: img), \
                patch("campaign_bot.cv2.threshold", side_effect=lambda img, *a, **k: (None, img)), \
                patch(
                    "campaign_bot.pytesseract.image_to_data",
                    return_value={"text": ["12/34"], "conf": ["70"]},
                ):
                cb.read_population_from_hud(retries=1, conf_threshold=cb.CFG["ocr_conf_threshold"])

            roi = recorded["roi"]
            bbox = recorded["bbox"]
            expected_left = anchor["left"] + int(pop_box[0] * anchor["width"])
            expected_top = anchor["top"] + int(pop_box[1] * anchor["height"])
            expected_w = int(pop_box[2] * anchor["width"])
            expected_h = int(pop_box[3] * anchor["height"])
            expected_roi = frame[
                expected_top : expected_top + expected_h,
                expected_left : expected_left + expected_w,
            ]
            self.assertTrue(np.array_equal(roi, expected_roi))
            self.assertEqual(
                bbox,
                {
                    "left": expected_left,
                    "top": expected_top,
                    "width": expected_w,
                    "height": expected_h,
                },
            )

    def test_non_positive_population_roi_raises_before_ocr(self):
        with patch("campaign_bot._screen_size", return_value=(200, 200)), \
            patch.dict(cb.CFG["areas"], {"pop_box": [0.1, 0.1, -0.5, 0.2]}), \
            patch("campaign_bot._grab_frame") as grab_mock, \
            patch("campaign_bot.pytesseract.image_to_data") as ocr_mock, \
            patch("campaign_bot.time.sleep") as sleep_mock:
            with self.assertRaises(cb.PopulationReadError) as ctx:
                cb.read_population_from_hud(
                    retries=1, conf_threshold=cb.CFG["ocr_conf_threshold"]
                )
            msg = str(ctx.exception).lower()
            self.assertIn("recalibrate areas.pop_box", msg)
            grab_mock.assert_not_called()
            ocr_mock.assert_not_called()
            sleep_mock.assert_not_called()

