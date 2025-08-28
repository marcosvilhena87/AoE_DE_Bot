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
sys.modules.setdefault(
    "cv2",
    types.SimpleNamespace(
        cvtColor=lambda src, code: src,
        resize=lambda img, *a, **k: img,
        threshold=lambda img, *a, **k: (None, img),
        imread=lambda *a, **k: np.zeros((1, 1), dtype=np.uint8),
        IMREAD_GRAYSCALE=0,
        COLOR_BGR2GRAY=0,
        INTER_LINEAR=0,
        THRESH_BINARY=0,
        THRESH_OTSU=0,
    ),
)
os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.common as common
import script.hud as hud


class TestPopulationROI(TestCase):
    def test_population_roi_outside_screen_raises_error(self):
        with patch("script.input_utils._screen_size", return_value=(200, 200)), \
            patch.dict(common.CFG["areas"], {"pop_box": [2.0, 2.0, 0.1, 0.1]}), \
            patch.dict(common.CFG, {"population_limit_roi": None}, clear=False), \
            patch("script.resources.locate_resource_panel", return_value={}), \
            patch("script.screen_utils._grab_frame", return_value=np.zeros((1, 1, 3))) as grab_mock, \
            patch("script.hud.pytesseract.image_to_data") as ocr_mock:
            with self.assertRaises(common.PopulationReadError) as ctx:
                hud.read_population_from_hud(
                    retries=1, conf_threshold=common.CFG["ocr_conf_threshold"]
                )
            msg = str(ctx.exception).lower()
            self.assertIn("recalibrate areas.pop_box", msg)
            self.assertIn("left=", msg)
            self.assertIn("top=", msg)
            self.assertNotIn("hud_anchor", msg)
            grab_mock.assert_called_once()
            ocr_mock.assert_not_called()

    def test_read_population_raises_when_no_digits(self):
        def fake_grab(bbox=None):
            if bbox is None:
                return np.zeros((200, 200, 3), dtype=np.uint8)
            h, w = bbox["height"], bbox["width"]
            return np.zeros((h, w, 3), dtype=np.uint8)

        with patch("script.screen_utils._grab_frame", side_effect=fake_grab), \
            patch("script.resources.locate_resource_panel", return_value={}), \
            patch("script.input_utils._screen_size", return_value=(200, 200)), \
            patch.dict(common.CFG["areas"], {"pop_box": [0.1, 0.1, 0.5, 0.5]}), \
            patch.dict(common.CFG, {"population_limit_roi": None}, clear=False), \
            patch("script.hud.cv2.cvtColor", side_effect=lambda img, code: img), \
            patch("script.hud.cv2.resize", side_effect=lambda img, *a, **k: img), \
            patch("script.hud.cv2.threshold", side_effect=lambda img, *a, **k: (None, img)), \
            patch(
                "script.hud.pytesseract.image_to_data",
                return_value={"text": ["xx"], "conf": ["70"]},
            ):
            with self.assertRaises(common.PopulationReadError):
                hud.read_population_from_hud(
                    retries=1, conf_threshold=common.CFG["ocr_conf_threshold"]
                )

    def test_population_roi_ignores_hud_anchor(self):
        frame = np.arange(200 * 200 * 3, dtype=np.uint8).reshape(200, 200, 3)
        pop_box = [0.1, 0.2, 0.5, 0.4]

        recorded = {}

        def fake_grab(bbox=None):
            if bbox is None:
                return frame
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

        with patch("script.screen_utils._grab_frame", side_effect=fake_grab), \
            patch("script.resources.locate_resource_panel", return_value={}), \
            patch("script.input_utils._screen_size", return_value=(200, 200)), \
            patch.dict(common.CFG["areas"], {"pop_box": pop_box}), \
            patch.dict(common.CFG, {"population_limit_roi": None}, clear=False), \
            patch("script.common.HUD_ANCHOR", {"left": 50, "top": 60, "width": 10, "height": 10}), \
            patch("script.hud.cv2.cvtColor", side_effect=fake_cvtColor), \
            patch("script.hud.cv2.resize", side_effect=lambda img, *a, **k: img), \
            patch("script.hud.cv2.threshold", side_effect=lambda img, *a, **k: (None, img)), \
            patch(
                "script.hud.pytesseract.image_to_data",
                return_value={"text": ["12/34"], "conf": ["70"]},
            ):
            hud.read_population_from_hud(
                retries=1, conf_threshold=common.CFG["ocr_conf_threshold"]
            )

        roi = recorded["roi"]
        bbox = recorded["bbox"]
        expected_left = int(pop_box[0] * 200)
        expected_top = int(pop_box[1] * 200)
        expected_w = int(pop_box[2] * 200)
        expected_h = int(pop_box[3] * 200)
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
        with patch("script.input_utils._screen_size", return_value=(200, 200)), \
            patch.dict(common.CFG["areas"], {"pop_box": [0.1, 0.1, -0.5, 0.2]}), \
            patch.dict(common.CFG, {"population_limit_roi": None}, clear=False), \
            patch("script.resources.locate_resource_panel", return_value={}), \
            patch("script.screen_utils._grab_frame", return_value=np.zeros((1, 1, 3))) as grab_mock, \
            patch("script.hud.pytesseract.image_to_data") as ocr_mock, \
            patch("script.hud.time.sleep") as sleep_mock:
            with self.assertRaises(common.PopulationReadError) as ctx:
                hud.read_population_from_hud(
                    retries=1, conf_threshold=common.CFG["ocr_conf_threshold"]
                )
            msg = str(ctx.exception).lower()
            self.assertIn("recalibrate areas.pop_box", msg)
            grab_mock.assert_called_once()
            ocr_mock.assert_not_called()
            sleep_mock.assert_not_called()
