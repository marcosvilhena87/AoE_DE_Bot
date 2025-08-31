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
        imwrite=lambda *a, **k: True,
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
import script.resources as resources


class TestPopulationROI(TestCase):
    def test_population_roi_outside_screen_raises_error(self):
        with patch("script.input_utils._screen_size", return_value=(200, 200)), \
            patch.dict(common.CFG["areas"], {"pop_box": [2.0, 2.0, 0.1, 0.1]}), \
            patch.dict(common.CFG, {"population_limit_roi": None}, clear=False), \
            patch("script.resources.locate_resource_panel", return_value={}), \
            patch("script.screen_utils._grab_frame", return_value=np.zeros((1, 1, 3))) as grab_mock, \
            patch("script.resources.ocr.executor.execute_ocr") as ocr_mock:
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
            patch("script.resources.panel._auto_calibrate_from_icons", return_value={}), \
            patch("script.resources.cv2.cvtColor", side_effect=lambda img, code: img), \
            patch("script.resources.cv2.resize", side_effect=lambda img, *a, **k: img), \
            patch("script.resources.cv2.threshold", side_effect=lambda img, *a, **k: (None, img)), \
            patch(
                "script.resources.ocr.executor.execute_ocr",
                return_value=("", {"text": ["xx"], "conf": ["70"]}, None, False),
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
            patch("script.resources.panel._auto_calibrate_from_icons", return_value={}), \
            patch("script.resources.cv2.cvtColor", side_effect=fake_cvtColor), \
            patch("script.resources.cv2.resize", side_effect=lambda img, *a, **k: img), \
            patch("script.resources.cv2.threshold", side_effect=lambda img, *a, **k: (None, img)), \
            patch(
                "script.resources.ocr.executor.execute_ocr",
                return_value=("1234", {"text": ["12", "34"], "conf": ["70"]}, None, False),
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
            patch("script.resources.ocr.executor.execute_ocr") as ocr_mock, \
            patch("script.resources.ocr.executor.time.sleep") as sleep_mock:
            with self.assertRaises(common.PopulationReadError) as ctx:
                hud.read_population_from_hud(
                    retries=1, conf_threshold=common.CFG["ocr_conf_threshold"]
                )
            msg = str(ctx.exception).lower()
            self.assertIn("recalibrate areas.pop_box", msg)
            grab_mock.assert_called_once()
            ocr_mock.assert_not_called()
            sleep_mock.assert_not_called()

    def test_population_roi_expands_and_succeeds(self):
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        bbox = {"left": 10, "top": 10, "width": 4, "height": 4}
        widths = []

        def fake_grab(bbox=None):
            if bbox is None:
                return frame
            l, t, w, h = (
                bbox["left"],
                bbox["top"],
                bbox["width"],
                bbox["height"],
            )
            return frame[t : t + h, l : l + w]

        def fake_pop(roi, conf_threshold=None, roi_bbox=None):
            widths.append(roi.shape[1])
            if roi.shape[1] <= 4:
                raise common.PopulationReadError("tight")
            return 12, 34

        resources.RESOURCE_CACHE.resource_failure_counts.pop("population_limit", None)
        with patch.dict(
            common.CFG,
            {
                "population_ocr_roi_expand_px": 3,
                "population_ocr_roi_expand_step": 0,
                "population_ocr_roi_expand_growth": 1.0,
            },
            clear=False,
        ), patch("script.screen_utils._grab_frame", side_effect=fake_grab), patch(
            "script.resources.ocr.executor._read_population_from_roi",
            side_effect=fake_pop,
        ), patch(
            "script.resources.reader.roi._read_population_from_roi",
            side_effect=fake_pop,
        ):
            cur, cap = resources.read_population_from_roi(bbox, retries=1)

        self.assertEqual((cur, cap), (12, 34))
        self.assertGreater(widths[1], widths[0])
        self.assertEqual(
            resources.RESOURCE_CACHE.resource_failure_counts.get("population_limit"),
            0,
        )

    def test_population_string_with_slash(self):
        roi = np.zeros((10, 10, 3), dtype=np.uint8)
        gray = np.zeros((10, 10), dtype=np.uint8)
        with patch(
            "script.resources.ocr.executor.preprocess_roi", return_value=gray
        ), patch(
            "script.resources.ocr.executor.execute_ocr",
            return_value=("2025", {"text": ["20/25"], "conf": ["80"]}, None, False),
        ) as ocr_mock:
            cur, cap = resources._read_population_from_roi(roi)

        self.assertEqual((cur, cap), (20, 25))
        ocr_mock.assert_called_once()
        _, kwargs = ocr_mock.call_args
        self.assertEqual(kwargs.get("whitelist"), "0123456789/")

    def test_population_string_without_slash_two_digits(self):
        roi = np.zeros((10, 10, 3), dtype=np.uint8)
        gray = np.zeros((10, 10), dtype=np.uint8)
        with patch(
            "script.resources.ocr.executor.preprocess_roi", return_value=gray
        ), patch(
            "script.resources.ocr.executor.execute_ocr",
            return_value=("34", {"text": ["34"], "conf": ["80"]}, None, False),
        ) as ocr_mock:
            cur, cap = resources._read_population_from_roi(roi)

        self.assertEqual((cur, cap), (3, 4))
        ocr_mock.assert_called_once()
        _, kwargs = ocr_mock.call_args
        self.assertEqual(kwargs.get("whitelist"), "0123456789/")

