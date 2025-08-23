import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np

# Stub GUI-dependent modules before importing bot modules

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


class TestResourceDebugImages(TestCase):
    def test_debug_images_written_when_all_none_and_debug_off(self):
        common.CFG["debug"] = False
        common.CFG.get("resource_panel", {})["debug_failed_ocr"] = False

        def fake_grab_frame(bbox=None):
            if bbox:
                return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
            return np.zeros((600, 600, 3), dtype=np.uint8)

        def fake_ocr(gray):
            return "", {"text": [""]}, np.zeros((1, 1), dtype=np.uint8)

        with patch("script.common._grab_frame", side_effect=fake_grab_frame), \
             patch(
                 "script.common.locate_resource_panel",
                 return_value={"wood_stockpile": (0, 0, 50, 50)},
             ), \
             patch("script.common._ocr_digits_better", side_effect=fake_ocr), \
             patch("script.common.pytesseract.image_to_string", return_value=""), \
             patch("script.common.cv2.imwrite") as imwrite_mock:
            with self.assertRaises(common.ResourceReadError):
                common.read_resources_from_hud()
        paths = [call.args[0] for call in imwrite_mock.call_args_list]
        debug_dir = common.ROOT / "debug"
        self.assertGreaterEqual(imwrite_mock.call_count, 2)
        self.assertTrue(any("resource_panel_fail" in p for p in paths))
        self.assertTrue(any("resource_wood_stockpile_roi" in p for p in paths))
        self.assertTrue(all(str(debug_dir) in p for p in paths))

    def test_roi_images_written_when_single_resource_missing(self):
        common.CFG["debug"] = False
        common.CFG.get("resource_panel", {})["debug_failed_ocr"] = False

        def fake_grab_frame(bbox=None):
            if bbox:
                return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
            return np.zeros((600, 600, 3), dtype=np.uint8)

        ocr_results = [
            ("123", {"text": ["123"]}, np.zeros((1, 1), dtype=np.uint8)),
            ("", {"text": [""]}, np.zeros((1, 1), dtype=np.uint8)),
        ]

        with patch("script.common._grab_frame", side_effect=fake_grab_frame), \
             patch(
                 "script.common.locate_resource_panel",
                 return_value={
                     "wood_stockpile": (0, 0, 50, 50),
                     "food_stockpile": (50, 0, 50, 50),
                 },
             ), \
             patch("script.common._ocr_digits_better", side_effect=ocr_results), \
             patch("script.common.pytesseract.image_to_string", return_value=""), \
             patch("script.common.cv2.imwrite") as imwrite_mock:
            result = common.read_resources_from_hud()

        self.assertEqual(result["wood_stockpile"], 123)
        self.assertIsNone(result["food_stockpile"])

        paths = [call.args[0] for call in imwrite_mock.call_args_list]
        debug_dir = common.ROOT / "debug"
        self.assertTrue(any("resource_food_stockpile_roi" in p for p in paths))
        self.assertTrue(any("resource_food_stockpile_thresh" in p for p in paths))
        self.assertTrue(all(str(debug_dir) in p for p in paths))
