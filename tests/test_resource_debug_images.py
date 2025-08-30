import os
import sys
import types
import tempfile
from pathlib import Path
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
import script.resources as resources


class TestResourceDebugImages(TestCase):
    def setUp(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()

    def test_debug_images_written_when_all_none_and_debug_off(self):
        common.CFG["debug"] = False
        common.CFG.get("resource_panel", {})["debug_failed_ocr"] = False

        def fake_grab_frame(bbox=None):
            if bbox:
                return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
            return np.zeros((600, 600, 3), dtype=np.uint8)

        def fake_ocr(gray):
            return "", {"text": [""]}, np.zeros((1, 1), dtype=np.uint8)

        with patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
             patch(
                 "script.resources.locate_resource_panel",
                 return_value={
                     "wood_stockpile": (0, 0, 50, 50),
                     "food_stockpile": (50, 0, 50, 50),
                     "gold_stockpile": (100, 0, 50, 50),
                     "stone_stockpile": (150, 0, 50, 50),
                     "population_limit": (200, 0, 50, 50),
                     "idle_villager": (250, 0, 50, 50),
                 },
             ), \
            patch("script.resources._ocr_digits_better", side_effect=fake_ocr), \
            patch(
                "script.resources.pytesseract.image_to_data",
                return_value={"text": [""], "conf": ["0"]},
            ), \
            patch("script.resources.pytesseract.image_to_string", return_value=""), \
            patch("script.resources._read_population_from_roi", return_value=(0, 0)), \
            patch("script.resources.cv2.imwrite") as imwrite_mock:
            with self.assertRaises(common.ResourceReadError):
                resources.read_resources_from_hud()
        paths = [call.args[0] for call in imwrite_mock.call_args_list]
        debug_dir = common.ROOT / "debug"
        self.assertGreaterEqual(imwrite_mock.call_count, 2)
        self.assertTrue(any("resource_panel_fail" in p for p in paths))
        self.assertTrue(any("resource_wood_stockpile_roi" in p for p in paths))
        self.assertTrue(any("ocr_fail_roi" in p for p in paths))
        self.assertTrue(all(str(debug_dir) in p for p in paths))

    def test_roi_images_written_when_single_resource_missing(self):
        common.CFG["debug"] = False
        common.CFG.get("resource_panel", {})["debug_failed_ocr"] = False

        def fake_grab_frame(bbox=None):
            if bbox:
                return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
            return np.zeros((600, 600, 3), dtype=np.uint8)

        ocr_sequence = [
            ("123", {"text": ["123"]}, np.zeros((1, 1), dtype=np.uint8)),
        ] + [
            ("", {"text": [""]}, np.zeros((1, 1), dtype=np.uint8))
            for _ in range(20)
        ]

        def fake_ocr(gray):
            return ocr_sequence.pop(0) if ocr_sequence else (
                "",
                {"text": [""]},
                np.zeros((1, 1), dtype=np.uint8),
            )

        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()
        with patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
             patch(
                 "script.resources.locate_resource_panel",
                 return_value={
                     "wood_stockpile": (0, 0, 50, 50),
                     "food_stockpile": (50, 0, 50, 50),
                     "gold_stockpile": (100, 0, 50, 50),
                     "stone_stockpile": (150, 0, 50, 50),
                     "population_limit": (200, 0, 50, 50),
                     "idle_villager": (250, 0, 50, 50),
                 },
             ), \
             patch("script.resources._ocr_digits_better", side_effect=fake_ocr), \
            patch(
                "script.resources.pytesseract.image_to_data",
                side_effect=[{"text": [""], "conf": ["0"]}] * 20
                + [{"text": ["0"], "conf": ["90"]}],
            ), \
            patch("script.resources.pytesseract.image_to_string", return_value=""), \
            patch("script.resources._read_population_from_roi", return_value=(0, 0)), \
            patch("script.resources.cv2.imwrite") as imwrite_mock:
            with self.assertRaises(common.ResourceReadError) as ctx:
                resources.read_resources_from_hud()

        self.assertIn("food_stockpile", str(ctx.exception))

        paths = [call.args[0] for call in imwrite_mock.call_args_list]
        debug_dir = common.ROOT / "debug"
        self.assertTrue(any("resource_food_stockpile_roi" in p for p in paths))
        self.assertTrue(any("resource_food_stockpile_thresh" in p for p in paths))
        self.assertTrue(any("ocr_fail_roi" in p for p in paths))
        self.assertTrue(all(str(debug_dir) in p for p in paths))

    def test_debug_images_throttled(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        regions = {"wood_stockpile": (0, 0, 10, 10)}
        results = {"wood_stockpile": None}
        cache = resources.ResourceCache()

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch("script.resources.ROOT", Path(tmpdir)), \
             patch("script.resources.ocr.ROOT", Path(tmpdir)), \
             patch("script.resources._RESOURCE_DEBUG_COOLDOWN", 60):

            resources.handle_ocr_failure(frame, regions, results, [], cache_obj=cache)
            debug_dir = Path(tmpdir) / "debug"
            initial = {p.name for p in debug_dir.iterdir()}

            resources.handle_ocr_failure(frame, regions, results, [], cache_obj=cache)
            after = {p.name for p in debug_dir.iterdir()}

        self.assertEqual(initial, after)
