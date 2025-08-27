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
        matchTemplate=lambda *a, **k: np.zeros((1, 1), dtype=np.float32),
        minMaxLoc=lambda *a, **k: (0, 0, (0, 0), (0, 0)),
        imread=lambda *a, **k: np.zeros((1, 1), dtype=np.uint8),
        imwrite=lambda *a, **k: True,
        medianBlur=lambda src, k: src,
        bitwise_not=lambda src: src,
        threshold=lambda src, *a, **k: (None, src),
        rectangle=lambda img, pt1, pt2, color, thickness: img,
        IMREAD_GRAYSCALE=0,
        COLOR_BGR2GRAY=0,
        INTER_LINEAR=0,
        THRESH_BINARY=0,
        THRESH_OTSU=0,
        TM_CCOEFF_NORMED=0,
    ),
)

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.common as common
import script.resources as resources


class TestResourceOcrFailure(TestCase):
    def setUp(self):
        resources._LAST_RESOURCE_VALUES.clear()
        resources._LAST_RESOURCE_TS.clear()
        resources._RESOURCE_FAILURE_COUNTS.clear()
        resources._LAST_REGION_SPANS.clear()

    def tearDown(self):
        resources._LAST_RESOURCE_VALUES.clear()
        resources._LAST_RESOURCE_TS.clear()
        resources._RESOURCE_FAILURE_COUNTS.clear()
        resources._LAST_REGION_SPANS.clear()
    def test_read_resources_fallback(self):
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
             patch("script.resources.pytesseract.image_to_string", return_value="123"), \
             patch("script.resources._read_population_from_roi", return_value=(0, 0)):
            result, _ = resources.read_resources_from_hud()
            self.assertEqual(result["wood_stockpile"], 123)

    def test_optional_icon_failure_does_not_raise(self):
        def fake_grab_frame(bbox=None):
            if bbox:
                return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
            return np.zeros((600, 600, 3), dtype=np.uint8)

        def fake_detect(frame, required_icons):
            return {
                "wood_stockpile": (0, 0, 50, 50),
                "food_stockpile": (50, 0, 50, 50),
            }

        ocr_seq = [
            ("123", {"text": ["123"]}, np.zeros((1, 1), dtype=np.uint8)),
            ("", {"text": [""]}, np.zeros((1, 1), dtype=np.uint8)),
        ]

        def fake_ocr(gray):
            return ocr_seq.pop(0)

        with patch("script.resources.detect_resource_regions", side_effect=fake_detect), \
             patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
             patch("script.resources._ocr_digits_better", side_effect=fake_ocr), \
             patch("script.resources.pytesseract.image_to_string", return_value=""), \
             patch("script.resources.cv2.imwrite"):
            result, _ = resources.read_resources_from_hud(["wood_stockpile"])
        self.assertEqual(result.get("wood_stockpile"), 123)
        self.assertIsNone(result.get("food_stockpile"))

    def test_low_confidence_triggers_failure(self):
        def fake_grab_frame(bbox=None):
            if bbox:
                return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
            return np.zeros((600, 600, 3), dtype=np.uint8)

        def fake_detect(frame, required_icons):
            return {"wood_stockpile": (0, 0, 50, 50)}

        def fake_ocr(gray):
            data = {"text": ["123"], "conf": ["10", "20", "30"]}
            return "123", data, np.zeros((1, 1), dtype=np.uint8)

        with patch("script.resources.detect_resource_regions", side_effect=fake_detect), \
            patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
            patch("script.resources._ocr_digits_better", side_effect=fake_ocr), \
            patch("script.resources.pytesseract.image_to_string", return_value="") as img2str_mock, \
            patch("script.resources.cv2.imwrite"), \
            self.assertRaises(common.ResourceReadError):
            resources.read_resources_from_hud(["wood_stockpile"])
        self.assertGreaterEqual(img2str_mock.call_count, 1)

    def test_cached_value_used_for_optional_failure(self):
        def fake_detect(frame, required_icons):
            return {
                "wood_stockpile": (0, 0, 50, 50),
                "food_stockpile": (50, 0, 50, 50),
            }

        ocr_seq = [
            ("123", {"text": ["123"]}, np.zeros((1, 1), dtype=np.uint8)),
            ("234", {"text": ["234"]}, np.zeros((1, 1), dtype=np.uint8)),
            ("345", {"text": ["345"]}, np.zeros((1, 1), dtype=np.uint8)),
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
            first, _ = resources.read_resources_from_hud(["wood_stockpile"])
            second, _ = resources.read_resources_from_hud(["wood_stockpile"])

        self.assertNotIn("food_stockpile", first)
        self.assertNotIn("food_stockpile", second)


class TestGatherHudStatsSliding(TestCase):
    def setUp(self):
        resources._LAST_RESOURCE_VALUES.clear()
        resources._LAST_RESOURCE_TS.clear()
        resources._RESOURCE_FAILURE_COUNTS.clear()
        resources._LAST_REGION_SPANS.clear()

    def tearDown(self):
        resources._LAST_RESOURCE_VALUES.clear()
        resources._LAST_RESOURCE_TS.clear()
        resources._RESOURCE_FAILURE_COUNTS.clear()
        resources._LAST_REGION_SPANS.clear()

    def test_gather_hud_stats_succeeds_after_sliding(self):
        frame = np.tile(np.arange(120, dtype=np.uint8), (20, 1))
        frame = np.stack([frame] * 3, axis=-1)

        def fake_detect(frame_in, required_icons):
            return {"wood_stockpile": (10, 0, 50, 20)}

        calls = []

        def fake_execute(gray, conf_threshold=None, allow_fallback=True):
            h, w = gray.shape
            mean = gray.mean()
            x = int(round(mean - (w - 1) / 2))
            calls.append((x, w, allow_fallback))
            if mean > 80:
                return "789", {"text": ["789"]}, None
            return "", {"text": [""]}, None

        with patch("script.screen_utils._grab_frame", return_value=frame), \
             patch("script.resources.detect_resource_regions", side_effect=fake_detect), \
             patch("script.resources.preprocess_roi", side_effect=lambda roi: roi[..., 0]), \
             patch.dict("script.resources._LAST_REGION_SPANS", {"wood_stockpile": (0, 120)}, clear=True), \
             patch("script.resources.execute_ocr", side_effect=fake_execute), \
             patch("script.resources.pytesseract.image_to_string", return_value=""):
            res, _ = resources.gather_hud_stats(required_icons=["wood_stockpile"], optional_icons=[])

        self.assertEqual(res["wood_stockpile"], 789)
        self.assertEqual(len(calls), 4)
        self.assertEqual(len({(x, w) for x, w, _ in calls}), len(calls))
        self.assertEqual([a for _, _, a in calls], [True, False, False, False])


class TestResourceOcrRois(TestCase):
    def setUp(self):
        resources._LAST_RESOURCE_VALUES.clear()
        resources._LAST_RESOURCE_TS.clear()
        resources._RESOURCE_FAILURE_COUNTS.clear()

    def tearDown(self):
        resources._LAST_RESOURCE_VALUES.clear()
        resources._LAST_RESOURCE_TS.clear()
        resources._RESOURCE_FAILURE_COUNTS.clear()

    def _build_frame(self):
        icons = [
            "wood_stockpile",
            "food_stockpile",
            "gold_stockpile",
            "stone_stockpile",
            "population_limit",
            "idle_villager",
        ]
        positions = [0, 30, 60, 90, 120, 150]
        icon_w = 5
        icon_h = 5
        frame = np.zeros((20, 200, 3), dtype=np.uint8)
        icon_color = 255
        for pos in positions:
            frame[0:icon_h, pos : pos + icon_w] = icon_color
        values = {icons[i]: i + 1 for i in range(4)}
        for idx in range(4):
            start = positions[idx] + icon_w
            end = positions[idx + 1]
            frame[:, start:end] = values[icons[idx]]
        pop_start = positions[4] + icon_w
        pop_end = positions[5]
        mid = (pop_start + pop_end) // 2
        cur = 5
        cap = 8
        frame[:, pop_start:mid] = cur
        frame[:, mid:pop_end] = cap
        values["population_limit"] = cur
        detected = {
            icons[i]: (positions[i], 0, icon_w, icon_h) for i in range(len(icons))
        }
        regions, _spans, _narrow = resources.compute_resource_rois(
            0,
            200,
            0,
            20,
            [2] * 6,
            [2] * 6,
            [0] * 6,
            999,
            [0] * 6,
            detected=detected,
        )
        return frame, regions, values, cap, icon_color

    def test_ocr_reads_values_from_rois(self):
        frame, regions, values, pop_cap, icon_color = self._build_frame()
        required = [
            "wood_stockpile",
            "food_stockpile",
            "gold_stockpile",
            "stone_stockpile",
            "population_limit",
        ]

        def fake_grab_frame(bbox=None):
            return frame

        def fake_detect(frame_in, required_icons):
            return regions

        def fake_preprocess(roi):
            return roi[..., 0] if roi.ndim == 3 else roi

        def fake_ocr(gray):
            assert gray.shape[1] > 0 and gray.shape[0] > 0
            assert not np.any(gray == icon_color)
            val = int(np.unique(gray)[0])
            return str(val), {"text": [str(val)], "conf": ["95"]}, np.zeros(
                (1, 1), dtype=np.uint8
            )

        def fake_pop_reader(roi):
            gray = roi[..., 0] if roi.ndim == 3 else roi
            assert gray.shape[1] > 0 and gray.shape[0] > 0
            assert not np.any(gray == icon_color)
            mid = gray.shape[1] // 2
            cur = int(np.unique(gray[:, :mid])[0])
            cap = int(np.unique(gray[:, mid:])[0])
            return cur, cap

        with patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
            patch("script.resources.detect_resource_regions", side_effect=fake_detect), \
            patch("script.resources.preprocess_roi", side_effect=fake_preprocess), \
            patch("script.resources._ocr_digits_better", side_effect=fake_ocr), \
            patch(
                "script.resources._read_population_from_roi",
                side_effect=fake_pop_reader,
            ):
            results, pop = resources.read_resources_from_hud(
                required_icons=required, icons_to_read=required
            )

        for name in required:
            self.assertEqual(results[name], values[name])
        self.assertEqual(pop[0], values["population_limit"])
        self.assertEqual(pop[1], pop_cap)
