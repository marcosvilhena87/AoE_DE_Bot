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
try:  # pragma: no cover - used for environments without OpenCV
    import cv2  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - fallback stub
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
            bitwise_or=lambda a, b: a,
            morphologyEx=lambda src, op, kernel, iterations=1: src,
            threshold=lambda src, *a, **k: (None, src),
            rectangle=lambda img, pt1, pt2, color, thickness: img,
            bilateralFilter=lambda src, d, sigmaColor, sigmaSpace: src,
            adaptiveThreshold=lambda src, maxValue, adaptiveMethod, thresholdType, blockSize, C: src,
            dilate=lambda src, kernel, iterations=1: src,
            equalizeHist=lambda src: src,
            inRange=lambda src, lower, upper: np.zeros(src.shape[:2], dtype=np.uint8),
            countNonZero=lambda src: int(np.count_nonzero(src)),
            ADAPTIVE_THRESH_GAUSSIAN_C=0,
            ADAPTIVE_THRESH_MEAN_C=0,
            IMREAD_GRAYSCALE=0,
            COLOR_BGR2GRAY=0,
            COLOR_GRAY2BGR=0,
            COLOR_BGR2HSV=0,
            INTER_LINEAR=0,
            THRESH_BINARY=0,
            THRESH_OTSU=0,
            MORPH_CLOSE=0,
            TM_CCOEFF_NORMED=0,
        ),
    )

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.common as common
import script.resources as resources


class TestResourceOcrFailure(TestCase):
    def setUp(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()
        resources._LAST_REGION_SPANS.clear()

    def tearDown(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()
        resources._LAST_REGION_SPANS.clear()
    def test_read_resources_fallback(self):
        def fake_ocr(gray):
            return "", {"text": [""]}, np.zeros((1, 1), dtype=np.uint8)
        frame = np.zeros((600, 600, 3), dtype=np.uint8)

        with patch(
            "script.resources.detect_resource_regions",
            return_value={
                "wood_stockpile": (0, 0, 50, 50),
                "food_stockpile": (50, 0, 50, 50),
                "gold_stockpile": (100, 0, 50, 50),
                "stone_stockpile": (150, 0, 50, 50),
                "population_limit": (200, 0, 50, 50),
                "idle_villager": (250, 0, 50, 50),
            },
        ), patch("script.resources._ocr_digits_better", side_effect=fake_ocr), patch(
            "script.resources.pytesseract.image_to_data",
            return_value={"text": [""], "conf": ["0"]},
        ), patch(
            "script.resources.pytesseract.image_to_string", return_value="123"
        ), patch("script.resources._read_population_from_roi", return_value=(0, 0)):
            icons = resources.RESOURCE_ICON_ORDER[:-1]
            result, _ = resources._read_resources(
                frame,
                icons,
                icons,
            )
            self.assertEqual(result["wood_stockpile"], 123)

    def test_optional_icon_failure_does_not_raise(self):
        def fake_grab_frame(bbox=None):
            if bbox:
                return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
            return np.zeros((600, 600, 3), dtype=np.uint8)

        def fake_detect(frame, required_icons, cache=None):
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

    def test_low_confidence_triggers_retry(self):
        def fake_grab_frame(bbox=None):
            if bbox:
                return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
            return np.zeros((600, 600, 3), dtype=np.uint8)

        def fake_detect(frame, required_icons, cache=None):
            return {"wood_stockpile": (0, 0, 50, 50)}

        def fake_ocr(gray):
            data = {"text": ["123"], "conf": ["10", "20", "30"]}
            return "123", data, np.zeros((1, 1), dtype=np.uint8)

        with patch("script.resources.detect_resource_regions", side_effect=fake_detect), \
            patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
            patch("script.resources._ocr_digits_better", side_effect=fake_ocr) as ocr_mock, \
            patch("script.resources.pytesseract.image_to_string", return_value="") as img2str_mock, \
            patch("script.resources.cv2.imwrite"):
            result, _ = resources.read_resources_from_hud(["wood_stockpile"])

        self.assertEqual(result["wood_stockpile"], 123)
        self.assertGreaterEqual(ocr_mock.call_count, 2)
        img2str_mock.assert_called()

    def test_low_confidence_single_digit_triggers_retry(self):
        def fake_grab_frame(bbox=None):
            if bbox:
                return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
            return np.zeros((600, 600, 3), dtype=np.uint8)

        def fake_detect(frame, required_icons, cache=None):
            return {"wood_stockpile": (0, 0, 50, 50)}

        ocr_seq = [
            ("7", {"text": ["7"], "conf": ["10"]}, np.zeros((1, 1), dtype=np.uint8)),
            ("", {"text": [""], "conf": [""]}, np.zeros((1, 1), dtype=np.uint8)),
            ("", {"text": [""], "conf": [""]}, np.zeros((1, 1), dtype=np.uint8)),
        ]

        def fake_ocr(gray):
            return ocr_seq.pop(0) if ocr_seq else ("", {"text": [""], "conf": [""]}, np.zeros((1, 1), dtype=np.uint8))

        with patch("script.resources.detect_resource_regions", side_effect=fake_detect), \
            patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
            patch("script.resources._ocr_digits_better", side_effect=fake_ocr) as ocr_mock, \
            patch("script.resources.pytesseract.image_to_string", return_value="") as img2str_mock, \
            patch("script.resources.cv2.imwrite"):
            result, _ = resources.read_resources_from_hud(["wood_stockpile"])

        self.assertEqual(result["wood_stockpile"], 7)
        self.assertGreaterEqual(ocr_mock.call_count, 2)
        self.assertGreaterEqual(img2str_mock.call_count, 1)

    def test_zero_confidence_triggers_failure(self):
        def fake_grab_frame(bbox=None):
            if bbox:
                return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
            return np.zeros((600, 600, 3), dtype=np.uint8)

        def fake_detect(frame, required_icons, cache=None):
            return {"wood_stockpile": (0, 0, 50, 50)}

        def fake_ocr(gray):
            data = {"text": ["7"], "conf": ["0"]}
            return "7", data, np.zeros((1, 1), dtype=np.uint8)

        with patch("script.resources.detect_resource_regions", side_effect=fake_detect), \
            patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
            patch("script.resources._ocr_digits_better", side_effect=fake_ocr), \
            patch("script.resources.pytesseract.image_to_string", return_value="") as img2str_mock, \
            patch("script.resources.cv2.imwrite"):
            result, _ = resources.read_resources_from_hud(["wood_stockpile"])
        self.assertEqual(result["wood_stockpile"], 7)
        self.assertGreaterEqual(img2str_mock.call_count, 1)

    def test_cached_value_used_for_optional_failure(self):
        def fake_detect(frame, required_icons, cache=None):
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

    def test_zero_roi_returns_zero(self):
        gray = np.full((20, 20), 128, dtype=np.uint8)
        with patch(
            "script.resources.pytesseract.image_to_data",
            return_value={"text": [""], "conf": ["-1"]},
        ):
            digits, data, _ = resources._ocr_digits_better(gray)
        self.assertEqual(digits, "0")
        self.assertTrue(data.get("zero_variance"))

    def test_gold_and_stone_zero_digits_return_zero(self):
        def make_gold_roi():
            roi = np.full((10, 10), 210, dtype=np.uint8)
            roi[2:-2, 2] = 200
            roi[2:-2, -3] = 200
            roi[2, 2:-2] = 200
            roi[-3, 2:-2] = 200
            return roi

        def make_stone_roi():
            roi = np.full((10, 10), 180, dtype=np.uint8)
            roi[2:-2, 2] = 170
            roi[2:-2, -3] = 170
            roi[2, 2:-2] = 170
            roi[-3, 2:-2] = 170
            return roi

        with patch(
            "script.resources.pytesseract.image_to_data",
            return_value={"text": [""], "conf": ["-1"]},
        ), patch.dict(resources.CFG, {"ocr_zero_variance": 50}, clear=False):
            gold, _, _ = resources._ocr_digits_better(make_gold_roi())
            stone, _, _ = resources._ocr_digits_better(make_stone_roi())
        self.assertEqual(gold, "0")
        self.assertEqual(stone, "0")

    def test_narrow_roi_failure_includes_note(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        regions = {"wood_stockpile": (0, 0, 10, 10)}
        results = {"wood_stockpile": None}
        with patch("script.resources.cv2.imwrite"), \
            patch("script.resources.logger.error") as err_mock, \
            patch("script.resources.pytesseract.pytesseract.tesseract_cmd", "/usr/bin/true"), \
            patch.object(resources, "_NARROW_ROIS", {"wood_stockpile"}):
            with self.assertRaises(common.ResourceReadError) as ctx:
                resources.handle_ocr_failure(
                    frame, regions, results, ["wood_stockpile"]
                )
        self.assertIn("narrow ROI span", err_mock.call_args[0][1])
        self.assertIn("narrow ROI span", str(ctx.exception))

    def test_overlapping_rois_are_trimmed(self):
        regions = {
            "wood_stockpile": (0, 0, 50, 10),
            "food_stockpile": (40, 0, 50, 10),
        }
        regions = resources._remove_overlaps(
            regions, ["wood_stockpile", "food_stockpile"]
        )

        self.assertEqual(regions["wood_stockpile"], (0, 0, 40, 10))
        self.assertEqual(regions["food_stockpile"], (40, 0, 50, 10))
        self.assertLessEqual(
            regions["wood_stockpile"][0] + regions["wood_stockpile"][2],
            regions["food_stockpile"][0],
        )


class TestGatherHudStatsSliding(TestCase):
    def setUp(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()
        resources._LAST_REGION_SPANS.clear()

    def tearDown(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()
        resources._LAST_REGION_SPANS.clear()

    def test_gather_hud_stats_succeeds_after_sliding(self):
        frame = np.tile(np.arange(120, dtype=np.uint8), (20, 1))
        frame = np.stack([frame] * 3, axis=-1)

        def fake_detect(frame_in, required_icons, cache=None):
            return {"wood_stockpile": (10, 0, 50, 20)}

        calls = []

        def fake_execute(
            gray, conf_threshold=None, allow_fallback=True, roi=None, resource=None
        ):
            h, w = gray.shape
            mean = gray.mean()
            x = int(round(mean - (w - 1) / 2))
            calls.append((x, w, allow_fallback))
            if mean > 80:
                return "789", {"text": ["789"]}, None, False
            return "", {"text": [""]}, None, False

        with patch(
            "script.resources.detect_resource_regions", side_effect=fake_detect
        ), patch(
            "script.resources.preprocess_roi", side_effect=lambda roi: roi[..., 0]
        ), patch.dict(
            "script.resources._LAST_REGION_SPANS",
            {"wood_stockpile": (0, 120)},
            clear=True,
        ), patch(
            "script.resources.execute_ocr", side_effect=fake_execute
        ), patch(
            "script.resources.pytesseract.image_to_string", return_value=""
        ):
            res, _ = resources._read_resources(
                frame,
                ["wood_stockpile"],
                ["wood_stockpile"],
            )

        self.assertEqual(res["wood_stockpile"], 789)
        self.assertEqual(len(calls), 5)
        self.assertEqual(len({(x, w) for x, w, _ in calls}), len(calls))
        self.assertEqual([a for _, _, a in calls], [True, True, False, False, False])


class TestResourceOcrRois(TestCase):
    def setUp(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()

    def tearDown(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()

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

        def fake_detect(frame_in, required_icons, cache=None):
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

        with patch(
            "script.resources.detect_resource_regions", side_effect=fake_detect
        ), patch(
            "script.resources.preprocess_roi", side_effect=fake_preprocess
        ), patch(
            "script.resources._ocr_digits_better", side_effect=fake_ocr
        ), patch(
            "script.resources._read_population_from_roi",
            side_effect=fake_pop_reader,
        ):
            results, pop = resources._read_resources(frame, required, required)

        for name in required:
            self.assertEqual(results[name], values[name])
        self.assertEqual(pop[0], values["population_limit"])
        self.assertEqual(pop[1], pop_cap)
