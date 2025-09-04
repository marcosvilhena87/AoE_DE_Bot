import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch
import cv2
import time

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
import script.resources.reader as resources


class TestDetectResourceRegions(TestCase):
    def test_missing_icons_raises_error(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        required = ["wood_stockpile", "food_stockpile"]
        with patch("script.resources.panel.locate_resource_panel", return_value={"wood_stockpile": (0, 0, 10, 10)}), \
             patch.object(common, "HUD_ANCHOR", None), \
             patch("script.resources.panel._auto_calibrate_from_icons", return_value=None), \
            patch.dict(resources.CFG, {"food_stockpile_roi": None}, clear=False):
            with self.assertRaises(common.ResourceReadError):
                resources.detect_resource_regions(frame, required)


class TestPreprocessRoi(TestCase):
    def test_preprocess_returns_grayscale(self):
        roi = np.zeros((5, 5, 3), dtype=np.uint8)
        gray = resources.preprocess_roi(roi)
        self.assertEqual(gray.shape, (5, 5))
        self.assertEqual(gray.dtype, np.uint8)

    def test_blur_kernel_from_config(self):
        roi = np.zeros((40, 5, 3), dtype=np.uint8)
        with patch("script.resources.reader.cv2.medianBlur", return_value=np.zeros((40, 5), dtype=np.uint8)) as blur_mock, \
             patch.dict(resources.CFG, {"ocr_blur_kernel": 5}, clear=False):
            resources.preprocess_roi(roi)
        args, _ = blur_mock.call_args
        self.assertEqual(args[1], 5)

    def test_blur_kernel_zero_disables(self):
        roi = np.zeros((40, 5, 3), dtype=np.uint8)
        with patch("script.resources.reader.cv2.medianBlur") as blur_mock, \
             patch.dict(resources.CFG, {"ocr_blur_kernel": 0}, clear=False):
            gray = resources.preprocess_roi(roi)
        blur_mock.assert_not_called()
        expected = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(gray, expected)

    def test_small_roi_disables_blur(self):
        roi = np.zeros((20, 5, 3), dtype=np.uint8)
        with patch("script.resources.reader.cv2.medianBlur") as blur_mock, \
             patch.dict(resources.CFG, {"ocr_blur_kernel": 5}, clear=False):
            gray = resources.preprocess_roi(roi)
        blur_mock.assert_not_called()
        expected = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(gray, expected)


class TestExecuteOcr(TestCase):
    def test_execute_ocr_fallback(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        with patch("script.resources.ocr.masks._ocr_digits_better", return_value=("", {}, None)), \
             patch("script.resources.reader.pytesseract.image_to_string", return_value="456"):
            digits, data, mask, low_conf = resources.execute_ocr(
                gray, resource="wood_stockpile"
            )
        self.assertEqual(digits, "456")
        self.assertTrue(low_conf)
        self.assertEqual(data["text"], ["456"])
        self.assertIsNone(mask)

    def test_execute_ocr_warns_low_confidence(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        data = {"text": ["123"], "conf": ["10", "20", "30"]}
        with patch("script.resources.ocr.masks._ocr_digits_better", return_value=("123", data, None)), \
             patch("script.resources.reader.pytesseract.image_to_string", return_value="") as img2str_mock:
            digits, data_out, _, low_conf = resources.execute_ocr(
                gray, conf_threshold=60, resource="wood_stockpile"
            )
        self.assertEqual(digits, "123")
        self.assertTrue(low_conf)
        img2str_mock.assert_not_called()

    def test_execute_ocr_warns_low_mean_confidence(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        data = {"text": ["1", "2"], "conf": ["80", "20"]}
        with patch("script.resources.ocr.masks._ocr_digits_better", return_value=("12", data, None)), \
             patch("script.resources.reader.pytesseract.image_to_string", return_value="") as img2str_mock, \
             patch.dict(resources.CFG, {"ocr_conf_decay": 1.0}, clear=False):
            digits, data_out, _, low_conf = resources.execute_ocr(
                gray, conf_threshold=60, resource="wood_stockpile"
            )
        self.assertEqual(digits, "12")
        self.assertTrue(low_conf)
        img2str_mock.assert_not_called()

    def test_execute_ocr_accepts_mixed_confidence_digits(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        data = {"text": ["789"], "conf": ["90", "10", "90"]}
        with patch("script.resources.ocr.masks._ocr_digits_better", return_value=("789", data, None)), \
             patch("script.resources.reader.pytesseract.image_to_string", return_value="") as img2str_mock, \
             patch.dict(resources.CFG, {"ocr_conf_decay": 1.0}, clear=False):
            digits, data_out, _, low_conf = resources.execute_ocr(
                gray, conf_threshold=60, resource="wood_stockpile"
            )
        self.assertEqual(digits, "789")
        self.assertFalse(low_conf)
        img2str_mock.assert_not_called()

    def test_execute_ocr_accepts_low_conf_single_digit(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        data = {"text": ["0"], "conf": ["10"]}
        with patch("script.resources.ocr.masks._ocr_digits_better", return_value=("0", data, None)), \
             patch("script.resources.reader.pytesseract.image_to_string", return_value="") as img2str_mock:
            digits, data_out, _, low_conf = resources.execute_ocr(
                gray, conf_threshold=60, resource="wood_stockpile"
            )
        self.assertEqual(digits, "0")
        self.assertTrue(low_conf)
        img2str_mock.assert_not_called()

    def test_execute_ocr_does_not_rerun_without_preprocessing(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        data1 = {"text": ["123"], "conf": ["0", "0", "0"]}
        data2 = {"text": ["789"], "conf": ["80", "90", "100"]}
        with patch(
            "script.resources.ocr.masks._ocr_digits_better",
            side_effect=[("123", data1, None), ("789", data2, None)],
        ) as ocr_mock, patch("script.resources.reader.pytesseract.image_to_string") as img2str_mock, patch.dict(resources.CFG, {"allow_zero_confidence_digits": False}, clear=False):
            digits, _, _, low_conf = resources.execute_ocr(
                gray, conf_threshold=60, resource="wood_stockpile"
            )
        self.assertEqual(digits, "123")
        self.assertTrue(low_conf)
        img2str_mock.assert_not_called()
        ocr_mock.assert_called_once()

    def test_execute_ocr_zero_variance_shortcut(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        with patch(
            "script.resources.ocr.masks._ocr_digits_better",
            return_value=("0", {"zero_variance": True}, None),
        ), patch("script.resources.reader.pytesseract.image_to_string") as img2str_mock, patch(
            "script.resources.reader.logger.warning"
        ) as warn_mock:
            digits, data_out, mask, low_conf = resources.execute_ocr(
                gray, conf_threshold=60, resource="wood_stockpile"
            )
        self.assertEqual(digits, "0")
        self.assertFalse(low_conf)
        self.assertFalse(data_out.get("low_conf_single"))
        self.assertFalse(data_out.get("low_conf_multi"))
        self.assertIsNone(mask)
        img2str_mock.assert_not_called()
        warn_mock.assert_not_called()

    def test_execute_ocr_accepts_after_threshold_decay(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        side_effect = [
            ("12", {"text": ["12"], "conf": ["40", "40"]}, None)
            for _ in range(4)
        ]
        with patch("script.resources.ocr.masks._ocr_digits_better", side_effect=side_effect) as ocr_mock, \
             patch("script.resources.reader.pytesseract.image_to_string", return_value="") as img2str_mock, \
             patch.dict(resources.CFG, {"ocr_conf_min": 30, "ocr_conf_decay": 0.5}, clear=False):
            digits, data_out, _, low_conf = resources.execute_ocr(
                gray, conf_threshold=60, resource="wood_stockpile"
            )
        self.assertEqual(digits, "12")
        self.assertFalse(low_conf)
        self.assertNotIn("low_conf_multi", data_out)
        img2str_mock.assert_not_called()
        ocr_mock.assert_called_once()

    def test_execute_ocr_ignores_zero_confidences_when_others_high(self):
        gray = np.zeros((5, 5), dtype=np.uint8)
        data = {"text": ["foo", "", "12"], "conf": [-1, "0", "95"]}
        with patch(
            "script.resources.ocr.masks._ocr_digits_better",
            return_value=("12", data, None),
        ), patch(
            "script.resources.reader.pytesseract.image_to_string", return_value=""
        ) as img2str_mock:
            digits, _, _, low_conf = resources.execute_ocr(
                gray, conf_threshold=60, resource="wood_stockpile"
            )
        self.assertEqual(digits, "12")
        self.assertFalse(low_conf)
        img2str_mock.assert_not_called()


class TestHandleOcrFailure(TestCase):
    def test_handle_ocr_failure_raises(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        regions = {"wood_stockpile": (0, 0, 10, 10)}
        results = {"wood_stockpile": None}
        with patch("script.resources.reader.cv2.imwrite"), \
             patch("script.resources.ocr.executor.logger.error") as err_mock, \
             patch("script.resources.reader.pytesseract.pytesseract.tesseract_cmd", "/usr/bin/true"):
            resources.handle_ocr_failure(frame, regions, results, ["wood_stockpile"])
        err_mock.assert_called()

    def test_handle_ocr_failure_noop_when_success(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        regions = {"wood_stockpile": (0, 0, 10, 10)}
        results = {"wood_stockpile": 1}
        with patch("script.resources.reader.cv2.imwrite") as imwrite_mock:
            resources.handle_ocr_failure(frame, regions, results, ["wood_stockpile"])
        imwrite_mock.assert_not_called()

    def test_handle_ocr_failure_low_confidence_fallback(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        regions = {"wood_stockpile": (0, 0, 10, 10)}
        results = {"wood_stockpile": None}
        cache = resources.ResourceCache()
        cache.last_resource_values["wood_stockpile"] = 99
        with patch("script.resources.reader.cv2.imwrite"), \
             patch("script.resources.ocr.executor.logger.warning") as warn_mock:
            resources.handle_ocr_failure(
                frame,
                regions,
                results,
                ["wood_stockpile"],
                cache_obj=cache,
                retry_limit=2,
                low_confidence={"wood_stockpile"},
            )
            self.assertIsNone(results["wood_stockpile"])
            self.assertEqual(cache.resource_failure_counts["wood_stockpile"], 1)
            resources.handle_ocr_failure(
                frame,
                regions,
                results,
                ["wood_stockpile"],
                cache_obj=cache,
                retry_limit=2,
                low_confidence={"wood_stockpile"},
            )
        self.assertEqual(results["wood_stockpile"], 99)
        self.assertEqual(warn_mock.call_count, 1)


class TestReadResourcesLowConfFallback(TestCase):
    def test_low_confidence_uses_cached_value(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        cache = resources.ResourceCache()
        cache.last_resource_values["wood_stockpile"] = 77
        roi_info = (
            0,
            0,
            1,
            1,
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
            0,
            0,
        )
        with patch(
            "script.resources.reader.core._ocr_resource",
            return_value=("123", {}, None, True),
        ), patch(
            "script.resources.reader.core._retry_ocr",
            return_value=(
                "123",
                {},
                None,
                np.zeros((1, 1, 3), dtype=np.uint8),
                np.zeros((1, 1), dtype=np.uint8),
                0,
                0,
                1,
                1,
                True,
            ),
        ), patch.dict(
            resources.CFG,
            {
                "treat_low_conf_as_failure": True,
                "wood_stockpile_low_conf_fallback": True,
            },
            clear=False,
        ):
            value, _, _, _, _ = resources.core._process_resource(
                frame,
                "wood_stockpile",
                roi_info,
                cache_obj=cache,
                res_conf_threshold=60,
                max_cache_age=None,
                low_conf_counts={},
            )
        self.assertEqual(value, 77)


class TestIdleVillagerPopulationMismatch(TestCase):
    def test_idle_count_exceeding_population_falls_back(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        cache = resources.ResourceCache()
        results = {"idle_villager": 5}
        regions = {
            "idle_villager": (0, 0, 1, 1),
            "population_limit": (1, 0, 1, 1),
        }
        with patch(
            "script.resources.reader.core._extract_population",
            return_value=(3, 10),
        ), patch(
            "script.resources.reader.core.logger.warning"
        ) as warn_mock:
            _ = resources.core._post_process_population(
                frame,
                regions,
                ["idle_villager", "population_limit"],
                {"idle_villager", "population_limit"},
                results,
                cache_obj=cache,
                max_cache_age=None,
                conf_threshold=60,
                low_conf_counts={},
                low_confidence=set(),
                cache_hits=set(),
                prev_idle_val=None,
                prev_idle_ts=None,
            )
        self.assertIsNone(results["idle_villager"])
        warn_mock.assert_called()


class TestValidateStartingResources(TestCase):
    def test_deviation_raises_error(self):
        with self.assertRaises(ValueError):
            resources.validate_starting_resources(
                {"wood_stockpile": 50},
                {"wood_stockpile": 80},
                tolerance=10,
                raise_on_error=True,
            )

    def test_logs_warning_without_raise(self):
        with patch("script.resources.reader.logger.warning") as warn_mock:
            resources.validate_starting_resources(
                {"wood_stockpile": 50},
                {"wood_stockpile": 80},
                tolerance=10,
                raise_on_error=False,
            )
            warn_mock.assert_called_once()

    def test_logs_low_confidence_warning(self):
        with patch("script.resources.reader.logger.warning") as warn_mock:
            resources.RESOURCE_CACHE.last_low_confidence = {"wood_stockpile"}
            resources.RESOURCE_CACHE.last_no_digits = set()
            resources.validate_starting_resources(
                {"wood_stockpile": None},
                {"wood_stockpile": 80},
                tolerance=10,
                raise_on_error=False,
            )
            warn_mock.assert_called_once_with(
                "Low-confidence OCR for 'wood_stockpile'"
            )
        resources.RESOURCE_CACHE.last_low_confidence = set()
        resources.RESOURCE_CACHE.last_no_digits = set()

    def test_logs_missing_reading_warning(self):
        with patch("script.resources.reader.logger.warning") as warn_mock:
            resources.RESOURCE_CACHE.last_low_confidence = set()
            resources.RESOURCE_CACHE.last_no_digits = {"wood_stockpile"}
            resources.validate_starting_resources(
                {"wood_stockpile": None},
                {"wood_stockpile": 80},
                tolerance=10,
                raise_on_error=False,
            )
            warn_mock.assert_called_once_with(
                "Missing OCR reading for 'wood_stockpile'"
            )
        resources.RESOURCE_CACHE.last_low_confidence = set()
        resources.RESOURCE_CACHE.last_no_digits = set()

    def test_deviation_saves_roi_image(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        rois = {"wood_stockpile": (0, 0, 5, 5)}
        ts = 1.234
        expected_ts = int(ts * 1000)
        expected_path = resources.ROOT / "debug" / f"resource_roi_wood_stockpile_{expected_ts}.png"
        with patch("script.resources.reader.cv2.imwrite") as imwrite_mock, \
             patch("script.resources.reader.time.time", return_value=ts), \
             patch("script.resources.reader.logger.error"):
            with self.assertRaises(ValueError) as ctx:
                resources.validate_starting_resources(
                    {"wood_stockpile": 50},
                    {"wood_stockpile": 80},
                    tolerance=10,
                    raise_on_error=True,
                    frame=frame,
                    rois=rois,
                )
        self.assertEqual(imwrite_mock.call_count, 3)
        self.assertIn(str(expected_path), str(ctx.exception))

    def test_aggregates_all_discrepancies_and_saves_each_roi(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        rois = {
            "wood_stockpile": (0, 0, 5, 5),
            "food_stockpile": (5, 5, 5, 5),
        }
        with patch("script.resources.reader.cv2.imwrite") as imwrite_mock, \
             patch("script.resources.reader.logger.error") as error_mock, \
             patch("script.resources.reader.time.time", side_effect=[1.0, 2.0]):
            with self.assertRaises(ValueError) as ctx:
                resources.validate_starting_resources(
                    {"wood_stockpile": 50, "food_stockpile": 30},
                    {
                        "wood_stockpile": 80,
                        "food_stockpile": 60,
                        "stone_stockpile": 40,
                    },
                    tolerance=10,
                    raise_on_error=True,
                    frame=frame,
                    rois=rois,
                )
        self.assertEqual(imwrite_mock.call_count, 6)
        self.assertEqual(error_mock.call_count, 3)
        msg = str(ctx.exception)
        self.assertIn("wood_stockpile reading 50", msg)
        self.assertIn("food_stockpile reading 30", msg)
        self.assertIn("Missing OCR reading for 'stone_stockpile'", msg)

    def test_per_resource_tolerance_overrides_default(self):
        with patch("script.resources.reader.logger.warning") as warn_mock:
            resources.validate_starting_resources(
                {"wood_stockpile": 50, "food_stockpile": 70},
                {"wood_stockpile": 80, "food_stockpile": 90},
                tolerance=10,
                tolerances={"wood_stockpile": 40},
                raise_on_error=False,
            )
            warn_mock.assert_called_once_with(
                "food_stockpile reading 70 deviates from expected 90 (±10)"
            )

    def test_low_confidence_saves_debug_images(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        rois = {"wood_stockpile": (0, 0, 5, 5)}
        resources.RESOURCE_CACHE.last_low_confidence = {"wood_stockpile"}
        ts = 1.111
        with patch("script.resources.reader.cv2.imwrite") as imwrite_mock, \
             patch("script.resources.reader.time.time", return_value=ts), \
             patch("script.resources.reader.logger.warning") as warn_mock:
            resources.validate_starting_resources(
                {"wood_stockpile": 80},
                {"wood_stockpile": 80},
                tolerance=10,
                raise_on_error=False,
                frame=frame,
                rois=rois,
            )
        self.assertEqual(imwrite_mock.call_count, 3)
        self.assertGreaterEqual(warn_mock.call_count, 1)
        resources.RESOURCE_CACHE.last_low_confidence = set()


class TestCacheDiscrepancy(TestCase):
    def test_mismatched_cache_ignored_and_validation_succeeds(self):
        cache = resources.ResourceCache()
        cache.last_resource_values["wood_stockpile"] = 900
        cache.last_resource_ts["wood_stockpile"] = time.time()

        with patch.dict(
            resources.CFG,
            {
                "wood_stockpile_low_conf_fallback": True,
                "resource_cache_tolerance": 50,
                "starting_resources": {"wood_stockpile": 80},
            },
            clear=False,
        ):
            value, cache_hit, low_conf, no_digit = resources.core._handle_cache_and_fallback(
                "wood_stockpile",
                "80",
                True,
                {"text": ["80"]},
                np.zeros((1, 1), dtype=np.uint8),
                None,
                0,
                cache_obj=cache,
                max_cache_age=None,
                low_conf_counts={},
            )

        self.assertEqual(value, 80)
        self.assertFalse(cache_hit)
        self.assertFalse(low_conf)
        failing = resources.validate_starting_resources(
            {"wood_stockpile": value}, {"wood_stockpile": 80}, raise_on_error=False
        )
        self.assertEqual(failing, set())

    def test_large_difference_discards_cached_food_value(self):
        cache = resources.ResourceCache()
        cache.last_resource_values["food_stockpile"] = 441
        cache.last_resource_ts["food_stockpile"] = time.time()

        with patch.dict(
            resources.CFG,
            {
                "food_stockpile_low_conf_fallback": True,
                "resource_cache_tolerance": 100,
            },
            clear=False,
        ):
            value, cache_hit, low_conf, no_digit = resources.core._handle_cache_and_fallback(
                "food_stockpile",
                "140",
                True,
                {"text": ["140"]},
                np.zeros((1, 1), dtype=np.uint8),
                None,
                0,
                cache_obj=cache,
                max_cache_age=None,
                low_conf_counts={},
            )

        self.assertEqual(value, 140)
        self.assertFalse(cache_hit)
        self.assertTrue(low_conf)
        self.assertEqual(cache.last_resource_values["food_stockpile"], 140)
