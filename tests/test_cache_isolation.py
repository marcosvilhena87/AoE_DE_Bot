import numpy as np
from unittest import TestCase
from unittest.mock import patch

from script.resources.reader.core import _read_resources, ResourceCache


class TestCacheIsolation(TestCase):
    def test_last_flags_isolated_between_reads(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        cache1 = ResourceCache()
        cache2 = ResourceCache()

        def handle_side_effect(
            name,
            digits,
            low_conf,
            data,
            roi,
            mask,
            failure_count,
            *,
            cache_obj,
            max_cache_age,
            low_conf_counts,
        ):
            if cache_obj is cache1:
                return None, False, True, False
            else:
                return None, False, False, True

        roi_ret = (
            0,
            0,
            10,
            10,
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.zeros((10, 10), dtype=np.uint8),
            0,
            0,
        )

        def dummy_retry(
            frame,
            name,
            digits,
            data,
            mask,
            roi,
            gray,
            x,
            y,
            w,
            h,
            top_crop,
            failure_count,
            res_conf_threshold,
            low_conf,
        ):
            return digits, data, mask, roi, gray, x, y, w, h, low_conf

        with patch(
            "script.resources.reader.core.detect_resource_regions",
            return_value={"wood_stockpile": (0, 0, 10, 10)},
        ), patch(
            "script.resources.reader.core.prepare_roi", return_value=roi_ret
        ), patch(
            "script.resources.reader.core._ocr_resource",
            return_value=("10", {}, None, False),
        ), patch(
            "script.resources.reader.core._retry_ocr", side_effect=dummy_retry
        ), patch(
            "script.resources.reader.core._handle_cache_and_fallback",
            side_effect=handle_side_effect,
        ), patch(
            "script.resources.reader.core.handle_ocr_failure"
        ), patch(
            "script.resources.reader.core.cv2.imwrite"
        ):
            _read_resources(frame, ["wood_stockpile"], ["wood_stockpile"], cache1)
            _read_resources(frame, ["wood_stockpile"], ["wood_stockpile"], cache2)

        self.assertEqual(cache1.last_low_confidence, {"wood_stockpile"})
        self.assertEqual(cache1.last_no_digits, set())
        self.assertEqual(cache2.last_low_confidence, set())
        self.assertEqual(cache2.last_no_digits, {"wood_stockpile"})
