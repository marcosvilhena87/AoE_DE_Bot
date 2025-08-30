import numpy as np
from unittest import TestCase
from unittest.mock import patch

import script.common as common
import script.resources as resources
import script.resources.reader as reader


class TestCalibration(TestCase):
    def setUp(self):
        reader.RESOURCE_CACHE.last_resource_values.clear()
        reader.RESOURCE_CACHE.last_resource_ts.clear()
        reader.RESOURCE_CACHE.resource_failure_counts.clear()
        reader._LAST_REGION_SPANS.clear()

    def tearDown(self):
        reader.RESOURCE_CACHE.last_resource_values.clear()
        reader.RESOURCE_CACHE.last_resource_ts.clear()
        reader.RESOURCE_CACHE.resource_failure_counts.clear()
        reader._LAST_REGION_SPANS.clear()

    def test_misaligned_roi_triggers_auto_calibration(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        block = np.arange(10 * 20 * 3, dtype=np.uint8).reshape(10, 20, 3)
        frame[5:15, 50:70] = block
        misaligned = {"wood_stockpile": (0, 0, 10, 10)}
        calibrated = {"wood_stockpile": (50, 5, 20, 10)}

        def fake_auto_calibrate(_frame, _cache):
            reader._LAST_REGION_SPANS["wood_stockpile"] = (
                calibrated["wood_stockpile"][0],
                calibrated["wood_stockpile"][0] + calibrated["wood_stockpile"][2],
            )
            return calibrated

        with patch(
            "script.resources.panel.calibration._auto_calibrate_from_icons",
            side_effect=fake_auto_calibrate,
        ) as mock_calib:
            regions = resources._recalibrate_low_variance(
                frame, misaligned, ["wood_stockpile"], resources.RESOURCE_CACHE
            )

        self.assertTrue(mock_calib.called)
        self.assertEqual(regions["wood_stockpile"], calibrated["wood_stockpile"])
        span = reader._LAST_REGION_SPANS.get("wood_stockpile")
        self.assertEqual(span, (50, 70))
        self.assertGreater(span[1], span[0])
