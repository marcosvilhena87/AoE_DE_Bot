import numpy as np
from unittest import TestCase
from unittest.mock import patch

import script.common as common
import script.resources as resources
import script.resources.reader as reader
from script.resources.panel import ResourcePanelCfg
import script.screen_utils as screen_utils


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

    def test_auto_calibrate_passes_idle_extra_width(self):
        frame = np.zeros((20, 40, 3), dtype=np.uint8)

        cfg_obj = ResourcePanelCfg(
            match_threshold=0.0,
            scales=[1.0],
            pad_left=[0] * 6,
            pad_right=[0] * 6,
            icon_trims=[0] * 6,
            max_widths=[0] * 6,
            min_widths=[0] * 6,
            min_requireds=[0] * 6,
            top_pct=0.0,
            height_pct=1.0,
            idle_roi_extra_width=7,
            min_pop_width=11,
            pop_roi_extra_width=0,
        )

        captured = {}

        def fake_compute(*args):
            captured["args"] = args
            return {}, {}, {}

        with patch.object(screen_utils, "_load_icon_templates", lambda: None), \
            patch(
                "script.resources.panel.calibration._get_resource_panel_cfg",
                return_value=cfg_obj,
            ), \
            patch.object(
                resources.panel.calibration.cv2,
                "cvtColor",
                lambda src, code: np.zeros(src.shape[:2], dtype=np.uint8),
            ), \
            patch.object(
                resources.panel.calibration.cv2,
                "resize",
                lambda img, *a, **k: img,
            ), \
            patch.object(
                resources.panel.calibration.cv2,
                "matchTemplate",
                lambda *a, **k: np.array([[1.0]], dtype=np.float32),
            ), \
            patch.object(
                resources.panel.calibration.cv2,
                "minMaxLoc",
                lambda res: (0.0, float(res.max()), (0, 0), (0, 0)),
            ), \
            patch.dict(
                screen_utils.ICON_TEMPLATES,
                {
                    "wood_stockpile": np.zeros((5, 5), dtype=np.uint8),
                    "food_stockpile": np.zeros((5, 5), dtype=np.uint8),
                },
                clear=True,
            ), \
            patch.object(
                resources.panel.calibration,
                "compute_resource_rois",
                side_effect=fake_compute,
            ):
            resources.panel.calibration._auto_calibrate_from_icons(
                frame, resources.RESOURCE_CACHE
            )

        args = captured.get("args")
        self.assertIsNotNone(args)
        self.assertEqual(args[9], cfg_obj.min_pop_width)
        self.assertEqual(args[10], cfg_obj.idle_roi_extra_width)
        self.assertEqual(args[11], cfg_obj.min_requireds)
