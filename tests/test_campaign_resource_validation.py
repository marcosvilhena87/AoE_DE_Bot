import os
import sys
import types
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

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
import campaign


class TestCampaignResourceValidation(TestCase):
    def setUp(self):
        self.info = types.SimpleNamespace(
            starting_resources={
                "wood_stockpile": 100,
                "food_stockpile": 0,
                "gold_stockpile": 0,
                "stone_stockpile": 0,
            },
            starting_villagers=3,
            objective_villagers=5,
        )

    def _run_main(
        self, res_sequence, bounds=None, spans=None, low_conf=None, no_digits=None
    ):
        res_list = list(res_sequence)

        if bounds is None:
            bounds = {"wood_stockpile": (0, 0, 5, 5)}
        if spans is None:
            spans = {k: (v[0], v[0] + v[2]) for k, v in bounds.items()}

        campaign.resources.core._LAST_LOW_CONFIDENCE.clear()
        campaign.resources.core._LAST_NO_DIGITS.clear()
        low_conf = set() if low_conf is None else set(low_conf)
        no_digits = set() if no_digits is None else set(no_digits)

        def gh_side_effect(*args, **kwargs):
            campaign.resources._LAST_REGION_BOUNDS = bounds.copy()
            campaign.resources._LAST_REGION_SPANS = spans.copy()
            campaign.resources.core._LAST_LOW_CONFIDENCE = low_conf.copy()
            campaign.resources.core._LAST_NO_DIGITS = no_digits.copy()
            return res_list.pop(0), (0, 0)

        logger_mock = MagicMock()

        dummy_module = types.SimpleNamespace(run_mission=lambda *a, **k: None)

        with tempfile.TemporaryDirectory() as tmpdir, \
            patch("campaign.parse_scenario_info", return_value=self.info), \
            patch(
                "campaign.argparse.ArgumentParser.parse_args",
                return_value=types.SimpleNamespace(scenario="dummy"),
            ), \
            patch("campaign.screen_utils.init_sct"), \
            patch("campaign.screen_utils.teardown_sct"), \
            patch("campaign.hud.wait_hud", return_value=({}, "asset")), \
            patch("campaign.resources.gather_hud_stats", side_effect=gh_side_effect), \
            patch(
                "campaign.screen_utils._grab_frame",
                return_value=np.zeros((10, 10, 3), dtype=np.uint8),
            ), \
            patch("campaign.logging.getLogger", return_value=logger_mock), \
            patch.object(campaign.resources, "ROOT", Path(tmpdir)), \
            patch("campaign.resources.cv2.imwrite"), \
            patch("importlib.import_module", return_value=dummy_module):
            campaign.resources._NARROW_ROI_DEFICITS.clear()
            campaign.main()

        return logger_mock

    def test_retry_warns_and_continues_on_near_match(self):
        res_seq = [
            {"wood_stockpile": 50},
            {"wood_stockpile": 88},
            {"wood_stockpile": 88},
        ]
        logger_mock = self._run_main(res_seq)
        self.assertGreaterEqual(logger_mock.warning.call_count, 1)

    def test_retry_exits_when_far_off(self):
        res_seq = [
            {"wood_stockpile": 50},
            {"wood_stockpile": 40},
            {"wood_stockpile": 30},
        ]
        with self.assertRaises(SystemExit):
            self._run_main(res_seq)

    def test_only_failing_resources_are_narrowed(self):
        self.info.starting_resources = {
            "wood_stockpile": 100,
            "gold_stockpile": 100,
        }
        bounds = {
            "wood_stockpile": (0, 0, 5, 5),
            "gold_stockpile": (10, 0, 5, 5),
        }
        spans = {
            "wood_stockpile": (0, 5),
            "gold_stockpile": (10, 15),
        }
        res_seq = [
            {"wood_stockpile": 50, "gold_stockpile": 100},
            {"wood_stockpile": 100, "gold_stockpile": 100},
        ]
        self._run_main(
            res_seq,
            bounds=bounds,
            spans=spans,
            low_conf={"wood_stockpile"},
        )
        self.assertEqual(
            campaign.resources._NARROW_ROI_DEFICITS.get("wood_stockpile"),
            2,
        )
        self.assertIsNone(
            campaign.resources._NARROW_ROI_DEFICITS.get("gold_stockpile")
        )
        self.assertEqual(
            campaign.resources._LAST_REGION_SPANS.get("gold_stockpile"),
            (10, 15),
        )

    def test_cached_value_used_after_low_conf_streak(self):
        cache = campaign.resources.ResourceCache()
        cache.last_resource_values["wood_stockpile"] = 100
        cache.last_resource_ts["wood_stockpile"] = 0
        cache.resource_low_conf_counts = {}
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        def fake_regions(frame, icons, cache_obj):
            return {"wood_stockpile": (0, 0, 5, 5)}

        def fake_prepare_roi(frame, regions, name, required_set, cache_obj):
            x, y, w, h = regions[name]
            roi = np.zeros((h, w, 3), dtype=np.uint8)
            gray = np.zeros((h, w), dtype=np.uint8)
            return x, y, w, h, roi, gray, 0, 0

        def fake_ocr(*args, **kwargs):
            return "90", {"text": ["90"]}, None, True

        with patch.dict(
            campaign.resources.CFG,
            {"resource_low_conf_streak": 2, "allow_low_conf_digits": True},
        ), patch(
            "script.resources.reader.core.detect_resource_regions",
            side_effect=fake_regions,
        ), patch(
            "script.resources.reader.core.prepare_roi", side_effect=fake_prepare_roi
        ), patch(
            "script.resources.reader.core.execute_ocr", side_effect=fake_ocr
        ):
            res1, _ = campaign.resources._read_resources(
                frame, ["wood_stockpile"], ["wood_stockpile"], cache
            )
            res2, _ = campaign.resources._read_resources(
                frame, ["wood_stockpile"], ["wood_stockpile"], cache
            )

        self.assertEqual(res1["wood_stockpile"], 90)
        self.assertEqual(res2["wood_stockpile"], 100)

