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

    def _run_main(self, res_sequence):
        res_list = list(res_sequence)

        def gh_side_effect(*args, **kwargs):
            campaign.resources._LAST_REGION_BOUNDS = {
                "wood_stockpile": (0, 0, 5, 5)
            }
            return res_list.pop(0), (0, 0)

        logger_mock = MagicMock()

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
            patch("campaign.screen_utils._grab_frame", return_value=np.zeros((10, 10, 3))), \
            patch("campaign.logging.getLogger", return_value=logger_mock), \
            patch.object(campaign.resources, "ROOT", Path(tmpdir)), \
            patch("campaign.resources.cv2.imwrite"):
            campaign.main()

        return logger_mock

    def test_retry_warns_and_continues_on_near_match(self):
        res_seq = [{"wood_stockpile": 50}, {"wood_stockpile": 88}]
        logger_mock = self._run_main(res_seq)
        self.assertGreaterEqual(logger_mock.warning.call_count, 1)

    def test_retry_exits_when_far_off(self):
        res_seq = [{"wood_stockpile": 50}, {"wood_stockpile": 40}]
        with self.assertRaises(SystemExit):
            self._run_main(res_seq)

