import os
import sys
import types
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure dummy modules for pyautogui and mss etc as in other tests

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


class TestCampaignNoStartingResources(TestCase):
    def test_skips_validation_when_starting_resources_none(self):
        info = types.SimpleNamespace(
            starting_resources=None,
            starting_villagers=3,
            starting_idle_villagers=0,
            objective_villagers=5,
        )

        logger_mock = MagicMock()
        validate_mock = MagicMock()

        dummy_module = types.SimpleNamespace(run_mission=lambda *a, **k: None)

        with patch("campaign.parse_scenario_info", return_value=info), \
            patch(
                "campaign.argparse.ArgumentParser.parse_args",
                return_value=types.SimpleNamespace(scenario="dummy"),
            ), \
            patch("campaign.screen_utils.screen_capture.init_sct"), \
            patch("campaign.screen_utils.screen_capture.teardown_sct"), \
            patch("campaign.hud.wait_hud", return_value=({}, "asset")), \
            patch("campaign.resources.gather_hud_stats", return_value=({}, (0, 0))), \
            patch("campaign.resources.validate_starting_resources", validate_mock), \
            patch("campaign.logging.getLogger", return_value=logger_mock), \
            patch("campaign.resources.cv2.imwrite"), \
            patch("importlib.import_module", return_value=dummy_module):
            campaign.main()

        validate_mock.assert_not_called()
        self.assertTrue(
            any(
                "starting resources" in str(c.args[0]).lower()
                for c in logger_mock.warning.call_args_list
            )
        )
