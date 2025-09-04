import os
import sys
import types
from unittest import TestCase
from unittest.mock import MagicMock, patch

# Setup dummy dependencies

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
        import numpy as np
        h, w = region["height"], region["width"]
        return np.zeros((h, w, 4), dtype=np.uint8)

sys.modules.setdefault("pyautogui", dummy_pg)
sys.modules.setdefault("mss", types.SimpleNamespace(mss=lambda: DummyMSS()))
os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import campaign


class TestResourceReadErrorSurface(TestCase):
    def test_resource_read_error_surfaces_message(self):
        info = types.SimpleNamespace(
            starting_resources={},
            starting_villagers=3,
            starting_idle_villagers=0,
            objective_villagers=5,
        )
        err_msg = "boom"
        logger_mock = MagicMock()
        dummy_module = types.SimpleNamespace(run_mission=lambda *a, **k: None)

        with patch("campaign.parse_scenario_info", return_value=info), \
            patch(
                "campaign.argparse.ArgumentParser.parse_args",
                return_value=types.SimpleNamespace(scenario="dummy"),
            ), \
            patch("campaign.screen_utils.screen_capture.init_sct"), \
            patch("campaign.screen_utils.screen_capture.teardown_sct"), \
            patch("campaign.hud.wait_hud", return_value=({}, "asset")), \
            patch(
                "campaign.resources.gather_hud_stats",
                side_effect=campaign.common.ResourceReadError(err_msg),
            ), \
            patch("campaign.logging.getLogger", return_value=logger_mock), \
            patch("importlib.import_module", return_value=dummy_module):
            with self.assertRaises(SystemExit) as ctx:
                campaign.main()

        self.assertIsInstance(ctx.exception.__cause__, campaign.common.ResourceReadError)
        self.assertEqual(str(ctx.exception.__cause__), err_msg)
        self.assertTrue(
            any(
                err_msg in " ".join(map(str, call.args))
                for call in logger_mock.error.call_args_list
            )
        )
