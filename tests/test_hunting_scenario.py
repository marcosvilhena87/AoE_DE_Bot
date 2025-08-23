import os
import sys
import types
import runpy
from unittest import TestCase
from unittest.mock import patch

import numpy as np


# ---------------------------------------------------------------------------
# Stub out heavy external modules before importing any of the project code.
# ---------------------------------------------------------------------------

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
    types.SimpleNamespace(imread=lambda *a, **k: None, IMREAD_GRAYSCALE=0),
)
sys.modules.setdefault(
    "pytesseract",
    types.SimpleNamespace(
        pytesseract=types.SimpleNamespace(tesseract_cmd=""),
        image_to_string=lambda *a, **k: "",
    ),
)

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.common as common
import script.config_utils as config_utils


class TestHuntingScenario(TestCase):
    """Ensure the Hunting scenario script wires up common routines correctly."""

    def test_main_runs_economic_loop(self):
        info = config_utils.ScenarioInfo(
            starting_villagers=3, population_limit=50, objective_villagers=7
        )

        with patch(
            "script.hud.wait_hud", return_value=((0, 0, 0, 0), "asset")
        ) as wait_mock, patch(
            "script.config_utils.parse_scenario_info", return_value=info
        ) as parse_mock, patch("script.villager.econ_loop") as econ_mock:
            runpy.run_path(
                os.path.join("campaigns", "Ascent_of_Egypt", "1.Hunting.py"),
                run_name="__main__",
            )

            self.assertEqual(common.CURRENT_POP, info.starting_villagers)
            self.assertEqual(common.POP_CAP, 4)
            self.assertEqual(common.TARGET_POP, info.objective_villagers)
            econ_mock.assert_called_once_with(minutes=common.CFG["loop_minutes"])
            wait_mock.assert_called_once()
            parse_mock.assert_called_once()

