import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np

# Stub modules

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

os.environ.setdefault("TESSERACT_CMD", "/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.common as common
import script.buildings.town_center as tc
import script.units.villager as villager
import script.config_utils as config_utils
import script.hud as hud


class TestInternalPopulation(TestCase):
    def test_parse_scenario_info(self):
        info = config_utils.parse_scenario_info("campaigns/Ascent_of_Egypt/Egypt_1_Hunting.txt")
        self.assertEqual(info.starting_villagers, 3)
        self.assertEqual(info.population_limit, 50)
        self.assertEqual(info.objective_villagers, 7)
        self.assertEqual(
            info.starting_resources,
            {
                "wood_stockpile": 80,
                "food_stockpile": 140,
                "gold_stockpile": 0,
                "stone_stockpile": 0,
            },
        )

    def test_train_villagers_updates_population_without_ocr(self):
        common.CURRENT_POP = 3
        common.POP_CAP = 4
        def fake_build_house():
            common.POP_CAP += 4
            return True

        with patch(
            "script.resources.read_resources_from_hud",
            return_value=({"food_stockpile": 500}, (None, None)),
        ), \
             patch("script.buildings.town_center.build_house", side_effect=fake_build_house) as build_house_mock, \
             patch("script.buildings.town_center.select_idle_villager", return_value=True), \
             patch("script.hud.read_population_from_hud") as read_pop_mock:
            tc.train_villagers(7)
            self.assertEqual(common.CURRENT_POP, 7)
            self.assertEqual(common.POP_CAP, 8)
            build_house_mock.assert_called_once()
            read_pop_mock.assert_not_called()

    def test_population_fallback_error_included(self):
        with patch(
            "script.resources.read_population_from_roi",
            side_effect=common.PopulationReadError("primary fail"),
        ), patch(
            "script.resources.read_resources_from_hud",
            side_effect=common.ResourceReadError("fallback fail"),
        ), patch("script.resources.locate_resource_panel", return_value={}):
            with self.assertRaises(common.PopulationReadError) as ctx:
                hud.read_population_from_hud()
        self.assertIn("fallback fail", str(ctx.exception))
