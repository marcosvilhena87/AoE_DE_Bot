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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.common as common
import script.town_center as tc
import script.villager as villager


class TestInternalPopulation(TestCase):
    def test_parse_scenario_info(self):
        info = common.parse_scenario_info("campaigns/Ascent_of_Egypt/1.Hunting.txt")
        self.assertEqual(info.starting_villagers, 3)
        self.assertEqual(info.population_limit, 50)
        self.assertEqual(info.objective_villagers, 7)

    def test_train_villagers_updates_population_without_ocr(self):
        common.CURRENT_POP = 3
        common.POP_CAP = 4
        def fake_build_house():
            common.POP_CAP += 4
            return True

        with patch(
            "script.common.read_resources_from_hud",
            return_value={"food_stockpile": 500},
        ), \
             patch("script.town_center.build_house", side_effect=fake_build_house) as build_house_mock, \
             patch("script.town_center.select_idle_villager"), \
             patch("script.common.read_population_from_hud") as read_pop_mock:
            tc.train_villagers(7)
            self.assertEqual(common.CURRENT_POP, 7)
            self.assertEqual(common.POP_CAP, 8)
            build_house_mock.assert_called_once()
            read_pop_mock.assert_not_called()
