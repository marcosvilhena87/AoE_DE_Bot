import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np

# Stub modules

sys.modules.setdefault(
    "cv2",
    types.SimpleNamespace(
        cvtColor=lambda src, code: src,
        resize=lambda img, *a, **k: img,
        threshold=lambda img, *a, **k: (None, img),
        imread=lambda *a, **k: np.zeros((1, 1), dtype=np.uint8),
        imwrite=lambda *a, **k: True,
        IMREAD_GRAYSCALE=0,
        COLOR_BGR2GRAY=0,
        INTER_LINEAR=0,
        THRESH_BINARY=0,
        THRESH_OTSU=0,
    ),
)

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
state = common.init_common()
import script.buildings.town_center as tc
import script.units.villager as villager
import script.config_utils as config_utils
import script.hud as hud


class TestInternalPopulation(TestCase):
    def test_parse_scenario_info(self):
        info = config_utils.parse_scenario_info("campaigns/Ascent_of_Egypt/Egypt_1_Hunting.txt")
        self.assertEqual(info.starting_villagers, 3)
        self.assertEqual(info.starting_idle_villagers, 3)
        self.assertEqual(info.population_limit, 4)
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
        self.assertEqual(info.starting_buildings, {"Town Center": 1})

    def test_train_villagers_updates_population_without_ocr(self):
        state.current_pop = 3
        state.pop_cap = 4
        def fake_build_house(state=state):
            state.pop_cap += 4
            return True

        pop_vals = [
            (4, 4, False),
            (5, 8, False),
            (6, 8, False),
            (7, 8, False),
        ]
        with patch("script.resources.reader.read_resources_from_hud") as read_mock, \
             patch("script.buildings.town_center.build_house", side_effect=fake_build_house) as build_house_mock, \
             patch("script.buildings.town_center.select_idle_villager", return_value=True), \
             patch("script.hud.read_population_from_hud", side_effect=pop_vals) as pop_mock:
            tc.train_villagers(7, state=state)
            self.assertEqual(state.current_pop, 7)
            self.assertEqual(state.pop_cap, 8)
            build_house_mock.assert_called_once()
            read_mock.assert_not_called()
            self.assertEqual(pop_mock.call_count, 4)

    def test_run_mission_builds_house_before_training(self):
        from campaigns.Ascent_of_Egypt import Egypt_1_Hunting as hunt

        info = config_utils.ScenarioInfo(
            starting_villagers=3,
            starting_idle_villagers=3,
            population_limit=4,
            starting_resources={},
            objective_villagers=4,
            starting_buildings={"Town Center": 1},
        )

        state.current_pop = 3
        state.pop_cap = 4
        state.target_pop = 4

        resources = hunt.resources
        resources.RESOURCE_CACHE.last_resource_values["food_stockpile"] = 0

        call_order = []

        def fake_build_house(state=state):
            call_order.append("build_house")
            state.pop_cap += 4
            return True

        def fake_train(target, state=state):
            call_order.append("train_villagers")
            state.current_pop += 1

        with patch("script.units.villager.select_idle_villager", return_value=True), \
             patch("script.units.villager.build_house", side_effect=fake_build_house), \
             patch("campaigns.Ascent_of_Egypt.Egypt_1_Hunting.train_villagers", side_effect=fake_train), \
             patch("script.resources.reader.read_resources_from_hud", return_value=({"food_stockpile": 100}, None)):
            hunt.run_mission(info, state=state)

        self.assertEqual(call_order[0], "build_house")
        self.assertIn("train_villagers", call_order)

