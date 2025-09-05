import os
import runpy
import sys
import types
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
    types.SimpleNamespace(
        cvtColor=lambda src, code: src,
        resize=lambda img, *a, **k: img,
        threshold=lambda img, *a, **k: (None, img),
        imread=lambda *a, **k: np.zeros((1, 1), dtype=np.uint8),
        imwrite=lambda *a, **k: True,
        matchTemplate=lambda *a, **k: np.zeros((1, 1), dtype=np.uint8),
        IMREAD_GRAYSCALE=0,
        COLOR_BGR2GRAY=0,
        INTER_LINEAR=0,
        THRESH_BINARY=0,
        THRESH_OTSU=0,
        TM_CCOEFF_NORMED=0,
    ),
)
sys.modules.setdefault(
    "pytesseract",
    types.SimpleNamespace(
        pytesseract=types.SimpleNamespace(tesseract_cmd=""),
        image_to_string=lambda *a, **k: "",
        image_to_data=lambda *a, **k: {"text": [""], "conf": ["0"]},
        Output=types.SimpleNamespace(DICT=0),
    ),
)

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.common as common
state = common.init_common()
import script.config_utils as config_utils
import script.resources as resources


class TestForagingScenario(TestCase):
    """Ensure the Foraging scenario script wires up common routines correctly."""

    def test_main_initialises_counters(self):
        info = config_utils.ScenarioInfo(
            starting_villagers=3,
            starting_idle_villagers=3,
            population_limit=4,
            starting_resources={
                "wood_stockpile": 200,
                "food_stockpile": 200,
                "gold_stockpile": 0,
                "stone_stockpile": 100,
            },
            objective_villagers=0,
            starting_buildings={"Town Center": 1},
        )

        gathered = dict(info.starting_resources)
        gathered["idle_villager"] = info.starting_idle_villagers

        with patch(
            "script.hud.wait_hud", return_value=((0, 0, 0, 0), "asset")
        ) as wait_mock, patch(
            "script.config_utils.parse_scenario_info", return_value=info
        ) as parse_mock, patch(
            "script.resources.reader.gather_hud_stats",
            return_value=(gathered, (info.starting_villagers, info.population_limit)),
        ) as gather_mock, patch.object(
            resources.reader, "RESOURCE_CACHE", resources.ResourceCache()
        ):
            runpy.run_path(
                os.path.join("campaigns", "Ascent_of_Egypt", "Egypt_2_Foraging.py"),
                run_name="__main__",
            )

            self.assertEqual(state.current_pop, info.starting_villagers)
            self.assertEqual(state.pop_cap, info.population_limit)
            self.assertEqual(state.target_pop, info.objective_villagers)
            self.assertEqual(
                resources.reader.RESOURCE_CACHE.last_resource_values, gathered
            )
            for name in gathered:
                self.assertIn(name, resources.reader.RESOURCE_CACHE.last_resource_ts)

            wait_mock.assert_called_once()
            parse_mock.assert_called_once()
            gather_mock.assert_called_once()

    def test_aborts_on_idle_villager_mismatch(self):
        info = config_utils.ScenarioInfo(
            starting_villagers=3,
            starting_idle_villagers=3,
            population_limit=4,
            starting_resources={
                "wood_stockpile": 200,
                "food_stockpile": 200,
                "gold_stockpile": 0,
                "stone_stockpile": 100,
            },
            objective_villagers=0,
            starting_buildings={"Town Center": 1},
        )

        gathered = dict(info.starting_resources)
        gathered["idle_villager"] = info.starting_idle_villagers - 1

        import importlib

        module = importlib.import_module(
            "campaigns.Ascent_of_Egypt.Egypt_2_Foraging"
        )

        with patch.object(module.hud, "wait_hud", return_value=((0, 0, 0, 0), "asset")), \
            patch.object(module, "parse_scenario_info", return_value=info), \
            patch.object(module.resources, "gather_hud_stats", return_value=(gathered, (info.starting_villagers, info.population_limit))) as gather_mock, \
            patch.object(module.resources, "RESOURCE_CACHE", resources.ResourceCache()), \
            self.assertLogs(module.logger, level="ERROR") as log_ctx:
            module.main()

        gather_mock.assert_called_once()
        self.assertEqual(module.resources.RESOURCE_CACHE.last_resource_values, {})
        self.assertTrue(
            any("idle villager" in m.lower() for m in log_ctx.output)
        )

    def test_aborts_on_population_mismatch(self):
        info = config_utils.ScenarioInfo(
            starting_villagers=3,
            starting_idle_villagers=3,
            population_limit=4,
            starting_resources={
                "wood_stockpile": 200,
                "food_stockpile": 200,
                "gold_stockpile": 0,
                "stone_stockpile": 100,
            },
            objective_villagers=0,
            starting_buildings={"Town Center": 1},
        )

        gathered = dict(info.starting_resources)
        gathered["idle_villager"] = info.starting_idle_villagers

        import importlib

        module = importlib.import_module(
            "campaigns.Ascent_of_Egypt.Egypt_2_Foraging"
        )

        with patch.object(module.hud, "wait_hud", return_value=((0, 0, 0, 0), "asset")), \
            patch.object(module, "parse_scenario_info", return_value=info), \
            patch.object(module.resources, "gather_hud_stats", return_value=(gathered, (info.starting_villagers - 1, info.population_limit))) as gather_mock, \
            patch.object(module.resources, "RESOURCE_CACHE", resources.ResourceCache()), \
            self.assertLogs(module.logger, level="ERROR") as log_ctx:
            module.main()

        gather_mock.assert_called_once()
        self.assertEqual(module.resources.RESOURCE_CACHE.last_resource_values, {})
        self.assertTrue(any("population" in m.lower() for m in log_ctx.output))

    def test_main_invokes_run_mission(self):
        info = config_utils.ScenarioInfo(
            starting_villagers=3,
            starting_idle_villagers=3,
            population_limit=4,
            starting_resources={
                "wood_stockpile": 200,
                "food_stockpile": 200,
                "gold_stockpile": 0,
                "stone_stockpile": 100,
            },
            objective_villagers=0,
            starting_buildings={"Town Center": 1},
        )

        gathered = dict(info.starting_resources)
        gathered["idle_villager"] = info.starting_idle_villagers

        import importlib

        module = importlib.import_module(
            "campaigns.Ascent_of_Egypt.Egypt_2_Foraging"
        )

        with patch.object(module.hud, "wait_hud", return_value=((0, 0, 0, 0), "asset")), \
            patch.object(module, "parse_scenario_info", return_value=info), \
            patch.object(module.resources, "gather_hud_stats", return_value=(gathered, (info.starting_villagers, info.population_limit))), \
            patch.object(module.resources, "validate_starting_resources"), \
            patch.object(module, "run_mission") as mission_mock, \
            patch.object(module.resources, "RESOURCE_CACHE", resources.ResourceCache()):
            module.main()

        mission_mock.assert_called_once_with(info, state=module.STATE)

    def test_run_mission_calls_build_functions(self):
        info = config_utils.ScenarioInfo(
            starting_villagers=3,
            starting_idle_villagers=2,
            population_limit=10,
            starting_resources=None,
            objective_villagers=5,
            starting_buildings={"Town Center": 1},
        )

        import importlib

        module = importlib.import_module(
            "campaigns.Ascent_of_Egypt.Egypt_2_Foraging"
        )

        state.current_pop = info.starting_villagers
        state.target_pop = info.objective_villagers

        calls = []

        def fake_select(*a, **k):
            return True

        def fake_click(*a, **k):
            pass

        def fake_granary(*a, **k):
            calls.append("granary")
            return True

        def fake_storage(*a, **k):
            calls.append("storage")
            return True

        def fake_dock(*a, **k):
            calls.append("dock")
            return True

        def fake_train(target_pop, state=state):
            state.current_pop += 1

        with patch.object(module.villager, "select_idle_villager", side_effect=fake_select), \
            patch.object(module.input_utils, "_click_norm", side_effect=fake_click), \
            patch.object(module.villager, "build_granary", side_effect=fake_granary), \
            patch.object(module.villager, "build_storage_pit", side_effect=fake_storage), \
            patch.object(module.villager, "build_dock", side_effect=fake_dock), \
            patch.object(module, "train_villagers", side_effect=fake_train):
            module.run_mission(info, state=state)

        self.assertIn("granary", calls)
        self.assertIn("storage", calls)
        self.assertIn("dock", calls)
        self.assertGreaterEqual(state.current_pop, state.target_pop)

    def test_aborts_on_pop_cap_mismatch(self):
        info = config_utils.ScenarioInfo(
            starting_villagers=3,
            starting_idle_villagers=3,
            population_limit=4,
            starting_resources={
                "wood_stockpile": 200,
                "food_stockpile": 200,
                "gold_stockpile": 0,
                "stone_stockpile": 100,
            },
            objective_villagers=0,
            starting_buildings={"Town Center": 1},
        )

        gathered = dict(info.starting_resources)
        gathered["idle_villager"] = info.starting_idle_villagers

        import importlib

        module = importlib.import_module(
            "campaigns.Ascent_of_Egypt.Egypt_2_Foraging"
        )

        with patch.object(module.hud, "wait_hud", return_value=((0, 0, 0, 0), "asset")), \
            patch.object(module, "parse_scenario_info", return_value=info), \
            patch.object(module.resources, "gather_hud_stats", return_value=(gathered, (info.starting_villagers, info.population_limit - 1))) as gather_mock, \
            patch.object(module.resources, "RESOURCE_CACHE", resources.ResourceCache()), \
            self.assertLogs(module.logger, level="ERROR") as log_ctx:
            module.main()

        gather_mock.assert_called_once()
        self.assertEqual(module.resources.RESOURCE_CACHE.last_resource_values, {})
        self.assertTrue(any("population" in m.lower() for m in log_ctx.output))


