# tests/test_hunting_scenario.py
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
import script.config_utils as config_utils
import script.resources as resources


class TestHuntingScenario(TestCase):
    """Ensure the Hunting scenario script wires up common routines correctly."""

    def test_main_initialises_counters(self):
        info = config_utils.ScenarioInfo(
            starting_villagers=3,
            starting_idle_villagers=3,
            population_limit=4,
            starting_resources={
                "wood_stockpile": 80,
                "food_stockpile": 140,
                "gold_stockpile": 0,
                "stone_stockpile": 0,
            },
            objective_villagers=7,
            starting_buildings={"Town Center": 1},
        )

        gathered = dict(info.starting_resources)
        gathered["idle_villager"] = info.starting_idle_villagers

        with patch("script.hud.wait_hud", return_value=((0, 0, 0, 0), "asset")) as wait_mock, \
             patch("script.config_utils.parse_scenario_info", return_value=info) as parse_mock, \
             patch(
                 "script.resources.reader.gather_hud_stats",
                 return_value=(gathered, (info.starting_villagers, info.population_limit)),
             ) as gather_mock, \
             patch.object(resources.reader, "RESOURCE_CACHE", resources.ResourceCache()):
            runpy.run_path(
                os.path.join("campaigns", "Ascent_of_Egypt", "Egypt_1_Hunting.py"),
                run_name="__main__",
            )

            self.assertEqual(common.CURRENT_POP, info.starting_villagers)
            self.assertEqual(common.POP_CAP, info.population_limit)
            self.assertEqual(common.TARGET_POP, info.objective_villagers)
            self.assertEqual(
                resources.reader.RESOURCE_CACHE.last_resource_values, gathered
            )
            for name in gathered:
                self.assertIn(name, resources.reader.RESOURCE_CACHE.last_resource_ts)

            wait_mock.assert_called_once()
            parse_mock.assert_called_once()
            gather_mock.assert_called_once()

    def test_aborts_when_town_center_missing(self):
        info = config_utils.ScenarioInfo(
            starting_villagers=3,
            starting_idle_villagers=3,
            population_limit=4,
            starting_resources=None,
            objective_villagers=7,
            starting_buildings={},
        )

        import importlib

        module = importlib.import_module(
            "campaigns.Ascent_of_Egypt.Egypt_1_Hunting"
        )

        with patch.object(module.hud, "wait_hud", return_value=((0, 0, 0, 0), "asset")), \
            patch.object(module, "parse_scenario_info", return_value=info), \
            patch.object(module.resources, "gather_hud_stats") as gather_mock, \
            self.assertLogs(module.logger, level="ERROR") as log_ctx:
            module.main()

        gather_mock.assert_not_called()
        self.assertTrue(
            any("town center" in m.lower() for m in log_ctx.output)
        )

    def test_idle_villager_cache_seeded(self):
        info = config_utils.ScenarioInfo(
            starting_villagers=3,
            starting_idle_villagers=3,
            population_limit=4,
            starting_resources=None,
            objective_villagers=8,
            starting_buildings={"Town Center": 1},
        )

        with patch(
            "script.hud.wait_hud", return_value=((0, 0, 0, 0), "asset")
        ) as wait_mock, patch(
            "script.config_utils.parse_scenario_info", return_value=info
        ) as parse_mock, patch(
            "script.resources.reader.gather_hud_stats",
            return_value=(
                {"idle_villager": info.starting_idle_villagers},
                (info.starting_villagers, info.population_limit),
            ),
        ) as gather_mock, patch.object(
            resources.reader, "RESOURCE_CACHE", resources.ResourceCache()
        ):
            runpy.run_path(
                os.path.join("campaigns", "Ascent_of_Egypt", "Egypt_1_Hunting.py"),
                run_name="__main__",
            )

            self.assertEqual(
                resources.reader.RESOURCE_CACHE.last_resource_values.get(
                    "idle_villager"
                ),
                info.starting_idle_villagers,
            )
            self.assertIn(
                "idle_villager", resources.reader.RESOURCE_CACHE.last_resource_ts
            )

            wait_mock.assert_called_once()
            parse_mock.assert_called_once()
            gather_mock.assert_called_once()

    def test_aborts_on_idle_villager_mismatch(self):
        info = config_utils.ScenarioInfo(
            starting_villagers=3,
            starting_idle_villagers=3,
            population_limit=4,
            starting_resources={
                "wood_stockpile": 80,
                "food_stockpile": 140,
                "gold_stockpile": 0,
                "stone_stockpile": 0,
            },
            objective_villagers=7,
            starting_buildings={"Town Center": 1},
        )

        gathered = dict(info.starting_resources)
        gathered["idle_villager"] = info.starting_idle_villagers - 1

        import importlib

        module = importlib.import_module(
            "campaigns.Ascent_of_Egypt.Egypt_1_Hunting"
        )

        with patch.object(module.hud, "wait_hud", return_value=((0, 0, 0, 0), "asset")), \
            patch.object(module, "parse_scenario_info", return_value=info), \
            patch.object(module.resources, "gather_hud_stats", return_value=(gathered, (info.starting_villagers, info.population_limit))) as gather_mock, \
            patch.object(module, "run_mission") as run_mock, \
            patch.object(module.resources, "RESOURCE_CACHE", resources.ResourceCache()), \
            self.assertLogs(module.logger, level="ERROR") as log_ctx:
            module.main()

        run_mock.assert_not_called()
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
                "wood_stockpile": 80,
                "food_stockpile": 140,
                "gold_stockpile": 0,
                "stone_stockpile": 0,
            },
            objective_villagers=7,
            starting_buildings={"Town Center": 1},
        )

        gathered = dict(info.starting_resources)
        gathered["idle_villager"] = info.starting_idle_villagers

        import importlib

        module = importlib.import_module(
            "campaigns.Ascent_of_Egypt.Egypt_1_Hunting"
        )

        with patch.object(module.hud, "wait_hud", return_value=((0, 0, 0, 0), "asset")), \
            patch.object(module, "parse_scenario_info", return_value=info), \
            patch.object(module.resources, "gather_hud_stats", return_value=(gathered, (info.starting_villagers - 1, info.population_limit))) as gather_mock, \
            patch.object(module, "run_mission") as run_mock, \
            patch.object(module.resources, "RESOURCE_CACHE", resources.ResourceCache()), \
            self.assertLogs(module.logger, level="ERROR") as log_ctx:
            module.main()

        run_mock.assert_not_called()
        gather_mock.assert_called_once()
        self.assertEqual(module.resources.RESOURCE_CACHE.last_resource_values, {})
        self.assertTrue(any("population" in m.lower() for m in log_ctx.output))

    def test_aborts_on_pop_cap_mismatch(self):
        info = config_utils.ScenarioInfo(
            starting_villagers=3,
            starting_idle_villagers=3,
            population_limit=4,
            starting_resources={
                "wood_stockpile": 80,
                "food_stockpile": 140,
                "gold_stockpile": 0,
                "stone_stockpile": 0,
            },
            objective_villagers=7,
            starting_buildings={"Town Center": 1},
        )

        gathered = dict(info.starting_resources)
        gathered["idle_villager"] = info.starting_idle_villagers

        import importlib

        module = importlib.import_module(
            "campaigns.Ascent_of_Egypt.Egypt_1_Hunting"
        )

        with patch.object(module.hud, "wait_hud", return_value=((0, 0, 0, 0), "asset")), \
            patch.object(module, "parse_scenario_info", return_value=info), \
            patch.object(module.resources, "gather_hud_stats", return_value=(gathered, (info.starting_villagers, info.population_limit - 1))) as gather_mock, \
            patch.object(module, "run_mission") as run_mock, \
            patch.object(module.resources, "RESOURCE_CACHE", resources.ResourceCache()), \
            self.assertLogs(module.logger, level="ERROR") as log_ctx:
            module.main()

        run_mock.assert_not_called()
        gather_mock.assert_called_once()
        self.assertEqual(module.resources.RESOURCE_CACHE.last_resource_values, {})
        self.assertTrue(any("population" in m.lower() for m in log_ctx.output))
