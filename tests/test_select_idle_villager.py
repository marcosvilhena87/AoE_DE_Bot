import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np

# Stub modules that require GUI

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
import script.units.villager as villager


class TestSelectIdleVillager(TestCase):
    def test_returns_true_when_count_decreases(self):
        villager._last_idle_villager_count = 0
        reads = iter([
            ({"idle_villager": 2}, (None, None)),
            ({"idle_villager": 1}, (None, None)),
        ])

        with patch("script.input_utils._press_key_safe") as press_mock, \
             patch("script.resources.reader.read_resources_from_hud", side_effect=lambda *a, **k: next(reads)), \
             patch("script.hud.read_population_from_hud", return_value=(2, 10)):
            self.assertTrue(villager.select_idle_villager())
            press_mock.assert_called_once()

    def test_returns_false_when_no_idle_villagers(self):
        villager._last_idle_villager_count = 0
        reads = iter([
            ({"idle_villager": 0}, (None, None)),
            ({"idle_villager": 0}, (None, None)),
        ])

        with patch("script.input_utils._press_key_safe") as press_mock, \
             patch("script.resources.reader.read_resources_from_hud", side_effect=lambda *a, **k: next(reads)), \
             patch("script.hud.read_population_from_hud", return_value=(0, 10)):
            self.assertFalse(villager.select_idle_villager())
            press_mock.assert_not_called()

    def test_ignores_idle_count_exceeding_population(self):
        villager._last_idle_villager_count = 0
        reads = iter([
            ({"idle_villager": 5}, (None, None)),
            ({"idle_villager": 5}, (None, None)),
        ])

        with patch("script.input_utils._press_key_safe") as press_mock, \
             patch("script.resources.reader.read_resources_from_hud", side_effect=lambda *a, **k: next(reads)), \
             patch("script.hud.read_population_from_hud", return_value=(3, 10)):
            self.assertFalse(villager.select_idle_villager())
            press_mock.assert_not_called()
