import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np

# Stub modules that require a GUI/display before importing the bot modules

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
import script.units.villager as villager


class TestMissingResourceBar(TestCase):
    def test_train_villagers_handles_missing_bar(self):
        common.CURRENT_POP = 3
        common.POP_CAP = 5
        with patch(
            "script.resources.read_resources_from_hud",
            side_effect=common.ResourceReadError("missing"),
        ), \
             patch("script.town_center.select_idle_villager", return_value=True), \
             patch("script.town_center.build_house"):
            tc.train_villagers(5)
        self.assertEqual(common.CURRENT_POP, 3)

    def test_build_house_handles_missing_bar(self):
        common.POP_CAP = 4
        with patch(
            "script.resources.read_resources_from_hud",
            side_effect=common.ResourceReadError("missing"),
        ):
            self.assertFalse(villager.build_house())
