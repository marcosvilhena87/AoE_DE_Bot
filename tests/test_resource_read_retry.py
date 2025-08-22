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


class TestResourceReadRetry(TestCase):
    def test_train_villagers_retries_before_stopping(self):
        common.CURRENT_POP = 3
        common.POP_CAP = 10
        side_effect = [
            common.ResourceReadError("fail1"),
            common.ResourceReadError("fail2"),
            {"food_stockpile": 100},
        ]
        with patch(
            "script.common.read_resources_from_hud",
            side_effect=side_effect,
        ) as read_mock, \
             patch("script.town_center.select_idle_villager"), \
             patch("script.town_center.build_house"):
            tc.train_villagers(4)
        self.assertEqual(common.CURRENT_POP, 4)
        self.assertEqual(read_mock.call_count, 3)
