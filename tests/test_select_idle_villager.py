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
import script.common as common

state = common.init_common()


class TestSelectIdleVillager(TestCase):
    def test_presses_idle_key_when_configured(self):
        with patch("script.input_utils._press_key_safe") as press_mock:
            self.assertTrue(villager.select_idle_villager())
            press_mock.assert_called_once_with(state.config["keys"]["idle_vill"], 0.1)

    def test_returns_false_when_key_missing(self):
        new_state = common.init_common(state=common.BotState())
        new_state.config["keys"].pop("idle_vill", None)
        with patch("script.input_utils._press_key_safe") as press_mock:
            self.assertFalse(villager.select_idle_villager(state=new_state))
            press_mock.assert_not_called()

