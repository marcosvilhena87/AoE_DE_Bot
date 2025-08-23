import os
import sys
import types
from unittest import TestCase

import numpy as np

# Stub GUI-dependent modules before importing the bot code

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
import script.config_utils as config_utils


class TestValidateKeys(TestCase):
    def test_missing_required_hotkeys(self):
        cfg = {
            "areas": {
                "house_spot": [0, 0],
                "granary_spot": [0, 0],
                "storage_spot": [0, 0],
                "wood_spot": [0, 0],
                "food_spot": [0, 0],
                "pop_box": [0, 0, 0, 0],
            },
            "keys": {
                "idle_vill": ".",
                "build_menu": "b",
                "select_tc": "h",
                "train_vill": "q",
            },
        }
        with self.assertRaises(RuntimeError) as cm:
            config_utils.validate_config(cfg)
        msg = str(cm.exception)
        self.assertIn("house", msg)
        self.assertIn("config.sample.json", msg)

