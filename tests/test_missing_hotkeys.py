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
import script.villager as villager


class TestMissingHotkeys(TestCase):
    def test_build_granary_missing_key(self):
        with patch.dict(common.CFG["keys"], {"granary": None}):
            with patch("script.common._press_key_safe"), patch("script.common._click_norm"):
                with self.assertLogs(level="WARNING") as log:
                    result = villager.build_granary()
        self.assertFalse(result)
        self.assertTrue(any("Granary" in message for message in log.output))

    def test_build_storage_pit_missing_key(self):
        with patch.dict(common.CFG["keys"], {"storage_pit": None}):
            with patch("script.common._press_key_safe"), patch("script.common._click_norm"):
                with self.assertLogs(level="WARNING") as log:
                    result = villager.build_storage_pit()
        self.assertFalse(result)
        self.assertTrue(any("Storage Pit" in message for message in log.output))
