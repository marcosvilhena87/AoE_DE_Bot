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

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.common as common
state = common.init_common()
import script.input_utils as input_utils
import script.units.villager as villager


class TestClickAndBuildHouse(TestCase):
    def test_click_norm_passes_button(self):
        with patch("script.input_utils.pg.click") as click_mock:
            input_utils._click_norm(0.5, 0.5, button="right")
        click_mock.assert_called_once_with(100, 100, button="right")

    def test_build_house_uses_right_click_and_updates_population(self):
        state.pop_cap = 4
        expected_coords = tuple(common.CFG["areas"]["house_spot"])
        with patch("script.resources.reader.read_resources_from_hud") as read_mock, \
            patch("script.hud.read_population_from_hud") as read_pop_mock, \
            patch("script.input_utils._press_key_safe"), \
            patch("script.input_utils._click_norm") as click_mock, \
            patch("script.units.villager.time.sleep"):
            result = villager.build_house(state=state)
        self.assertTrue(result)
        self.assertEqual(state.pop_cap, 8)
        self.assertEqual(click_mock.call_count, 2)
        self.assertEqual(click_mock.call_args_list[0].args, expected_coords)
        self.assertEqual(click_mock.call_args_list[1].args, expected_coords)
        self.assertEqual(click_mock.call_args_list[1].kwargs["button"], "right")
        read_mock.assert_not_called()
        read_pop_mock.assert_not_called()
