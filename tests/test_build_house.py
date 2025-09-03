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
import script.input_utils as input_utils
import script.units.villager as villager


class TestClickAndBuildHouse(TestCase):
    def test_click_norm_passes_button(self):
        with patch("script.input_utils.pg.click") as click_mock:
            input_utils._click_norm(0.5, 0.5, button="right")
        click_mock.assert_called_once_with(100, 100, button="right")

    def test_build_house_uses_right_click_and_updates_population(self):
        common.POP_CAP = 4
        expected_coords = tuple(common.CFG["areas"]["house_spot"])
        with patch(
            "script.resources.reader.read_resources_from_hud",
            return_value=({"wood_stockpile": 100}, (None, None)),
        ), \
            patch("script.input_utils._press_key_safe"), \
            patch("script.input_utils._click_norm") as click_mock, \
            patch("script.hud.read_population_from_hud", return_value=(0, 8, False)) as read_pop_mock, \
            patch("script.units.villager.time.sleep"):
            result = villager.build_house()
        self.assertTrue(result)
        self.assertEqual(common.POP_CAP, 8)
        self.assertEqual(click_mock.call_count, 2)
        self.assertEqual(click_mock.call_args_list[0].args, expected_coords)
        self.assertEqual(click_mock.call_args_list[1].args, expected_coords)
        self.assertEqual(click_mock.call_args_list[1].kwargs["button"], "right")
        read_pop_mock.assert_called_once()


class TestBuildHouseResourceRetry(TestCase):
    def test_build_house_stops_after_single_failed_attempt(self):
        """Ensure build_house aborts after one failed resource read."""
        side_effect = [
            common.ResourceReadError("fail1"),
            common.ResourceReadError("fail2"),
        ]
        with patch(
            "script.resources.reader.read_resources_from_hud",
            side_effect=side_effect,
        ) as read_mock, patch("script.input_utils._press_key_safe") as press_mock, patch(
            "script.input_utils._click_norm"
        ) as click_mock, patch("script.units.villager.time.sleep"), patch(
            "script.hud.wait_hud"
        ):
            result = villager.build_house()
        self.assertFalse(result)
        self.assertEqual(read_mock.call_count, 2)
        press_mock.assert_not_called()
        click_mock.assert_not_called()

    def test_build_house_recovers_from_transient_failure(self):
        common.POP_CAP = 4
        side_effect = [
            common.ResourceReadError("fail1"),
            ({"wood_stockpile": 100}, (None, None)),
        ]
        with patch(
            "script.resources.reader.read_resources_from_hud",
            side_effect=side_effect,
        ) as read_mock, patch("script.input_utils._press_key_safe"), patch(
            "script.input_utils._click_norm"
        ), patch(
            "script.hud.read_population_from_hud", return_value=(0, 8, False)
        ), patch("script.units.villager.time.sleep"), patch(
            "script.hud.wait_hud"
        ):
            result = villager.build_house()
        self.assertTrue(result)
        self.assertEqual(read_mock.call_count, 2)
        self.assertEqual(common.POP_CAP, 8)
