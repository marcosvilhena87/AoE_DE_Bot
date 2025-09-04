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
os.environ.setdefault("TESSERACT_CMD", "/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.common as common
state = common.init_common()
import script.buildings.town_center as tc
import script.units.villager as villager


class TestMissingResourceBar(TestCase):
    def test_train_villagers_does_not_read_missing_bar(self):
        state.current_pop = 3
        state.pop_cap = 5
        pop_vals = [(4, 5, False), (5, 5, False)]
        with patch(
            "script.resources.reader.read_resources_from_hud",
            side_effect=common.ResourceReadError("missing"),
        ) as read_mock, \
             patch("script.buildings.town_center.select_idle_villager", return_value=True), \
             patch("script.buildings.town_center.build_house"), \
             patch("script.hud.read_population_from_hud", side_effect=pop_vals) as pop_mock:
            tc.train_villagers(5, state=state)
        self.assertEqual(state.current_pop, 5)
        read_mock.assert_not_called()
        self.assertEqual(pop_mock.call_count, 2)

    def test_build_house_handles_missing_bar(self):
        state.pop_cap = 4

        def pop_side_effect(*a, **k):
            if pop_side_effect.calls == 0:
                pop_side_effect.calls += 1
                return (0, 4, False)
            return (0, 8, False)

        pop_side_effect.calls = 0

        with patch(
            "script.resources.reader.read_resources_from_hud",
            side_effect=common.ResourceReadError("missing"),
        ) as read_mock, patch(
            "script.hud.read_population_from_hud", side_effect=pop_side_effect
        ), patch(
            "script.template_utils.find_template", return_value=(None, 0.9, None)
        ), patch(
            "pathlib.Path.exists", return_value=True
        ), patch(
            "cv2.imread", return_value=np.zeros((10, 10), dtype=np.uint8)
        ), patch(
            "script.screen_utils.screen_capture.grab_frame",
            return_value=np.zeros((200, 200, 3), dtype=np.uint8),
        ), patch(
            "script.input_utils._press_key_safe"
        ), patch(
            "script.input_utils._click_norm"
        ), patch(
            "script.units.villager.time.sleep"
        ):
            result = villager.build_house(state=state)
        self.assertTrue(result)
        self.assertEqual(state.pop_cap, 8)
        read_mock.assert_called()
