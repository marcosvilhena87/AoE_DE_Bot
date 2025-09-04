import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch
import numpy as np

# Stub modules requiring GUI before importing bot modules

dummy_pg = types.SimpleNamespace(
    PAUSE=0,
    FAILSAFE=True,
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


dummy_cv2 = types.SimpleNamespace(
    imread=lambda *a, **k: np.zeros((1, 1), dtype=np.uint8),
    IMREAD_GRAYSCALE=0,
)

sys.modules.setdefault("pyautogui", dummy_pg)
sys.modules.setdefault("mss", types.SimpleNamespace(mss=lambda: DummyMSS()))
sys.modules.setdefault("cv2", dummy_cv2)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.input_utils as input_utils


class TestTemporaryFailsafe(TestCase):
    def test_restores_state_on_exit(self):
        pg = input_utils.pg
        pg.FAILSAFE = True
        with input_utils.temporary_failsafe_disabled():
            self.assertFalse(pg.FAILSAFE)
        self.assertTrue(pg.FAILSAFE)

    def test_restores_state_on_exception(self):
        pg = input_utils.pg
        pg.FAILSAFE = True
        with self.assertRaises(RuntimeError):
            with input_utils.temporary_failsafe_disabled():
                self.assertFalse(pg.FAILSAFE)
                raise RuntimeError("boom")
        self.assertTrue(pg.FAILSAFE)

    def test_move_cursor_safe_uses_center_and_restores_state(self):
        pg = input_utils.pg
        pg.FAILSAFE = True
        with patch("script.input_utils.pg.moveTo") as move_mock:
            input_utils._move_cursor_safe()
        move_mock.assert_called_once_with(100, 100)
        self.assertTrue(pg.FAILSAFE)
