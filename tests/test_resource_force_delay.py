import os
import sys
import types
import numpy as np
from unittest import TestCase
from unittest.mock import patch

# Stub GUI-dependent modules before importing bot modules
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
import script.resources.reader as resources


class TestResourceForceDelay(TestCase):
    def setUp(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()

    def test_force_delay_waits_before_grab(self):
        calls = []

        def fake_sleep(t):
            calls.append(("sleep", t))

        def fake_grab():
            calls.append(("grab", None))
            return np.zeros((1, 1, 3), dtype=np.uint8)

        with patch("script.resources.reader.time.sleep", side_effect=fake_sleep), \
            patch("script.resources.reader.screen_utils._grab_frame", side_effect=fake_grab), \
            patch("script.resources.reader.detect_resource_regions", return_value={}), \
            patch("script.resources.reader.handle_ocr_failure"):
            resources.read_resources_from_hud([], force_delay=0.1)

        assert calls[0] == ("sleep", 0.1)
        assert calls[1][0] == "grab"
