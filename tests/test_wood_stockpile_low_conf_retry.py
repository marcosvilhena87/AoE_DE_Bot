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
import script.resources as resources

class TestWoodStockpileLowConfRetry(TestCase):
    def setUp(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources._LAST_READ_FROM_CACHE.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()

    def tearDown(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources._LAST_READ_FROM_CACHE.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()

    def test_low_conf_retry_accepted(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        def fake_detect(frame, required_icons, cache=None):
            return {"wood_stockpile": (0, 0, 10, 10)}

        calls = []
        def fake_execute(gray, conf_threshold=None, allow_fallback=True, roi=None, resource=None):
            calls.append(conf_threshold)
            return "999", {"conf": [conf_threshold or 0]}, None, True

        with patch("script.resources.detect_resource_regions", side_effect=fake_detect), \
             patch("script.screen_utils._grab_frame", return_value=frame), \
             patch("script.resources.preprocess_roi", side_effect=lambda roi: roi[..., 0]), \
             patch("script.resources.execute_ocr", side_effect=fake_execute), \
             patch("script.resources.cv2.imwrite"), \
             patch.dict(resources.CFG, {"treat_low_conf_as_failure": True}, clear=False):
            result, _ = resources.read_resources_from_hud(["wood_stockpile"])

        self.assertEqual(result["wood_stockpile"], 999)
        self.assertIn("wood_stockpile", resources._LAST_LOW_CONFIDENCE)
        self.assertEqual(
            calls,
            [
                resources.CFG.get(
                    "wood_stockpile_ocr_conf_threshold",
                    resources.CFG.get("ocr_conf_threshold", 60),
                ),
                resources.CFG.get("ocr_conf_min", 0),
            ],
        )
