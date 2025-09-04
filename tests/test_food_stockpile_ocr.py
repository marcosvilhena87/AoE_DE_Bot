import os
import sys
import time
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
import script.resources.reader as resources


class TestFoodStockpileCacheTolerance(TestCase):
    def setUp(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()

    def tearDown(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()

    def test_stale_cache_ignored_on_large_delta(self):
        cache = resources.ResourceCache()
        cache.last_resource_values["food_stockpile"] = 900
        cache.last_resource_ts["food_stockpile"] = time.time() - 10

        with patch.dict(
            resources.CFG,
            {
                "food_stockpile_low_conf_fallback": True,
                "resource_cache_tolerance": 50,
                "starting_resources": {"food_stockpile": 80},
            },
            clear=False,
        ):
            value, cache_hit, low_conf, no_digit = resources.core._handle_cache_and_fallback(
                "food_stockpile",
                "80",
                True,
                {"text": ["80"]},
                np.zeros((1, 1, 3), dtype=np.uint8),
                None,
                0,
                cache_obj=cache,
                max_cache_age=None,
                low_conf_counts={},
            )

        self.assertEqual(value, 80)
        self.assertFalse(cache_hit)
        self.assertFalse(low_conf)
        self.assertEqual(cache.last_resource_values["food_stockpile"], 80)
