import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np


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

dummy_cv2 = types.SimpleNamespace(
    cvtColor=lambda src, code: src,
    resize=lambda img, *a, **k: img,
    matchTemplate=lambda *a, **k: np.zeros((1, 1), dtype=np.float32),
    minMaxLoc=lambda *a, **k: (0, 0, (0, 0), (0, 0)),
    imread=lambda *a, **k: np.zeros((1, 1), dtype=np.uint8),
    imwrite=lambda *a, **k: True,
    medianBlur=lambda src, k: src,
    bitwise_not=lambda src: src,
    threshold=lambda src, *a, **k: (None, src),
    rectangle=lambda img, pt1, pt2, color, thickness: img,
    IMREAD_GRAYSCALE=0,
    COLOR_BGR2GRAY=0,
)
sys.modules.setdefault("cv2", dummy_cv2)

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.resources.reader as reader


class TestPopulationIdleRetry(TestCase):
    def setUp(self):
        reader.RESOURCE_CACHE.last_resource_values.clear()
        reader.RESOURCE_CACHE.last_resource_ts.clear()
        reader.RESOURCE_CACHE.resource_failure_counts.clear()

    def test_retry_when_counts_exceed_cap(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        first = ({"idle_villager": 120, "population_limit": 120}, (120, 100))
        second = ({"idle_villager": 5, "population_limit": 5}, (5, 100))

        with patch(
            "script.resources.reader.core._read_resources",
            side_effect=[first, second],
        ) as read_mock, patch(
            "script.resources.reader.core.screen_utils._grab_frame",
            return_value=frame,
        ), self.assertLogs(reader.logger, level="WARNING") as cm:
            results, pop = reader.gather_hud_stats()

        self.assertEqual(read_mock.call_count, 2)
        self.assertEqual(results["idle_villager"], 5)
        self.assertEqual(pop, (5, 100))
        self.assertTrue(any("retrying OCR" in msg for msg in cm.output))
