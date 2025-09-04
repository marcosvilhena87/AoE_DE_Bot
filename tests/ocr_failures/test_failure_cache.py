import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import time
import re

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
sys.modules.setdefault(
    "cv2",
    types.SimpleNamespace(
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
        INTER_LINEAR=0,
        THRESH_BINARY=0,
        THRESH_OTSU=0,
        TM_CCOEFF_NORMED=0,
    ),
)

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import script.common as common
import script.resources.reader as resources


class TestFailureCache(TestCase):
    def setUp(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources._LAST_READ_FROM_CACHE.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()
        resources._LAST_REGION_SPANS.clear()

    def tearDown(self):
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources._LAST_READ_FROM_CACHE.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()
        resources._LAST_REGION_SPANS.clear()

    def test_required_icon_fallback(self):
        def fake_detect(frame, required_icons, cache=None):
            return {"wood_stockpile": (0, 0, 50, 50)}

        ocr_seq = [
            ("123", {"text": ["123"]}, None, False),
        ] + [
            ("", {"text": [""]}, None, False)
            for _ in range(20)
        ]

        def fake_ocr(gray, conf_threshold=None, allow_fallback=True, roi=None, resource=None):
            return ocr_seq.pop(0) if ocr_seq else ("", {"text": [""]}, None, False)

        frame = np.zeros((600, 600, 3), dtype=np.uint8)

        with patch("script.resources.reader.detect_resource_regions", side_effect=fake_detect), \
            patch("script.screen_utils.grab_frame", return_value=frame), \
             patch("script.resources.reader.execute_ocr", side_effect=fake_ocr), \
             patch("script.resources.reader.pytesseract.image_to_string", return_value=""), \
             patch("script.resources.reader.cv2.imwrite"):
            first, _ = resources.read_resources_from_hud(["wood_stockpile"])
            second, _ = resources.read_resources_from_hud(["wood_stockpile"])

        self.assertEqual(first["wood_stockpile"], 123)
        self.assertEqual(second["wood_stockpile"], 123)
        self.assertIn("wood_stockpile", resources._LAST_READ_FROM_CACHE)

    def test_retry_succeeds_after_expansion(self):
        def fake_detect(frame, required_icons, cache=None):
            return {"wood_stockpile": (0, 0, 50, 50)}

        ocr_seq = [
            ("", {"text": [""]}, None, False),
            ("456", {"text": ["456"]}, None, False),
        ]

        def fake_ocr(gray, conf_threshold=None, allow_fallback=True, roi=None, resource=None):
            return ocr_seq.pop(0)

        frame = np.zeros((600, 600, 3), dtype=np.uint8)

        with patch("script.resources.reader.detect_resource_regions", side_effect=fake_detect), \
            patch("script.screen_utils.grab_frame", return_value=frame), \
             patch("script.resources.reader.execute_ocr", side_effect=fake_ocr) as ocr_mock, \
             patch("script.resources.reader.pytesseract.image_to_string", return_value=""), \
             patch("script.resources.reader.cv2.imwrite"):
            result, _ = resources.read_resources_from_hud(["wood_stockpile"])

        self.assertEqual(result["wood_stockpile"], 456)
        self.assertEqual(ocr_mock.call_count, 2)

    def test_sliding_window_succeeds_when_anchor_right(self):
        def fake_detect(frame, required_icons, cache=None):
            return {"wood_stockpile": (10, 0, 50, 20)}

        frame = np.tile(np.arange(120, dtype=np.uint8), (20, 1))
        frame = np.stack([frame] * 3, axis=-1)

        calls = []

        def fake_execute(
            gray, conf_threshold=None, allow_fallback=True, roi=None, resource=None
        ):
            h, w = gray.shape
            mean = gray.mean()
            x = int(round(mean - (w - 1) / 2))
            calls.append((x, w, allow_fallback))
            if mean > 80:
                return "789", {"text": ["789"]}, None, False
            return "", {"text": [""]}, None, False

        with patch("script.resources.reader.detect_resource_regions", side_effect=fake_detect), \
            patch("script.screen_utils.grab_frame", return_value=frame), \
             patch("script.resources.reader.preprocess_roi", side_effect=lambda roi: roi[..., 0]), \
             patch("script.resources.reader.execute_ocr", side_effect=fake_execute), \
             patch.dict(resources._LAST_REGION_SPANS, {"wood_stockpile": (0, 120)}, clear=True), \
             patch("script.resources.reader.pytesseract.image_to_string", return_value=""):
            result, _ = resources.read_resources_from_hud(["wood_stockpile"])

        self.assertEqual(result["wood_stockpile"], 789)
        self.assertEqual(len(calls), 5)
        self.assertEqual(len({(x, w) for x, w, _ in calls}), len(calls))
        self.assertEqual([a for _, _, a in calls], [True, True, False, False, False])

    def test_expired_cache_used_after_consecutive_failures(self):
        def fake_detect(frame, required_icons, cache=None):
            return {"wood_stockpile": (0, 0, 50, 50)}

        def fake_ocr(gray, conf_threshold=None, allow_fallback=True, roi=None, resource=None):
            return "", {"text": [""]}, None, False

        frame = np.zeros((600, 600, 3), dtype=np.uint8)

        resources.RESOURCE_CACHE.last_resource_values["wood_stockpile"] = 999
        resources.RESOURCE_CACHE.last_resource_ts["wood_stockpile"] = (
            time.time() - (resources._RESOURCE_CACHE_TTL + 10)
        )

        with patch("script.resources.reader.detect_resource_regions", side_effect=fake_detect), \
            patch("script.screen_utils.grab_frame", return_value=frame), \
             patch("script.resources.reader.execute_ocr", side_effect=fake_ocr), \
             patch("script.resources.reader.pytesseract.image_to_string", return_value=""), \
             patch("script.resources.reader.cv2.imwrite"):
            with self.assertRaises(common.ResourceReadError):
                resources.read_resources_from_hud(["wood_stockpile"])
            result, _ = resources.read_resources_from_hud(["wood_stockpile"])

        self.assertEqual(result["wood_stockpile"], 999)
        self.assertIn("wood_stockpile", resources._LAST_READ_FROM_CACHE)

    def test_default_value_returned_after_threshold(self):
        def fake_detect(frame, required_icons, cache=None):
            return {"wood_stockpile": (0, 0, 50, 50)}

        def fake_ocr(gray, conf_threshold=None, allow_fallback=True, roi=None, resource=None):
            return "", {"text": [""]}, None, False

        frame = np.zeros((600, 600, 3), dtype=np.uint8)

        with patch.dict(resources.CFG, {"ocr_retry_limit": 2}, clear=False), \
             patch("script.resources.reader.detect_resource_regions", side_effect=fake_detect), \
            patch("script.screen_utils.grab_frame", return_value=frame), \
             patch("script.resources.reader.execute_ocr", side_effect=fake_ocr), \
             patch("script.resources.reader.pytesseract.image_to_string", return_value=""), \
             patch("script.resources.reader.cv2.imwrite"):
            with self.assertRaises(common.ResourceReadError):
                resources.read_resources_from_hud(["wood_stockpile"])
            result, _ = resources.read_resources_from_hud(["wood_stockpile"])

        self.assertEqual(result["wood_stockpile"], 0)

    def test_expand_increases_after_consecutive_failures(self):
        def fake_detect(frame, required_icons, cache=None):
            return {"wood_stockpile": (10, 10, 50, 50)}

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        rois = []

        def fake_execute(
            gray, conf_threshold=None, allow_fallback=True, roi=None, resource=None
        ):
            if roi is not None:
                rois.append((roi, allow_fallback))
            return "", {"text": [""]}, np.zeros((1, 1), dtype=np.uint8), False

        with patch("script.resources.reader.detect_resource_regions", side_effect=fake_detect), \
            patch("script.screen_utils.grab_frame", return_value=frame), \
             patch("script.resources.reader.preprocess_roi", side_effect=lambda r: r[..., 0] if r.ndim == 3 else r), \
             patch("script.resources.reader.execute_ocr", side_effect=fake_execute), \
             patch("script.resources.reader.pytesseract.image_to_string", return_value=""), \
             patch("script.resources.reader.handle_ocr_failure"), \
             patch("script.resources.reader.cv2.imwrite"), \
             self.assertLogs("script.resources", level="DEBUG") as cm:
            resources.read_resources_from_hud([], ["wood_stockpile"])
            resources.read_resources_from_hud([], ["wood_stockpile"])

        expanded = [r for r, allow in rois if allow and r[2] > 50]
        self.assertEqual(len(expanded), 2)
        self.assertLess(expanded[1][0], expanded[0][0])
        self.assertGreater(expanded[1][2], expanded[0][2])

        widths = []
        for line in cm.output:
            if "Expanding ROI for wood_stockpile" in line:
                m = re.search(r"w=(\d+)", line)
                if m:
                    widths.append(int(m.group(1)))
        self.assertEqual(len(widths), 2)
        self.assertLess(widths[0], widths[1])

    def test_cached_value_used_for_optional_failure(self):
        def fake_detect(frame, required_icons, cache=None):
            return {
                "wood_stockpile": (0, 0, 50, 50),
                "food_stockpile": (50, 0, 50, 50),
            }

        ocr_seq = [
            ("123", {"text": ["123"]}, None, False),
            ("234", {"text": ["234"]}, None, False),
            ("345", {"text": ["345"]}, None, False),
            ("", {"text": [""]}, None, False),
        ]

        def fake_ocr(gray, conf_threshold=None, allow_fallback=True, roi=None, resource=None):
            return ocr_seq.pop(0)

        frame = np.zeros((600, 600, 3), dtype=np.uint8)

        with patch("script.resources.reader.detect_resource_regions", side_effect=fake_detect), \
            patch("script.screen_utils.grab_frame", return_value=frame), \
             patch("script.resources.reader.execute_ocr", side_effect=fake_ocr), \
             patch("script.resources.reader.pytesseract.image_to_string", return_value=""), \
             patch("script.resources.reader.cv2.imwrite"):
            first, _ = resources.read_resources_from_hud(["wood_stockpile"])
            second, _ = resources.read_resources_from_hud(["wood_stockpile"])

        self.assertNotIn("food_stockpile", first)
        self.assertNotIn("food_stockpile", second)
