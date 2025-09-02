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

# Stub OpenCV before importing resources
sys.modules.setdefault(
    "cv2",
    types.SimpleNamespace(
        cvtColor=lambda src, code: src,
        resize=lambda img, *a, **k: img,
        threshold=lambda img, *a, **k: (None, img),
        imread=lambda *a, **k: np.zeros((1, 1), dtype=np.uint8),
        imwrite=lambda *a, **k: True,
        IMREAD_GRAYSCALE=0,
        COLOR_BGR2GRAY=0,
        INTER_LINEAR=0,
        THRESH_BINARY=0,
        THRESH_OTSU=0,
    ),
)

# Avoid invoking external tesseract binary
os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.resources as resources


class TestPopulationOcrConfidence(TestCase):
    def test_negative_confidences_raise_error(self):
        roi = np.zeros((10, 10, 3), dtype=np.uint8)
        with patch.dict(resources.common.CFG, {"allow_low_conf_population": False}, clear=False), patch(
            "script.resources.ocr.executor.execute_ocr",
            return_value=(
                "34",
                {"text": ["3", "4"], "conf": ["-1", "0", "95"]},
                None,
                True,
            ),
        ):
            with self.assertRaises(resources.common.PopulationReadError):
                resources._read_population_from_roi(
                    roi, conf_threshold=60
                )

    def test_fraction_patterns_extracted(self):
        roi = np.zeros((10, 10, 3), dtype=np.uint8)
        samples = [("3/4", (3, 4)), ("13/14", (13, 14))]
        for text, expected in samples:
            with self.subTest(text=text), patch(
                "script.resources.ocr.executor.execute_ocr",
                return_value=(text, {"text": [text], "conf": ["95"]}, None, False),
            ):
                cur, cap = resources._read_population_from_roi(
                    roi, conf_threshold=60
                )
                self.assertEqual((cur, cap), expected)

    def test_low_confidence_allowed(self):
        roi = np.zeros((10, 10, 3), dtype=np.uint8)
        with patch.dict(resources.common.CFG, {"allow_low_conf_population": True}, clear=False), \
            patch(
                "script.resources.ocr.executor.execute_ocr",
                return_value=(
                    "12/34",
                    {"text": ["12/34"], "conf": ["40", "40"]},
                    None,
                    True,
                ),
            ):
            cur, cap = resources._read_population_from_roi(
                roi, conf_threshold=60
            )
            self.assertEqual((cur, cap), (12, 34))

    def test_low_confidence_fallback_after_attempts(self):
        roi = np.zeros((10, 10, 3), dtype=np.uint8)

        with patch.dict(
            resources.common.CFG,
            {
                "population_limit_low_conf_fallback": True,
                "ocr_retry_limit": 3,
                "allow_low_conf_population": False,
            },
            clear=False,
        ), patch(
            "script.resources.ocr.executor.execute_ocr",
            return_value=(
                "12/34",
                {"text": ["12/34"], "conf": ["40", "40"]},
                None,
                True,
            ),
        ):
            for fc in (0, 1):
                with self.assertRaises(resources.common.PopulationReadError):
                    resources._read_population_from_roi(
                        roi, conf_threshold=60, failure_count=fc
                    )
            cur, cap = resources._read_population_from_roi(
                roi, conf_threshold=60, failure_count=2
            )
            self.assertEqual((cur, cap), (12, 34))

    def test_low_conf_message_and_fallback(self):
        roi = np.zeros((10, 10, 3), dtype=np.uint8)

        with patch.dict(
            resources.common.CFG,
            {
                "population_limit_low_conf_fallback": True,
                "ocr_retry_limit": 2,
                "allow_low_conf_population": False,
            },
            clear=False,
        ), patch(
            "script.resources.ocr.executor.execute_ocr",
            return_value=(
                "12/34",
                {"text": ["12/34"], "conf": ["40", "40"]},
                None,
                True,
            ),
        ):
            with self.assertRaises(resources.common.PopulationReadError) as ctx:
                resources._read_population_from_roi(
                    roi, conf_threshold=60, failure_count=0
                )
            msg = str(ctx.exception)
            self.assertIn("text='12/34'", msg)
            self.assertIn("confs=[40.0, 40.0]", msg)
            cur, cap = resources._read_population_from_roi(
                roi, conf_threshold=60, failure_count=1
            )
            self.assertEqual((cur, cap), (12, 34))

    def test_zero_confidence_requires_flag(self):
        roi = np.zeros((10, 10, 3), dtype=np.uint8)
        with patch.dict(resources.common.CFG, {"allow_low_conf_population": False}, clear=False), patch(
            "script.resources.ocr.executor.execute_ocr",
            return_value=(
                "12/34",
                {"text": ["12/34"], "conf": ["0", "0"], "zero_conf": True},
                None,
                True,
            ),
        ):
            with self.assertRaises(resources.common.PopulationReadError):
                resources._read_population_from_roi(roi, conf_threshold=60)

    def test_zero_confidence_allowed_with_fallback_flag(self):
        roi = np.zeros((10, 10, 3), dtype=np.uint8)
        with patch.dict(
            resources.common.CFG,
            {
                "population_limit_low_conf_fallback": True,
                "allow_low_conf_population": False,
            },
            clear=False,
        ), patch(
            "script.resources.ocr.executor.execute_ocr",
            return_value=(
                "12/34",
                {"text": ["12/34"], "conf": ["0", "0"], "zero_conf": True},
                None,
                True,
            ),
        ):
            cur, cap = resources._read_population_from_roi(roi, conf_threshold=60)
        self.assertEqual((cur, cap), (12, 34))
