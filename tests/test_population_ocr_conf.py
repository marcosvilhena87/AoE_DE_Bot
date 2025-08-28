import os
import sys
import types
from unittest import TestCase
from unittest.mock import patch

import numpy as np

# Stub OpenCV before importing resources
sys.modules.setdefault(
    "cv2",
    types.SimpleNamespace(
        cvtColor=lambda src, code: src,
        resize=lambda img, *a, **k: img,
        threshold=lambda img, *a, **k: (None, img),
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
    def test_negative_confidences_are_ignored(self):
        roi = np.zeros((10, 10, 3), dtype=np.uint8)
        with patch(
            "script.resources.pytesseract.image_to_data",
            return_value={"text": ["3/4"], "conf": ["-1", "-1", "95"]},
        ):
            cur, cap = resources._read_population_from_roi(
                roi, conf_threshold=60, save_debug=False
            )
        self.assertEqual((cur, cap), (3, 4))
