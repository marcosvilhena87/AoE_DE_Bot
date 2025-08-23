import os
import sys
import types
import re
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

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.common as common
import script.resources as resources


class TestOcrConfig(TestCase):
    def test_custom_kernel_and_psm_list(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        kernels = []
        psms = []

        def fake_dilate(src, kernel, iterations=1):
            kernels.append(kernel.shape)
            return src

        def fake_image_to_data(image, config="", output_type=None):
            m = re.search(r"--psm (\d+)", config)
            if m:
                psms.append(int(m.group(1)))
            return {"text": ["0"]}

        custom_cfg = {**common.CFG, "ocr_kernel_size": 3, "ocr_psm_list": [4, 5]}

        with patch.object(resources, "CFG", custom_cfg), \
             patch("script.resources.cv2.dilate", side_effect=fake_dilate), \
             patch("script.resources.pytesseract.image_to_data", side_effect=fake_image_to_data):
            resources._ocr_digits_better(gray)

        self.assertIn((3, 3), kernels)
        self.assertEqual(sorted(set(psms)), [4, 5])

