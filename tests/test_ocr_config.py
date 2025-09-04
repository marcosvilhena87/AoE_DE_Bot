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
common.init_common()
import script.resources as resources
from script.resources.ocr import masks
from script.resources.ocr.masks import _ocr_digits_better


class TestOcrConfig(TestCase):
    def test_custom_kernel_and_psm_list(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        gray[0, 0] = 255
        kernels = []
        psms = []
        expected_psms = list(dict.fromkeys([4, 5] + [6, 7, 8, 10, 13]))

        def fake_dilate(src, kernel, iterations=1):
            kernels.append(kernel.shape)
            return src

        call_count = [0]

        def fake_image_to_data(image, config="", output_type=None):
            call_count[0] += 1
            if call_count[0] <= len(expected_psms) * 2:
                return {"text": [""]}
            m = re.search(r"--psm (\d+)", config)
            if m:
                psms.append(int(m.group(1)))
            return {"text": ["0"]}

        custom_cfg = {**common.CFG, "ocr_kernel_size": 3, "ocr_psm_list": [4, 5]}

        with patch.object(resources, "CFG", custom_cfg), \
             patch.object(masks, "CFG", custom_cfg), \
             patch("script.resources.ocr.masks.cv2.dilate", side_effect=fake_dilate), \
             patch("pytesseract.image_to_data", side_effect=fake_image_to_data):
            _ocr_digits_better(gray)

        self.assertIn((3, 3), kernels)
        self.assertEqual(sorted(set(psms)), sorted(expected_psms))

    def test_per_resource_psm_override_reduces_calls(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        gray[0, 0] = 255

        def make_fake(counter):
            def fake_image_to_data(image, config="", output_type=None):
                counter[0] += 1
                return {"text": ["5"], "conf": ["80"]}
            return fake_image_to_data

        # Baseline using global PSM list
        baseline_counter = [0]
        with patch(
            "pytesseract.image_to_data",
            side_effect=make_fake(baseline_counter),
        ):
            digits_default, _data, _mask = _ocr_digits_better(
                gray, resource="wood_stockpile"
            )

        # Override PSM list for wood_stockpile
        override_counter = [0]
        custom_cfg = {**resources.CFG, "wood_stockpile_ocr_psm_list": [7]}
        with patch.object(resources, "CFG", custom_cfg), patch.object(
            masks, "CFG", custom_cfg
        ), patch(
            "pytesseract.image_to_data",
            side_effect=make_fake(override_counter),
        ):
            digits_override, _data2, _mask2 = _ocr_digits_better(
                gray, resource="wood_stockpile"
            )

        self.assertEqual(digits_default, digits_override)
        self.assertGreater(baseline_counter[0], override_counter[0])

