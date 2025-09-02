import numpy as np
from unittest.mock import patch

import script.resources as resources


def test_negative_confidences_flagged_low_confidence():
    gray = np.zeros((5, 5), dtype=np.uint8)
    data = {"text": ["12"], "conf": ["-5", "0"]}
    with patch("script.resources.ocr.masks._ocr_digits_better", return_value=("12", data, None)), \
         patch("script.resources.ocr.executor.pytesseract.image_to_string", return_value=""):
        digits, _, _, low_conf = resources.execute_ocr(
            gray, conf_threshold=60, resource="wood_stockpile"
        )
    assert digits == "12"
    assert low_conf


def test_negative_conf_preceding_digits_not_low_confidence():
    gray = np.zeros((5, 5), dtype=np.uint8)
    data = {"text": ["foo", "12"], "conf": ["-5", "80"]}
    with patch(
        "script.resources.ocr.masks._ocr_digits_better", return_value=("12", data, None)
    ), patch(
        "script.resources.ocr.executor.pytesseract.image_to_string", return_value=""
    ):
        digits, _, _, low_conf = resources.execute_ocr(
            gray, conf_threshold=60, resource="wood_stockpile"
        )
    assert digits == "12"
    assert not low_conf
