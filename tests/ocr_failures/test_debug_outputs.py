import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import script.resources.reader as resources


def test_discard_low_confidence_logged(caplog):
    def fake_grab_frame(bbox=None):
        if bbox:
            return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
        return np.zeros((600, 600, 3), dtype=np.uint8)

    def fake_detect(frame, required_icons, cache=None):
        return {"wood_stockpile": (0, 0, 50, 50)}

    def fake_ocr(gray):
        data = {"text": ["123"], "conf": ["10", "20", "30"]}
        return "123", data, np.zeros((1, 1), dtype=np.uint8)

    resources.RESOURCE_CACHE.last_resource_values["wood_stockpile"] = 0
    with patch.dict(resources.CFG, {"wood_stockpile_low_conf_fallback": False}, clear=False), \
         patch("script.resources.reader.detect_resource_regions", side_effect=fake_detect), \
         patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
         patch("script.resources.ocr.masks._ocr_digits_better", side_effect=fake_ocr), \
         patch("script.resources.reader.pytesseract.image_to_string", return_value=""), \
         patch("script.resources.reader.cv2.imwrite"), \
         caplog.at_level("INFO", logger=resources.logger.name):
        result, _ = resources.read_resources_from_hud(["wood_stockpile"])
    assert result["wood_stockpile"] is None
    logs = "\n".join(caplog.messages)
    assert "Discarding wood_stockpile=123 due to low-confidence OCR" in logs
    assert "Detected wood_stockpile=123" not in logs


def test_narrow_roi_failure_includes_note():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    regions = {"wood_stockpile": (0, 0, 10, 10)}
    results = {"wood_stockpile": None}
    with patch("script.resources.reader.cv2.imwrite"), \
         patch("script.resources.ocr.executor.logger.error") as err_mock, \
         patch("script.resources.reader.pytesseract.pytesseract.tesseract_cmd", "/usr/bin/true"), \
         patch.object(resources.cache, "_NARROW_ROIS", {"wood_stockpile"}):
        resources.handle_ocr_failure(frame, regions, results, ["wood_stockpile"])
    assert "narrow ROI span" in err_mock.call_args[0][1]


def test_fallback_failure_saves_debug_images():
    gray = np.zeros((5, 5), dtype=np.uint8)
    mask = np.ones((5, 5), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("script.resources.ocr.masks._ocr_digits_better", return_value=("", {"text": [""]}, mask)), \
         patch("script.resources.reader.pytesseract.image_to_string", return_value=""), \
         patch("script.resources.reader.cv2.imwrite") as imwrite_mock, \
         patch("script.resources.ocr.executor.ROOT", Path(tmpdir)):
        digits, data, mask_out, low_conf = resources.execute_ocr(gray)
    assert digits == ""
    assert low_conf
    assert mask_out is mask
    imgs = [call.args[1] for call in imwrite_mock.call_args_list]
    assert any(np.array_equal(img, mask) for img in imgs)
    assert any(np.array_equal(img, gray) for img in imgs)
