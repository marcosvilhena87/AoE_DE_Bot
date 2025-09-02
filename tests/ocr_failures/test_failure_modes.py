import numpy as np
import pytest
from unittest.mock import patch

import script.resources.reader as resources


def test_read_resources_returns_none_on_empty_ocr():
    def fake_ocr(gray):
        return "", {"text": [""]}, np.zeros((1, 1), dtype=np.uint8)

    frame = np.zeros((600, 600, 3), dtype=np.uint8)
    resources.RESOURCE_CACHE.last_resource_values["wood_stockpile"] = 0
    with patch.dict(resources.CFG, {"wood_stockpile_low_conf_fallback": False}, clear=False), \
         patch(
             "script.resources.reader.detect_resource_regions",
             return_value={
                 "wood_stockpile": (0, 0, 50, 50),
                 "food_stockpile": (50, 0, 50, 50),
                 "gold_stockpile": (100, 0, 50, 50),
                 "stone_stockpile": (150, 0, 50, 50),
                 "population_limit": (200, 0, 50, 50),
                 "idle_villager": (250, 0, 50, 50),
             },
         ), patch("script.resources.ocr.masks._ocr_digits_better", side_effect=fake_ocr), \
         patch(
             "script.resources.reader.pytesseract.image_to_data",
             return_value={"text": [""], "conf": ["0"]},
         ), patch(
             "script.resources.reader.pytesseract.image_to_string", return_value="123"
         ), patch(
             "script.resources.ocr.executor._read_population_from_roi",
             return_value=(0, 0),
         ):
        icons = resources.RESOURCE_ICON_ORDER[:-1]
        result, _ = resources._read_resources(frame, icons, icons)
    assert result["wood_stockpile"] is None


def test_read_resources_handles_missing_non_required_icons():
    def fake_grab_frame(bbox=None):
        if bbox:
            return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
        return np.zeros((600, 600, 3), dtype=np.uint8)

    def fake_detect(frame, required_icons, cache=None):
        return {
            "wood_stockpile": (0, 0, 50, 50),
            "food_stockpile": (50, 0, 50, 50),
        }

    ocr_seq = [
        ("123", {"text": ["123"]}, np.zeros((1, 1), dtype=np.uint8)),
        ("", {"text": [""]}, np.zeros((1, 1), dtype=np.uint8)),
    ]

    def fake_ocr(gray):
        return ocr_seq.pop(0)

    with patch("script.resources.reader.detect_resource_regions", side_effect=fake_detect), \
         patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
         patch("script.resources.ocr.masks._ocr_digits_better", side_effect=fake_ocr), \
         patch("script.resources.reader.pytesseract.image_to_string", return_value=""), \
         patch("script.resources.reader.cv2.imwrite"):
        result, _ = resources.read_resources_from_hud(["wood_stockpile"])
    assert result.get("wood_stockpile") == 123
    assert result.get("food_stockpile") is None


def test_discard_low_confidence_without_fallback():
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
         patch("script.resources.ocr.masks._ocr_digits_better", side_effect=fake_ocr) as ocr_mock, \
         patch("script.resources.reader.pytesseract.image_to_string", return_value="") as img2str_mock, \
         patch("script.resources.reader.cv2.imwrite"):
        result, _ = resources.read_resources_from_hud(["wood_stockpile"])
    assert result["wood_stockpile"] is None
    assert ocr_mock.call_count >= 1
    img2str_mock.assert_not_called()


def test_low_confidence_triggers_fallback():
    def fake_grab_frame(bbox=None):
        if bbox:
            return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
        return np.zeros((600, 600, 3), dtype=np.uint8)

    def fake_detect(frame, required_icons, cache=None):
        return {"wood_stockpile": (0, 0, 50, 50)}

    ocr_seq = [
        ("7", {"text": ["7"], "conf": ["10"]}, np.zeros((1, 1), dtype=np.uint8)),
        ("", {"text": [""], "conf": [""]}, np.zeros((1, 1), dtype=np.uint8)),
        ("", {"text": [""], "conf": [""]}, np.zeros((1, 1), dtype=np.uint8)),
    ]

    def fake_ocr(gray):
        return ocr_seq.pop(0) if ocr_seq else ("", {"text": [""], "conf": [""]}, np.zeros((1, 1), dtype=np.uint8))

    resources.RESOURCE_CACHE.last_resource_values["wood_stockpile"] = 0
    with patch.dict(resources.CFG, {"wood_stockpile_low_conf_fallback": False}, clear=False), \
         patch("script.resources.reader.detect_resource_regions", side_effect=fake_detect), \
         patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
         patch("script.resources.ocr.masks._ocr_digits_better", side_effect=fake_ocr) as ocr_mock, \
         patch("script.resources.reader.pytesseract.image_to_string", return_value="") as img2str_mock, \
         patch("script.resources.reader.cv2.imwrite"):
        result, _ = resources.read_resources_from_hud(["wood_stockpile"])
    assert result["wood_stockpile"] is None
    assert ocr_mock.call_count >= 1
    img2str_mock.assert_called()


def test_confidence_zero_does_not_call_image_to_string():
    def fake_grab_frame(bbox=None):
        if bbox:
            return np.zeros((bbox["height"], bbox["width"], 3), dtype=np.uint8)
        return np.zeros((600, 600, 3), dtype=np.uint8)

    def fake_detect(frame, required_icons, cache=None):
        return {"wood_stockpile": (0, 0, 50, 50)}

    def fake_ocr(gray):
        data = {"text": ["7"], "conf": ["0"]}
        return "7", data, np.zeros((1, 1), dtype=np.uint8)

    resources.RESOURCE_CACHE.last_resource_values["wood_stockpile"] = 0
    with patch.dict(resources.CFG, {"wood_stockpile_low_conf_fallback": False, "allow_zero_confidence_digits": False}, clear=False), \
         patch("script.resources.reader.detect_resource_regions", side_effect=fake_detect), \
         patch("script.screen_utils._grab_frame", side_effect=fake_grab_frame), \
         patch("script.resources.ocr.masks._ocr_digits_better", side_effect=fake_ocr), \
         patch("script.resources.reader.pytesseract.image_to_string", return_value="") as img2str_mock, \
         patch("script.resources.reader.cv2.imwrite"):
        result, _ = resources.read_resources_from_hud(["wood_stockpile"])
    assert result["wood_stockpile"] is None
    img2str_mock.assert_not_called()


def test_missing_optional_icon_not_in_results():
    def fake_detect(frame, required_icons, cache=None):
        return {
            "wood_stockpile": (0, 0, 50, 50),
            "food_stockpile": (50, 0, 50, 50),
        }

    ocr_seq = [
        ("123", {"text": ["123"]}, np.zeros((1, 1), dtype=np.uint8)),
        ("234", {"text": ["234"]}, np.zeros((1, 1), dtype=np.uint8)),
        ("345", {"text": ["345"]}, np.zeros((1, 1), dtype=np.uint8)),
        ("", {"text": [""]}, np.zeros((1, 1), dtype=np.uint8)),
    ]

    def fake_ocr(gray):
        return ocr_seq.pop(0)

    frame = np.zeros((600, 600, 3), dtype=np.uint8)
    with patch("script.resources.reader.detect_resource_regions", side_effect=fake_detect), \
         patch("script.screen_utils._grab_frame", return_value=frame), \
         patch("script.resources.ocr.masks._ocr_digits_better", side_effect=fake_ocr), \
         patch("script.resources.reader.pytesseract.image_to_string", return_value=""), \
         patch("script.resources.reader.cv2.imwrite"):
        first, _ = resources.read_resources_from_hud(["wood_stockpile"])
        second, _ = resources.read_resources_from_hud(["wood_stockpile"])
    assert "food_stockpile" not in first
    assert "food_stockpile" not in second


def test_zero_variance_returns_zero():
    def make_gold_roi():
        roi = np.full((10, 10), 210, dtype=np.uint8)
        roi[2:-2, 2] = 200
        roi[2:-2, -3] = 200
        roi[2, 2:-2] = 200
        roi[-3, 2:-2] = 200
        return roi

    def make_stone_roi():
        roi = np.full((10, 10), 180, dtype=np.uint8)
        roi[2:-2, 2] = 170
        roi[2:-2, -3] = 170
        roi[2, 2:-2] = 170
        roi[-3, 2:-2] = 170
        return roi

    with patch(
        "script.resources.reader.pytesseract.image_to_data",
        return_value={"text": [""], "conf": ["-1"]},
    ), patch.dict(resources.CFG, {"ocr_zero_variance": 50}, clear=False):
        gold, _, _ = resources._ocr_digits_better(make_gold_roi())
        stone, _, _ = resources._ocr_digits_better(make_stone_roi())
    assert gold == "0"
    assert stone == "0"


def test_low_conf_required_without_cache_raises():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    regions = {"wood_stockpile": (0, 0, 10, 10)}
    results = {"wood_stockpile": None}
    cache = resources.ResourceCache()
    with patch("script.resources.ocr.executor.cv2.imwrite"), pytest.raises(
        resources.common.ResourceReadError
    ):
        resources.handle_ocr_failure(
            frame,
            regions,
            results,
            ["wood_stockpile"],
            cache_obj=cache,
            retry_limit=1,
            low_confidence={"wood_stockpile"},
        )


def test_low_conf_optional_without_cache_keeps_none():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    regions = {"wood_stockpile": (0, 0, 10, 10)}
    results = {"wood_stockpile": None}
    cache = resources.ResourceCache()
    with patch("script.resources.ocr.executor.cv2.imwrite"):
        resources.handle_ocr_failure(
            frame,
            regions,
            results,
            [],
            cache_obj=cache,
            retry_limit=1,
            low_confidence={"wood_stockpile"},
        )
    assert results["wood_stockpile"] is None


