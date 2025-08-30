import pytest

from script.resources.ocr.confidence import parse_confidences


def test_clamps_negative_and_preserves_zero():
    data = {"conf": ["42.5", "-1", "0", "abc", "77"]}
    assert parse_confidences(data) == [42.5, 0.0, 0.0, 77.0]


def test_handles_missing_conf_key():
    assert parse_confidences({}) == []
    assert parse_confidences({"conf": ["foo", None]}) == []
