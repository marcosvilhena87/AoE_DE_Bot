import pytest

from script.resources.ocr import parse_confidences


def test_filters_invalid_and_non_positive_values():
    data = {"conf": ["42.5", "-1", "0", "abc", "77"]}
    assert parse_confidences(data) == [42.5, 77.0]


def test_handles_missing_conf_key():
    assert parse_confidences({}) == []
    assert parse_confidences({"conf": ["foo", None]}) == []
