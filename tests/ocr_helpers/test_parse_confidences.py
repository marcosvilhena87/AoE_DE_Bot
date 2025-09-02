import logging
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from script.resources.ocr.confidence import parse_confidences
from script.resources.reader import core


def test_clamps_negative_and_excludes_zero():
    data = {"conf": ["42.5", "-1", "0", "abc", "77"]}
    assert parse_confidences(data) == [42.5, 77.0]


def test_handles_missing_conf_key():
    assert parse_confidences({}) is None
    assert parse_confidences({"conf": ["foo", None]}) is None


def test_read_resources_logs_sanitized_confidences(caplog):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    gray = np.zeros((10, 10), dtype=np.uint8)
    data = {"text": ["12"], "conf": ["-5", "50"]}
    cache_obj = SimpleNamespace(
        last_resource_values={},
        last_resource_ts={},
        resource_failure_counts={},
    )
    with patch.object(core, "CFG", {}), \
         patch.object(
            core,
            "detect_resource_regions",
            return_value={"wood_stockpile": (0, 0, 10, 10)},
        ), \
         patch.object(
            core,
            "prepare_roi",
            return_value=(0, 0, 10, 10, frame, gray, 0, 0),
        ), \
         patch.object(
            core,
            "execute_ocr",
            return_value=("12", data, None, False),
        ):
        caplog.set_level(logging.INFO, logger="script.resources")
        core._read_resources(
            frame,
            ["wood_stockpile"],
            ["wood_stockpile"],
            cache_obj=cache_obj,
        )
    messages = [r.getMessage() for r in caplog.records]
    ocr_msgs = [m for m in messages if m.startswith("OCR wood_stockpile")]
    assert ocr_msgs, "OCR log message not found"
    msg = ocr_msgs[0]
    assert "digits=12" in msg
    assert "conf=[50.0]" in msg
    assert "low_conf=False" in msg
