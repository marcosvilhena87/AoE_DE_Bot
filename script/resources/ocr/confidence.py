from __future__ import annotations

"""Confidence utilities for OCR results."""


def parse_confidences(data):
    """Convert OCR confidence values to floats, ignoring non-positive entries."""

    confs = []
    for c in data.get("conf", []):
        try:
            val = float(c)
        except (ValueError, TypeError):
            continue
        if val > 0:
            confs.append(val)
    return confs


def _sanitize_digits(digits: str) -> str:
    """Trim OCR output to at most three significant digits.

    Trailing zeros are stripped *after* limiting to three digits to ensure
    values like ``1400`` are sanitised to ``140`` instead of ``14``.
    """

    if len(digits) <= 3:
        return digits

    first_three = digits[:3]
    trimmed = digits.rstrip("0")
    if len(trimmed) <= 3:
        return first_three
    return trimmed[:3]


__all__ = ["parse_confidences", "_sanitize_digits"]
