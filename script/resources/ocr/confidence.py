from __future__ import annotations

"""Confidence utilities for OCR results."""


def parse_confidences(data):
    """Return positive OCR confidence values or ``None`` when unavailable.

    Values that cannot be parsed as floats are ignored. Negative confidences
    are clamped to ``0`` and, along with explicit zeros, are treated as
    "unknown" and therefore omitted from the returned list. If no positive
    confidences remain after filtering, ``None`` is returned to signal that the
    OCR engine did not provide meaningful confidence information.
    """

    raw = data.get("conf")
    if not raw:
        return None

    confs = []
    for c in raw:
        try:
            val = float(c)
        except (ValueError, TypeError):
            continue
        if val <= 0:
            continue
        confs.append(val)
    return confs or None


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
