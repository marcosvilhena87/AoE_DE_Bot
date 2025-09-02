from __future__ import annotations

"""Confidence utilities for OCR results."""


def parse_confidences(data):
    """Return OCR confidences aligned with the ``text`` output.

    ``pytesseract.image_to_data`` returns separate ``text`` and ``conf`` lists
    that are expected to have matching lengths.  Earlier implementations of
    this helper filtered out non-positive or unparsable confidence values which
    could lead to misalignment between the two lists.  The executor would then
    zip the shortened confidence list with the full ``text`` list, causing
    confidences to be associated with the wrong text entries.

    This function now preserves list alignment by returning a list of the same
    length as ``data['text']`` (when present).  Invalid confidence values are
    converted to ``0`` and non-positive values are also normalised to ``0`` so
    callers can easily filter them out while maintaining positional alignment.
    If the ``conf`` key is missing or empty, ``None`` is returned to signal
    that no confidence information is available.
    """

    raw = data.get("conf")
    if not raw:
        return None

    texts = data.get("text")
    confs = []
    for c in raw:
        try:
            val = float(c)
        except (ValueError, TypeError):
            val = 0.0
        if val <= 0:
            val = 0.0
        confs.append(val)

    if texts is not None:
        text_len = len(texts)
        if len(confs) < text_len:
            confs.extend([0.0] * (text_len - len(confs)))
        elif len(confs) > text_len:
            confs = confs[:text_len]

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
