from __future__ import annotations

"""Confidence utilities for OCR results."""


def parse_confidences(data):
    """Return OCR confidences for non-empty OCR ``text`` entries.

    ``pytesseract.image_to_data`` returns separate ``text`` and ``conf`` lists
    that ideally have matching lengths.  Some Tesseract versions, however,
    omit confidence values for boxes where the recognised text is empty or
    whitespace.  When this happens the lists become misaligned, which can in
    turn cause confidences to be associated with the wrong text entries.

    This helper resolves the issue by first filtering out any boxes whose
    corresponding ``text`` value is empty or only whitespace.  The remaining
    ``text`` entries are written back to ``data['text']`` so callers operating
    on ``data`` remain in sync with the returned confidence list.  Invalid or
    non-positive confidence values are normalised to ``0`` so callers can
    easily ignore them.  If the ``conf`` key is missing or empty, ``None`` is
    returned to signal that no confidence information is available.
    """

    raw_conf = data.get("conf")
    if not raw_conf:
        return None

    filtered_texts: list[str] = []
    filtered_conf: list[float] = []
    for t, c in zip(data.get("text") or [], raw_conf):
        if not str(t).strip():
            continue
        try:
            val = float(c)
        except (ValueError, TypeError):
            val = 0.0
        if val <= 0:
            val = 0.0
        filtered_texts.append(t)
        filtered_conf.append(val)

    data["text"] = filtered_texts
    data["conf"] = filtered_conf
    return filtered_conf


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
