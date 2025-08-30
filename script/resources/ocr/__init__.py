"""OCR helpers for reading HUD resources."""

from .preprocess import preprocess_roi
from . import masks
from .confidence import parse_confidences, _sanitize_digits
from .executor import (
    execute_ocr,
    handle_ocr_failure,
    _read_population_from_roi,
    read_population_from_roi,
    _extract_population,
)

__all__ = [
    "preprocess_roi",
    "parse_confidences",
    "_sanitize_digits",
    "execute_ocr",
    "handle_ocr_failure",
    "_read_population_from_roi",
    "read_population_from_roi",
    "_extract_population",
]
