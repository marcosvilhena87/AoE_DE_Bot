"""Functions for reading resource values from the HUD."""

import time
from typing import Iterable

import cv2
import numpy as np
import pytesseract

from .. import CFG, ROOT, cache, common, logger, screen_utils, RESOURCE_ICON_ORDER
from ..panel import detect_resource_regions, locate_resource_panel
from .. import ocr
from ..ocr import masks

# Re-export OCR helpers
preprocess_roi = ocr.preprocess_roi
_ocr_digits_better = masks._ocr_digits_better
execute_ocr = ocr.execute_ocr
handle_ocr_failure = ocr.handle_ocr_failure
_read_population_from_roi = ocr._read_population_from_roi
read_population_from_roi = ocr.read_population_from_roi
_extract_population = ocr._extract_population

from .cache_utils import (
    ResourceCache,
    RESOURCE_CACHE,
    _LAST_READ_FROM_CACHE,
    _NARROW_ROIS,
    _NARROW_ROI_DEFICITS,
    _RESOURCE_CACHE_TTL,
    _RESOURCE_CACHE_MAX_AGE,
    _RESOURCE_DEBUG_COOLDOWN,
    _LAST_REGION_BOUNDS,
    _LAST_REGION_SPANS,
)
from .roi import prepare_roi, expand_roi_after_failure
from .core import (
    _read_resources,
    read_resources_from_hud,
    gather_hud_stats,
    validate_starting_resources,
)

__all__ = [
    "ResourceCache",
    "RESOURCE_CACHE",
    "_LAST_READ_FROM_CACHE",
    "_NARROW_ROIS",
    "_NARROW_ROI_DEFICITS",
    "_RESOURCE_CACHE_TTL",
    "_RESOURCE_DEBUG_COOLDOWN",
    "_LAST_REGION_BOUNDS",
    "_LAST_REGION_SPANS",
    "preprocess_roi",
    "_ocr_digits_better",
    "execute_ocr",
    "handle_ocr_failure",
    "_read_population_from_roi",
    "read_population_from_roi",
    "_read_resources",
    "read_resources_from_hud",
    "gather_hud_stats",
    "validate_starting_resources",
]
