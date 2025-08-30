"""Resource handling package for HUD detection and OCR."""

import logging
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from ..config_utils import load_config
from .. import screen_utils, common
from ..template_utils import find_template

ROOT = Path(__file__).resolve().parent.parent
CFG = load_config()
logger = logging.getLogger(__name__)

RESOURCE_ICON_ORDER = [
    "wood_stockpile",
    "food_stockpile",
    "gold_stockpile",
    "stone_stockpile",
    "population_limit",
    "idle_villager",
]

# Submodules
from . import cache, panel, ocr, reader

# Re-export cache and helper functions
ResourceCache = cache.ResourceCache
RESOURCE_CACHE = cache.RESOURCE_CACHE
_LAST_READ_FROM_CACHE = cache._LAST_READ_FROM_CACHE
_NARROW_ROIS = cache._NARROW_ROIS
_NARROW_ROI_DEFICITS = cache._NARROW_ROI_DEFICITS
_RESOURCE_CACHE_TTL = cache._RESOURCE_CACHE_TTL
_RESOURCE_CACHE_MAX_AGE = cache._RESOURCE_CACHE_MAX_AGE
_RESOURCE_DEBUG_COOLDOWN = cache._RESOURCE_DEBUG_COOLDOWN
_LAST_REGION_BOUNDS = cache._LAST_REGION_BOUNDS
_LAST_REGION_SPANS = cache._LAST_REGION_SPANS


def _screen_size():
    monitor = screen_utils.get_monitor()
    return monitor["width"], monitor["height"]


input_utils = SimpleNamespace(_screen_size=_screen_size)

# Panel helpers
from .panel import (
    detect_hud,
    compute_resource_rois,
    _get_resource_panel_cfg,
    locate_resource_panel,
    detect_resource_regions,
    _auto_calibrate_from_icons,
    _fallback_rois_from_slice,
    _apply_custom_rois,
    _recalibrate_low_variance,
    _remove_overlaps,
)

# OCR helpers
from .ocr import (
    preprocess_roi,
    execute_ocr,
    handle_ocr_failure,
    _read_population_from_roi,
    read_population_from_roi,
    _extract_population,
)

# Reader functions
from .reader import (
    _read_resources,
    read_resources_from_hud,
    gather_hud_stats,
    validate_starting_resources,
)

__all__ = [
    "ResourceCache",
    "RESOURCE_CACHE",
    "detect_hud",
    "compute_resource_rois",
    "_get_resource_panel_cfg",
    "locate_resource_panel",
    "detect_resource_regions",
    "_auto_calibrate_from_icons",
    "_fallback_rois_from_slice",
    "_apply_custom_rois",
    "_recalibrate_low_variance",
    "_remove_overlaps",
    "preprocess_roi",
    "execute_ocr",
    "handle_ocr_failure",
    "read_population_from_roi",
    "_read_resources",
    "read_resources_from_hud",
    "gather_hud_stats",
    "validate_starting_resources",
]
