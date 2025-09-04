"""Resource handling package for HUD detection and OCR.

Supported API:
    ResourceCache, RESOURCE_CACHE
    detect_hud, compute_resource_rois, locate_resource_panel, detect_resource_regions
    preprocess_roi, execute_ocr, handle_ocr_failure, read_population_from_roi
    read_resources_from_hud, gather_hud_stats, validate_starting_resources,
    ResourceValidationError
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from .. import common, screen_utils
from ..template_utils import find_template

ROOT = Path(__file__).resolve().parent.parent
CFG = common.STATE.config
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

# Re-export cache class and shared instance
ResourceCache = cache.ResourceCache
RESOURCE_CACHE = cache.RESOURCE_CACHE

# Panel helpers
from .panel import (
    detect_hud,
    compute_resource_rois,
    locate_resource_panel,
    detect_resource_regions,
)

# OCR helpers
from .ocr.preprocess import preprocess_roi
from .ocr.executor import (
    execute_ocr,
    handle_ocr_failure,
    read_population_from_roi,
)

# Reader functions
from .reader import (
    read_resources_from_hud,
    gather_hud_stats,
    validate_starting_resources,
    ResourceValidationError,
)

__all__ = [
    "ResourceCache",
    "RESOURCE_CACHE",
    "detect_hud",
    "compute_resource_rois",
    "locate_resource_panel",
    "detect_resource_regions",
    "preprocess_roi",
    "execute_ocr",
    "handle_ocr_failure",
    "read_population_from_roi",
    "read_resources_from_hud",
    "gather_hud_stats",
    "validate_starting_resources",
    "ResourceValidationError",
]
