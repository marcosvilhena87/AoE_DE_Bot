"""Caching utilities for resource detection and OCR."""

from dataclasses import dataclass, field

from ..config_utils import load_config

# Load configuration at module import
CFG = load_config()


@dataclass
class ResourceCache:
    """In-memory cache for resource detection state."""

    last_icon_bounds: dict = field(default_factory=dict)
    last_resource_values: dict = field(default_factory=dict)
    last_resource_ts: dict = field(default_factory=dict)
    resource_failure_counts: dict = field(default_factory=dict)
    # Track when debug images were last written for each resource
    last_debug_image_ts: dict = field(default_factory=dict)
    # Track the last failure set and timestamp for throttling debug output
    last_debug_failure_set: set = field(default_factory=set)
    last_debug_failure_ts: float | None = None


# Shared cache instance used by default
RESOURCE_CACHE = ResourceCache()

# Icons fulfilled from cache on the most recent read
_LAST_READ_FROM_CACHE = set()

# Resource names whose available span fell below the configured minimum width
_NARROW_ROIS: set[str] = set()
# Track width deficit for resources with narrow spans
_NARROW_ROI_DEFICITS: dict[str, int] = {}

# Maximum age (in seconds) for cached resource values
_RESOURCE_CACHE_TTL = CFG.get("resource_cache_ttl", 1.5)
# Optional hard limit on cache age before rejection
_RESOURCE_CACHE_MAX_AGE = CFG.get("resource_cache_max_age")

# Minimum seconds between debug image dumps for identical failures
_RESOURCE_DEBUG_COOLDOWN = CFG.get("ocr_debug_cooldown", 2.0)

# Track last set of regions returned to invalidate cached values
_LAST_REGION_BOUNDS = None
# Track last available spans for each region
_LAST_REGION_SPANS: dict[str, tuple[int, int]] = {}

__all__ = [
    "ResourceCache",
    "RESOURCE_CACHE",
    "_LAST_READ_FROM_CACHE",
    "_NARROW_ROIS",
    "_NARROW_ROI_DEFICITS",
    "_RESOURCE_CACHE_TTL",
    "_RESOURCE_CACHE_MAX_AGE",
    "_RESOURCE_DEBUG_COOLDOWN",
    "_LAST_REGION_BOUNDS",
    "_LAST_REGION_SPANS",
]
