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
    # Track resources flagged in the most recent read
    last_low_confidence: set = field(default_factory=set)
    last_no_digits: set = field(default_factory=set)


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


def reset(cache_obj: ResourceCache = RESOURCE_CACHE) -> None:
    """Clear cached resource detection state.

    Parameters
    ----------
    cache_obj:
        The :class:`ResourceCache` instance to reset. Defaults to the module's
        shared :data:`RESOURCE_CACHE`.
    """

    cache_obj.last_icon_bounds.clear()
    cache_obj.last_resource_values.clear()
    cache_obj.last_resource_ts.clear()
    cache_obj.resource_failure_counts.clear()
    cache_obj.last_debug_image_ts.clear()
    cache_obj.last_debug_failure_set.clear()
    cache_obj.last_debug_failure_ts = None
    cache_obj.last_low_confidence.clear()
    cache_obj.last_no_digits.clear()

    _LAST_READ_FROM_CACHE.clear()
    _NARROW_ROIS.clear()
    _NARROW_ROI_DEFICITS.clear()
    global _LAST_REGION_BOUNDS
    _LAST_REGION_BOUNDS = None
    _LAST_REGION_SPANS.clear()

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
    "reset",
]
