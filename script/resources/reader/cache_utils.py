from .. import cache

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

def get_narrow_roi_deficit(name: str) -> int | None:
    return _NARROW_ROI_DEFICITS.get(name)

def get_failure_count(cache_obj: ResourceCache, name: str) -> int:
    return cache_obj.resource_failure_counts.get(name, 0)

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
    "get_narrow_roi_deficit",
    "get_failure_count",
]
