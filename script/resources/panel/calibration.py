"""Resource panel calibration utilities."""
import logging

from .. import CFG, screen_utils, common, RESOURCE_ICON_ORDER, cache, cv2, np
from .roi import compute_resource_rois
from . import _get_resource_panel_cfg

logger = logging.getLogger(__name__)


def _auto_calibrate_from_icons(frame, cache_obj: cache.ResourceCache = cache.RESOURCE_CACHE):
    """Locate resource icons directly on the frame to derive ROI regions."""

    screen_utils._load_icon_templates()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cfg = _get_resource_panel_cfg()

    detected = {}
    for name in RESOURCE_ICON_ORDER:
        icon = screen_utils.ICON_TEMPLATES.get(name)
        if icon is None:
            continue
        best = (0, None, None)
        for scale in cfg.scales:
            icon_scaled = cv2.resize(icon, None, fx=scale, fy=scale)
            result = cv2.matchTemplate(gray, icon_scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best[0]:
                best = (max_val, max_loc, icon_scaled.shape[::-1])
        if best[0] >= cfg.match_threshold and best[1] is not None:
            bw, bh = best[2]
            detected[name] = (best[1][0], best[1][1], bw, bh)
            cache_obj.last_icon_bounds[name] = (best[1][0], best[1][1], bw, bh)
        elif name in cache_obj.last_icon_bounds:
            detected[name] = cache_obj.last_icon_bounds[name]

    if len(detected) < 2:
        return {}

    panel_left = min(x for x, _y, _w, _h in detected.values())
    panel_top = min(y for _x, y, _w, _h in detected.values())
    panel_right = max(x + w for x, _y, w, _h in detected.values())
    panel_bottom = max(y + h for _x, y, _w, h in detected.values())
    panel_height = panel_bottom - panel_top

    detected_rel = {
        name: (x - panel_left, y - panel_top, w, h)
        for name, (x, y, w, h) in detected.items()
    }

    top = panel_top + int(cfg.top_pct * panel_height)
    height = int(cfg.height_pct * panel_height)

    regions, spans, narrow = compute_resource_rois(
        panel_left,
        panel_right,
        top,
        height,
        cfg.pad_left,
        cfg.pad_right,
        cfg.icon_trims,
        cfg.max_widths,
        cfg.min_widths,
        cfg.min_pop_width,
        cfg.min_requireds,
        detected_rel,
    )

    cache._NARROW_ROIS = set(narrow.keys())
    cache._NARROW_ROI_DEFICITS = narrow.copy()
    cache._LAST_REGION_SPANS = spans.copy()

    if "idle_villager" in detected_rel:
        xi, yi, wi, hi = detected_rel["idle_villager"]
        span = spans.get("idle_villager")
        if span:
            left, right = span
        else:
            left = panel_left + xi + wi
            right = left
        pop_span = spans.get("population_limit")
        if pop_span and pop_span[0] > left and right > pop_span[0]:
            right = pop_span[0]
        if right > panel_right:
            right = panel_right
        width = max(0, right - left)
        regions["idle_villager"] = (left, panel_top + yi, width, hi)

    if cache._LAST_REGION_BOUNDS != regions:
        cache._LAST_REGION_BOUNDS = regions.copy()
        cache_obj.last_resource_values.clear()
        cache_obj.last_resource_ts.clear()

    return regions


def _apply_custom_rois(frame, regions, names=None, include_idle=True):
    """Apply ROI overrides defined in the configuration."""

    from .. import input_utils

    if include_idle and "idle_villager" not in regions:
        idle_cfg = CFG.get("idle_villager_roi")
        if idle_cfg:
            W, H = input_utils._screen_size()
            left = int(idle_cfg.get("left_pct", 0) * W)
            top = int(idle_cfg.get("top_pct", 0) * H)
            width = int(idle_cfg.get("width_pct", 0) * W)
            height = int(idle_cfg.get("height_pct", 0) * H)
            regions["idle_villager"] = (
                left,
                top,
                max(40, width),
                max(20, height),
            )
            logger.debug(
                "Custom ROI applied for idle_villager: %s", regions["idle_villager"]
            )

    if names is None:
        names = [
            "wood_stockpile",
            "food_stockpile",
            "gold_stockpile",
            "stone_stockpile",
            "population_limit",
        ]

    W, H = input_utils._screen_size()
    for name in names:
        cfg = CFG.get(f"{name}_roi")
        if not cfg:
            continue
        left = int(cfg.get("left_pct", 0) * W)
        top = int(cfg.get("top_pct", 0) * H)
        width = int(cfg.get("width_pct", 0) * W)
        height = int(cfg.get("height_pct", 0) * H)
        regions[name] = (left, top, width, height)
        logger.debug("Custom ROI applied for %s: %s", name, regions[name])

    return regions


def _recalibrate_low_variance(frame, regions, required_icons, cache_obj):
    """Recalibrate ROIs that exhibit low variance."""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = CFG.get("roi_variance_threshold", 5.0)
    low_variance = []
    for name in required_icons:
        roi = regions.get(name)
        if not roi:
            continue
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            continue
        sub = gray[y : y + h, x : x + w]
        if sub.size == 0 or float(np.var(sub)) < thresh:
            low_variance.append(name)

    if low_variance:
        calibrated = _auto_calibrate_from_icons(frame, cache_obj)
        if calibrated:
            if CFG.get("disable_roi_overrides_on_calibration"):
                regions = calibrated
            else:
                for name in low_variance:
                    if name in calibrated:
                        cx, cy, cw, ch = calibrated[name]
                        sub2 = gray[cy : cy + ch, cx : cx + cw]
                        if sub2.size and float(np.var(sub2)) >= thresh:
                            regions[name] = calibrated[name]

    return regions


def _remove_overlaps(regions, required_icons):
    """Trim overlapping ROIs to ensure distinct regions."""

    ordered = [
        name for name in RESOURCE_ICON_ORDER if name in regions and name in required_icons
    ]
    for prev, curr in zip(ordered, ordered[1:]):
        l1, t1, w1, h1 = regions[prev]
        l2, t2, w2, h2 = regions[curr]
        overlap = (l1 + w1) - l2
        if overlap > 0:
            new_w1 = l2 - l1
            if new_w1 <= 0:
                logger.warning(
                    "ROI '%s' removed due to complete overlap with '%s'", prev, curr
                )
                del regions[prev]
            else:
                regions[prev] = (l1, t1, new_w1, h1)
                logger.debug(
                    "ROI for '%s' reduced by %dpx to avoid overlap with '%s'",
                    prev,
                    overlap,
                    curr,
                )

    return regions
