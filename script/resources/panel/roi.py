"""Resource ROI computations."""
import logging

from .. import CFG, RESOURCE_ICON_ORDER, cache
from . import _get_resource_panel_cfg

logger = logging.getLogger(__name__)

_THREE_DIGIT_SPAN = 30


def compute_resource_rois(
    panel_left,
    panel_right,
    top,
    height,
    pad_left,
    pad_right,
    icon_trims,
    max_widths,
    min_widths,
    min_pop_width,
    idle_extra_width,
    min_requireds=None,
    detected=None,
):
    """Compute resource ROIs from detected icon bounds."""
    if not max_widths:
        raise ValueError("max_widths must contain at least one element")
    if not min_widths:
        raise ValueError("min_widths must contain at least one element")

    if min_requireds is None:
        min_requireds = [0] * len(RESOURCE_ICON_ORDER)
    if detected is None:
        detected = {}
    regions = {}
    spans = {}
    narrow = {}

    for idx, current in enumerate(RESOURCE_ICON_ORDER):
        next_name = RESOURCE_ICON_ORDER[idx + 1] if idx + 1 < len(RESOURCE_ICON_ORDER) else None
        cur_bounds = detected.get(current)
        if cur_bounds is None:
            continue
        cur_x, _cy, cur_w, _ch = cur_bounds

        if current == "idle_villager":
            inner_trim = CFG.get("idle_icon_inner_trim")
            if inner_trim is None:
                inner_trim = int(round(CFG.get("idle_icon_inner_pct", 0.0) * cur_w))
            else:
                inner_trim = int(inner_trim)
            max_trim = cur_w // 4
            inner_trim = max(0, min(inner_trim, max_trim))

            left = panel_left + cur_x + inner_trim
            right = panel_left + cur_x + cur_w - inner_trim
            width = max(0, right - left)
            spans[current] = (left, right)

            if CFG.get("ocr_debug"):
                logger.info("Span for '%s': (%d, %d)", current, left, right)

            regions[current] = (left, top, width, height)
            logger.debug(
                "ROI for '%s': available=(%d,%d) width=%d",
                current,
                left,
                right,
                width,
            )
            continue

        cur_right = panel_left + cur_x + cur_w

        next_bounds = detected.get(next_name) if next_name else None
        if next_bounds is not None:
            next_x, _ny, next_w, _nh = next_bounds
            next_left = panel_left + next_x
        else:
            next_left = panel_right
            next_w = 0

        idle_padding = (
            CFG.get("population_idle_padding", 6)
            if current == "population_limit" and next_name == "idle_villager"
            else 0
        )

        min_req = min_requireds[idx] if idx < len(min_requireds) else min_requireds[-1]
        min_w = min_widths[idx] if idx < len(min_widths) else min_widths[-1]
        min_span = max(_THREE_DIGIT_SPAN, min_req, min_w)

        pad_l = pad_left[idx] if idx < len(pad_left) else pad_left[-1]
        pad_r = pad_right[idx] if idx < len(pad_right) else pad_right[-1]

        available_left = cur_right + pad_l
        available_right = next_left - pad_r
        available_width = available_right - available_left

        left = max(panel_left, available_left)
        right_limit = min(panel_right, available_right - idle_padding)

        if right_limit <= left:
            logger.warning(
                "Skipping ROI for icon '%s' due to non-positive span (left=%d, right=%d)",
                current,
                left,
                right_limit,
            )
            continue

        if available_width < min_span:
            narrow[current] = min_span - available_width
            logger.warning(
                "Narrow ROI for '%s': available=%d min=%d",
                current,
                available_width,
                min_span,
            )

        roi_available = right_limit - left

        max_w = max_widths[idx] if idx < len(max_widths) else max_widths[-1]
        if current == "food_stockpile":
            max_w = min(max_w, CFG.get("food_stockpile_max_width", max_w))
        if current == "population_limit" and next_bounds is not None:
            max_w = roi_available
        width = min(max_w, roi_available)

        if roi_available >= min_req:
            width = max(width, min_req)

        if roi_available >= min_w:
            width = max(width, min_w)

        if current == "population_limit":
            width = max(width, min_pop_width)
            extra = CFG.get("pop_roi_extra_width", 0)
            right = min(panel_right, right_limit, left + width + extra)
            width = right - left
        else:
            right = left + width

        spans[current] = (left, right)

        if CFG.get("ocr_debug"):
            logger.info("Span for '%s': (%d, %d)", current, left, right)

        regions[current] = (left, top, width, height)
        logger.debug(
            "ROI for '%s': available=(%d,%d) width=%d",
            current,
            left,
            right,
            width,
        )
    return regions, spans, narrow


def _fallback_rois_from_slice(
    left,
    width,
    top,
    height,
    icon_trims,
    right_trim,
    required_icons,
):
    """Construct resource ROIs from generic slice bounds."""

    slice_w = width / len(RESOURCE_ICON_ORDER)
    detected = {
        name: (int(idx * slice_w), 0, 0, 0)
        for idx, name in enumerate(RESOURCE_ICON_ORDER)
    }

    pad_left_fallback = [0] * len(RESOURCE_ICON_ORDER)
    pad_right_fallback = [int(round(right_trim * slice_w))] * len(RESOURCE_ICON_ORDER)
    icon_trims_zero = [0] * len(RESOURCE_ICON_ORDER)

    cfg = _get_resource_panel_cfg()
    max_widths = cfg.max_widths
    min_widths = cfg.min_widths
    regions, spans, _narrow = compute_resource_rois(
        left,
        left + width,
        top,
        height,
        pad_left_fallback,
        pad_right_fallback,
        icon_trims_zero,
        max_widths,
        min_widths,
        cfg.min_pop_width,
        cfg.idle_roi_extra_width,
        detected=detected,
    )

    cache._NARROW_ROIS.clear()
    cache._NARROW_ROI_DEFICITS.clear()

    for name in list(regions.keys()):
        l, t, w, hgt = regions[name]
        spans[name] = (l, l + w)

    cache._LAST_REGION_SPANS = spans.copy()

    return regions
