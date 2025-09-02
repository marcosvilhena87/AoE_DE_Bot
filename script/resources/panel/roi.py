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
    if not pad_left:
        raise ValueError("pad_left must contain at least one element")
    if not pad_right:
        raise ValueError("pad_right must contain at least one element")
    if not icon_trims:
        raise ValueError("icon_trims must contain at least one element")
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

        pad_l = pad_left[idx] if idx < len(pad_left) else pad_left[-1]
        pad_r = pad_right[idx] if idx < len(pad_right) else pad_right[-1]

        cur_x, _cy, cur_w, _ch = cur_bounds
        pad_extra = 0
        if current in ("wood_stockpile", "food_stockpile"):
            pad_extra = max(1, int(round(cur_w * 0.25)))
        if current in ("food_stockpile", "idle_villager"):
            pad_l = max(pad_l, 3)
            pad_r = max(pad_r, 3)

        cur_trim_val = icon_trims[idx] if idx < len(icon_trims) else icon_trims[-1]
        cur_trim = int(round(cur_trim_val * cur_w)) if 0 <= cur_trim_val <= 1 else int(
            round(cur_trim_val)
        )
        cur_right = panel_left + cur_x + cur_w - cur_trim

        next_bounds = detected.get(next_name) if next_name else None
        if next_bounds is not None:
            next_x, _ny, next_w, _nh = next_bounds
            next_trim_val = icon_trims[idx + 1] if idx + 1 < len(icon_trims) else icon_trims[-1]
            next_trim = (
                int(round(next_trim_val * next_w)) if 0 <= next_trim_val <= 1 else int(round(next_trim_val))
            )
            next_left = panel_left + next_x - next_trim
        else:
            next_left = panel_right

        min_req = min_requireds[idx] if idx < len(min_requireds) else min_requireds[-1]
        min_w = min_widths[idx] if idx < len(min_widths) else min_widths[-1]
        min_span = max(_THREE_DIGIT_SPAN, min_req, min_w)
        gap = next_left - cur_right
        if gap - pad_l - pad_r < min_span:
            deficit = min_span - (gap - pad_l - pad_r)
            reduce_l = (
                min(pad_l - pad_extra, deficit // 2) if pad_l > pad_extra else 0
            )
            reduce_r = min(pad_r, deficit - reduce_l)
            pad_l -= reduce_l
            pad_r -= reduce_r

        left = cur_right + pad_l
        right = next_left - pad_r

        # Clamp ROI boundaries to the panel limits after applying padding
        left = max(panel_left, left)
        right = min(panel_right, right)

        if right <= left:
            logger.warning(
                "Skipping ROI for icon '%s' due to non-positive span (left=%d, right=%d)",
                current,
                left,
                right,
            )
            continue

        available_width = right - left
        max_w = max_widths[idx] if idx < len(max_widths) else max_widths[-1]
        if current == "food_stockpile":
            max_w = min(max_w, CFG.get("food_stockpile_max_width", max_w))
        width = min(max_w, available_width)

        min_req = min_requireds[idx] if idx < len(min_requireds) else min_requireds[-1]
        if available_width >= min_req:
            width = max(width, min_req)

        min_w = min_widths[idx] if idx < len(min_widths) else min_widths[-1]
        if available_width >= min_w:
            width = max(width, min_w)
        else:
            width = available_width
            narrow[current] = min_w - available_width
            logger.warning(
                "Narrow ROI for '%s': available=%d min=%d",
                current,
                available_width,
                min_w,
            )

        if current == "population_limit" and width < min_pop_width:
            width = min_pop_width
            right = min(panel_right, left + width)
            left = max(panel_left, right - width)
            width = right - left
        else:
            right = left + width
        if current in ("wood_stockpile", "food_stockpile"):
            pad = max(1, int(round(cur_w * 0.25)))
            left = max(panel_left, left - pad)
            right = min(panel_right, right + pad)
            width = right - left

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
    if (
        "idle_villager" in detected
        and "idle_villager" not in regions
        and idle_extra_width > 0
    ):
        xi, _yi, wi, _hi = detected["idle_villager"]
        left = panel_left + xi + wi
        right = left + idle_extra_width
        pop_span = spans.get("population_limit")
        if pop_span and pop_span[0] > left and right > pop_span[0]:
            right = pop_span[0]
        if right > panel_right:
            right = panel_right
        width = max(0, right - left)
        regions["idle_villager"] = (left, top, width, height)
        spans["idle_villager"] = (left, right)

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

    pad_left_fallback = [
        int(
            round(
                (icon_trims[idx] if idx < len(icon_trims) else icon_trims[-1])
                * slice_w
            )
        )
        for idx in range(len(RESOURCE_ICON_ORDER))
    ]
    pad_right_fallback = [
        int(round(right_trim * slice_w))
    ] * len(RESOURCE_ICON_ORDER)
    icon_trims_zero = [0] * len(RESOURCE_ICON_ORDER)

    cfg = _get_resource_panel_cfg()
    max_widths = cfg.max_widths
    min_widths = cfg.min_widths
    regions, spans, narrow = compute_resource_rois(
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
    cache._NARROW_ROIS.update(narrow.keys())
    cache._NARROW_ROI_DEFICITS.clear()
    cache._NARROW_ROI_DEFICITS.update(narrow)

    for name in list(regions.keys()):
        l, t, w, hgt = regions[name]
        spans[name] = (l, l + w)

    cache._LAST_REGION_SPANS = spans.copy()

    return regions
