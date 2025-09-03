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
    # ``pad_left``, ``pad_right`` and ``max_widths`` previously affected the
    # computed span.  Width now always spans the full gap between consecutive
    # icons so these parameters are effectively ignored but kept for backwards
    # compatibility with callers.
    if not min_widths:
        min_widths = [0] * len(RESOURCE_ICON_ORDER)
    if min_requireds is None:
        min_requireds = [0] * len(RESOURCE_ICON_ORDER)
    if detected is None:
        detected = {}

    if not detected:
        return {}, {}, {}

    # Offset from icon-relative coordinates to absolute screen coordinates
    min_y = min(v[1] for v in detected.values())
    panel_top = top - min_y
    panel_bottom = top + height

    regions = {}
    spans = {}
    narrow = {}

    for idx, current in enumerate(RESOURCE_ICON_ORDER):
        next_name = (
            RESOURCE_ICON_ORDER[idx + 1] if idx + 1 < len(RESOURCE_ICON_ORDER) else None
        )
        cur_bounds = detected.get(current)
        if cur_bounds is None:
            continue
        cur_x, cur_y, cur_w, cur_h = cur_bounds

        if current == "idle_villager":
            inner_trim = CFG.get("idle_icon_inner_trim")
            if inner_trim is None:
                inner_trim = int(round(CFG.get("idle_icon_inner_pct", 0.0) * cur_w))
            else:
                inner_trim = int(inner_trim)
            max_trim = cur_w // 4
            inner_trim = max(0, min(inner_trim, max_trim))
            left = panel_left + cur_x + inner_trim
            right = panel_left + cur_x + cur_w - inner_trim + idle_extra_width

            # Keep span within panel bounds and prevent overlap with previous resource
            prev_name = RESOURCE_ICON_ORDER[idx - 1] if idx > 0 else None
            prev_span = spans.get(prev_name)
            if prev_span:
                left = max(left, prev_span[1])
            right = min(right, panel_right)

            if right <= left:
                logger.warning(
                    "Skipping ROI for icon '%s' due to non-positive span (left=%d, right=%d)",
                    current,
                    left,
                    right,
                )
                continue

            width = right - left
            spans[current] = (left, right)

            if CFG.get("ocr_debug"):
                logger.info("Span for '%s': (%d, %d)", current, left, right)

            regions[current] = (left, panel_top + cur_y, width, cur_h)
            logger.debug(
                "ROI for '%s': available=(%d,%d) width=%d",
                current,
                left,
                right,
                width,
            )
            continue

        # Determine the icons that bound this span
        cur_right = panel_left + cur_x + cur_w

        next_bounds = detected.get(next_name) if next_name else None
        if next_bounds is not None:
            next_x, next_y, next_w, next_h = next_bounds
            next_left = panel_left + next_x
        else:
            next_left = panel_right
            next_y = cur_y
            next_h = cur_h

        left = cur_right
        right = next_left

        if right <= left:
            logger.warning(
                "Skipping ROI for icon '%s' due to non-positive span (left=%d, right=%d)",
                current,
                left,
                right,
            )
            continue

        width = right - left

        min_req = min_requireds[idx] if idx < len(min_requireds) else min_requireds[-1]
        min_w = min_widths[idx] if idx < len(min_widths) else min_widths[-1]
        min_span = max(_THREE_DIGIT_SPAN, min_req, min_w)
        if current == "population_limit":
            min_span = max(min_span, min_pop_width)

        if width < min_span:
            narrow[current] = min_span - width
            logger.warning(
                "Narrow ROI for '%s': available=%d min=%d",
                current,
                width,
                min_span,
            )

        spans[current] = (left, right)

        if CFG.get("ocr_debug"):
            logger.info("Span for '%s': (%d, %d)", current, left, right)

        roi_y = min(cur_y, next_y)
        roi_bottom = max(cur_y + cur_h, next_y + next_h)
        roi_top = panel_top + roi_y
        abs_bottom = panel_top + roi_bottom
        if current == "food_stockpile":
            roi_top = max(top, roi_top - 1)
            abs_bottom = min(panel_bottom, abs_bottom + 1)
        roi_height = abs_bottom - roi_top

        regions[current] = (left, roi_top, width, roi_height)
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
        name: (int(idx * slice_w), 0, 0, height)
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
