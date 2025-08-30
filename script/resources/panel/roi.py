"""Resource ROI computations."""
import logging

from .. import RESOURCE_ICON_ORDER, cache
from . import _get_resource_panel_cfg

logger = logging.getLogger(__name__)


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
    min_requireds=None,
    detected=None,
):
    """Compute resource ROIs from detected icon bounds."""

    if min_requireds is None:
        min_requireds = [0] * len(RESOURCE_ICON_ORDER)
    if detected is None:
        detected = {}
    regions = {}
    spans = {}
    narrow = {}

    for idx, current in enumerate(RESOURCE_ICON_ORDER[:-1]):
        next_name = RESOURCE_ICON_ORDER[idx + 1]
        cur_bounds = detected.get(current)
        if cur_bounds is None:
            continue

        pad_l = pad_left[idx] if idx < len(pad_left) else pad_left[-1]
        pad_r = pad_right[idx] if idx < len(pad_right) else pad_right[-1]

        if current == "food_stockpile":
            pad_l = max(pad_l, 2)
            pad_r = max(pad_r, 2)

        cur_x, _cy, cur_w, _ch = cur_bounds
        cur_trim_val = icon_trims[idx] if idx < len(icon_trims) else icon_trims[-1]
        cur_trim = int(round(cur_trim_val * cur_w)) if 0 <= cur_trim_val <= 1 else int(
            round(cur_trim_val)
        )
        cur_right = panel_left + cur_x + cur_w - cur_trim

        next_bounds = detected.get(next_name)
        if next_bounds is not None:
            next_x, _ny, next_w, _nh = next_bounds
            next_trim_val = icon_trims[idx + 1] if idx + 1 < len(icon_trims) else icon_trims[-1]
            next_trim = (
                int(round(next_trim_val * next_w)) if 0 <= next_trim_val <= 1 else int(round(next_trim_val))
            )
            next_left = panel_left + next_x - next_trim
        else:
            next_left = panel_right

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
            max_w = min(max_w, 50)
        width = min(max_w, available_width)

        min_req = min_requireds[idx] if idx < len(min_requireds) else min_requireds[-1]
        if available_width >= min_req:
            width = max(width, min_req)

        min_w = min_widths[idx] if idx < len(min_widths) else min_widths[-1]
        if available_width < min_w:
            width = available_width
            narrow[current] = min_w - available_width
            logger.warning(
                "Narrow ROI for '%s': available=%d min=%d",
                current,
                available_width,
                min_w,
            )

        right = left + width
        spans[current] = (left, right)

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
        detected=detected,
    )

    cache._NARROW_ROIS = set(narrow.keys())
    cache._NARROW_ROI_DEFICITS = narrow.copy()

    for name in list(regions.keys()):
        l, t, w, hgt = regions[name]
        spans[name] = (l, l + w)

    if "idle_villager" in required_icons:
        idx_iv = RESOURCE_ICON_ORDER.index("idle_villager")
        left_iv = left + int(idx_iv * slice_w + pad_left_fallback[idx_iv])
        right_iv = left + int(width - pad_right_fallback[idx_iv])
        max_iv = (
            cfg.max_widths[idx_iv]
            if idx_iv < len(cfg.max_widths)
            else cfg.max_widths[-1]
        )
        width_iv = min(max_iv, right_iv - left_iv)
        regions["idle_villager"] = (left_iv, top, width_iv, height)
        spans["idle_villager"] = (left_iv, left_iv + width_iv)

    cache._LAST_REGION_SPANS = spans.copy()

    return regions
