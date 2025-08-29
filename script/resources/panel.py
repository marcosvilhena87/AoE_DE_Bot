"""HUD panel and ROI detection utilities."""
import logging
import time
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from . import CFG, ROOT, screen_utils, common, RESOURCE_ICON_ORDER
from . import cache

logger = logging.getLogger(__name__)


def detect_hud(frame):
    """Locate the resource panel and return its bounding box and score."""

    from . import find_template

    tmpl = screen_utils.HUD_TEMPLATE
    if tmpl is None:
        return None, 0.0

    def _save_debug(img, heatmap):
        debug_dir = ROOT / "debug"
        debug_dir.mkdir(exist_ok=True)
        ts = int(time.time() * 1000)
        cv2.imwrite(str(debug_dir / f"resource_panel_fail_{ts}.png"), img)
        if heatmap is not None:
            hm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(
                str(debug_dir / f"resource_panel_heat_{ts}.png"), hm.astype("uint8")
            )

    box, score, heat = find_template(
        frame, tmpl, threshold=CFG["threshold"], scales=CFG["scales"]
    )
    if not box:
        logger.warning(
            "Resource panel template not matched; score=%.3f", score
        )
        _save_debug(frame, heat)
        fallback = CFG.get("threshold_fallback")
        if fallback is not None:
            box, score, heat = find_template(
                frame, tmpl, threshold=fallback, scales=CFG["scales"]
            )
            if not box:
                logger.warning(
                    "Resource panel template not matched with fallback; score=%.3f",
                    score,
                )
                _save_debug(frame, heat)
                return None, score
        else:
            return None, score

    return box, score


def compute_resource_rois(
    panel_left,
    panel_right,
    top,
    height,
    pad_left,
    pad_right,
    icon_trims,
    max_width,
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

        spans[current] = (left, right)

        available_width = right - left
        width = min(max_width, available_width)

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

        regions[current] = (left, top, width, height)
        logger.debug(
            "ROI for '%s': available=(%d,%d) width=%d",
            current,
            left,
            right,
            width,
        )

    return regions, spans, narrow


@dataclass
class ResourcePanelCfg:
    match_threshold: float
    scales: Iterable[float]
    pad_left: list
    pad_right: list
    icon_trims: list
    max_width: int
    min_widths: list
    min_requireds: list
    top_pct: float
    height_pct: float
    idle_roi_extra_width: int


def _get_resource_panel_cfg():
    """Return processed configuration values for the resource panel."""

    res_cfg = CFG.get("resource_panel", {})
    profile = CFG.get("profile")
    profile_cfg = CFG.get("profiles", {}).get(profile, {})
    profile_res = profile_cfg.get("resource_panel", {})

    match_threshold = profile_res.get(
        "match_threshold", res_cfg.get("match_threshold", 0.8)
    )
    scales = res_cfg.get("scales", CFG.get("scales", [1.0]))

    pad_left = res_cfg.get("roi_padding_left", 2)
    pad_right = res_cfg.get("roi_padding_right", 2)
    num_icons = len(RESOURCE_ICON_ORDER)
    pad_left = pad_left if isinstance(pad_left, (list, tuple)) else [pad_left] * num_icons
    pad_right = pad_right if isinstance(pad_right, (list, tuple)) else [pad_right] * num_icons

    icon_trims = profile_res.get(
        "icon_trim_pct", res_cfg.get("icon_trim_pct", [0] * num_icons)
    )
    icon_trims = (
        icon_trims if isinstance(icon_trims, (list, tuple)) else [icon_trims] * num_icons
    )

    max_width = res_cfg.get("max_width", 160)

    min_width_cfg = res_cfg.get("min_width", 90)
    min_widths = (
        min_width_cfg
        if isinstance(min_width_cfg, (list, tuple))
        else [min_width_cfg] * num_icons
    )

    min_req_cfg = res_cfg.get("min_required_width", 0)
    min_requireds = (
        min_req_cfg
        if isinstance(min_req_cfg, (list, tuple))
        else [min_req_cfg] * num_icons
    )

    top_pct = profile_res.get("top_pct", res_cfg.get("top_pct", 0.08))
    height_pct = profile_res.get("height_pct", res_cfg.get("height_pct", 0.84))

    idle_extra = res_cfg.get("idle_roi_extra_width", 0)

    return ResourcePanelCfg(
        match_threshold,
        scales,
        pad_left,
        pad_right,
        icon_trims,
        max_width,
        min_widths,
        min_requireds,
        top_pct,
        height_pct,
        idle_extra,
    )


def locate_resource_panel(frame, cache_obj: cache.ResourceCache = cache.RESOURCE_CACHE):
    """Locate the resource panel and return bounding boxes for each value."""

    box, _score = detect_hud(frame)
    if not box:
        return {}

    x, y, w, h = box
    panel_gray = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)

    cfg = _get_resource_panel_cfg()
    screen_utils._load_icon_templates()

    detected = {}
    for name in RESOURCE_ICON_ORDER:
        icon = screen_utils.ICON_TEMPLATES.get(name)
        if icon is None:
            continue
        best = (0, None, None)
        for scale in cfg.scales:
            icon_scaled = cv2.resize(icon, None, fx=scale, fy=scale)
            result = cv2.matchTemplate(panel_gray, icon_scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best[0]:
                best = (max_val, max_loc, icon_scaled.shape[::-1])
        if best[0] >= cfg.match_threshold and best[1] is not None:
            bw, bh = best[2]
            detected[name] = (best[1][0], best[1][1], bw, bh)
            cache_obj.last_icon_bounds[name] = (best[1][0], best[1][1], bw, bh)
        elif name in cache_obj.last_icon_bounds:
            logger.info(
                "Using previous position for icon '%s'; score=%.3f", name, best[0]
            )
            detected[name] = cache_obj.last_icon_bounds[name]
        else:
            logger.warning("Icon '%s' not matched; score=%.3f", name, best[0])

    if "population_limit" not in detected and "idle_villager" in detected:
        xi, yi, wi, hi = detected["idle_villager"]
        prev = cache_obj.last_icon_bounds.get("population_limit")
        if prev:
            pw, ph = prev[2], prev[3]
        else:
            pw, ph = wi, hi
        px = max(0, xi - pw)
        detected["population_limit"] = (px, yi, pw, ph)
        cache_obj.last_icon_bounds["population_limit"] = (px, yi, pw, ph)

    top = y + int(cfg.top_pct * h)
    height = int(cfg.height_pct * h)

    regions, spans, narrow = compute_resource_rois(
        x,
        x + w,
        top,
        height,
        cfg.pad_left,
        cfg.pad_right,
        cfg.icon_trims,
        cfg.max_width,
        cfg.min_widths,
        cfg.min_requireds,
        detected,
    )

    cache._NARROW_ROIS = set(narrow.keys())
    cache._NARROW_ROI_DEFICITS = narrow.copy()
    cache._LAST_REGION_SPANS = spans.copy()

    if "idle_villager" in detected:
        xi, yi, wi, hi = detected["idle_villager"]
        extra = cfg.idle_roi_extra_width
        left = x + xi
        width = wi + extra
        right = left + width
        if right > x + w:
            width = (x + w) - left
        regions["idle_villager"] = (left, y + yi, width, hi)
        logger.debug(
            "ROI for 'idle_villager': icon=(%d,%d) width=%d", left, y + yi, width
        )

    if cache._LAST_REGION_BOUNDS != regions:
        cache._LAST_REGION_BOUNDS = regions.copy()
        cache_obj.last_resource_values.clear()
        cache_obj.last_resource_ts.clear()

    return regions


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
    min_widths = [90] * len(RESOURCE_ICON_ORDER)

    regions, spans, narrow = compute_resource_rois(
        left,
        left + width,
        top,
        height,
        pad_left_fallback,
        pad_right_fallback,
        icon_trims_zero,
        width,
        min_widths,
        detected=detected,
    )

    cache._NARROW_ROIS = set(narrow.keys())
    cache._NARROW_ROI_DEFICITS = narrow.copy()

    for name in RESOURCE_ICON_ORDER[:-1]:
        if name in regions:
            l, t, w, hgt = regions[name]
            if w < 90:
                w = 90
                regions[name] = (l, t, w, hgt)
            spans[name] = (l, l + w)

    if "idle_villager" in required_icons:
        idx_iv = RESOURCE_ICON_ORDER.index("idle_villager")
        left_iv = left + int(idx_iv * slice_w + pad_left_fallback[idx_iv])
        right_iv = left + int(width - pad_right_fallback[idx_iv])
        width_iv = max(90, right_iv - left_iv)
        regions["idle_villager"] = (left_iv, top, width_iv, height)
        spans["idle_villager"] = (left_iv, left_iv + width_iv)

    cache._LAST_REGION_SPANS = spans.copy()

    return regions


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
        cfg.max_width,
        cfg.min_widths,
        cfg.min_requireds,
        detected_rel,
    )

    cache._NARROW_ROIS = set(narrow.keys())
    cache._NARROW_ROI_DEFICITS = narrow.copy()
    cache._LAST_REGION_SPANS = spans.copy()

    if "idle_villager" in detected_rel:
        xi, yi, wi, hi = detected_rel["idle_villager"]
        extra = cfg.idle_roi_extra_width
        left = panel_left + xi
        width = wi + extra
        right = left + width
        if right > panel_right:
            width = panel_right - left
        regions["idle_villager"] = (panel_left + xi, panel_top + yi, width, hi)

    if cache._LAST_REGION_BOUNDS != regions:
        cache._LAST_REGION_BOUNDS = regions.copy()
        cache_obj.last_resource_values.clear()
        cache_obj.last_resource_ts.clear()

    return regions


def _apply_custom_rois(frame, regions, names=None, include_idle=True):
    """Apply ROI overrides defined in the configuration."""

    from . import input_utils

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


def detect_resource_regions(
    frame, required_icons, cache_obj: cache.ResourceCache = cache.RESOURCE_CACHE
):
    """Detect resource value regions on the HUD."""

    regions = locate_resource_panel(frame, cache_obj)
    regions = _apply_custom_rois(frame, regions)
    missing = [name for name in required_icons if name not in regions]

    if missing:
        calibrated = _auto_calibrate_from_icons(frame, cache_obj)
        if calibrated:
            regions = calibrated
            if not CFG.get("disable_roi_overrides_on_calibration"):
                regions = _apply_custom_rois(frame, regions, include_idle=False)
            missing = [name for name in required_icons if name not in regions]

    if missing and common.HUD_ANCHOR:
        if common.HUD_ANCHOR.get("asset") == "assets/resources.png":
            x = common.HUD_ANCHOR["left"]
            y = common.HUD_ANCHOR["top"]
            w = common.HUD_ANCHOR["width"]
            h = common.HUD_ANCHOR["height"]

            res_cfg = CFG.get("resource_panel", {})
            profile = CFG.get("profile")
            profile_res = CFG.get("profiles", {}).get(profile, {}).get(
                "resource_panel", {},
            )
            top_pct = profile_res.get("top_pct", res_cfg.get("top_pct", 0.08))
            height_pct = profile_res.get("height_pct", res_cfg.get("height_pct", 0.84))
            icon_trims_cfg = profile_res.get(
                "icon_trim_pct",
                res_cfg.get(
                    "icon_trim_pct",
                    [0.25, 0.20, 0.20, 0.20, 0.20, 0.20],
                ),
            )
            if not isinstance(icon_trims_cfg, (list, tuple)):
                icon_trims_cfg = [icon_trims_cfg] * 6
            right_trim = profile_res.get(
                "right_trim_pct", res_cfg.get("right_trim_pct", 0.02)
            )

            top = y + int(top_pct * h)
            height = int(height_pct * h)

            regions = _fallback_rois_from_slice(
                x,
                w,
                top,
                height,
                icon_trims_cfg,
                right_trim,
                required_icons,
            )
        else:
            from . import input_utils
            W, H = input_utils._screen_size()
            margin = int(0.01 * W)
            panel_w = int(568 / 1920 * W)
            panel_h = int(59 / 1080 * H)
            x = common.HUD_ANCHOR["left"] + common.HUD_ANCHOR["width"] + margin
            y = common.HUD_ANCHOR["top"]

            res_cfg = CFG.get("resource_panel", {})
            profile = CFG.get("profile")
            profile_res = CFG.get("profiles", {}).get(profile, {}).get(
                "resource_panel", {},
            )
            top_pct = profile_res.get(
                "anchor_top_pct", res_cfg.get("anchor_top_pct", 0.15)
            )
            height_pct = profile_res.get(
                "anchor_height_pct", res_cfg.get("anchor_height_pct", 0.70)
            )
            icon_trims = profile_res.get(
                "anchor_icon_trim_pct",
                res_cfg.get(
                    "anchor_icon_trim_pct",
                    [0.42, 0.42, 0.35, 0.35, 0.35, 0.35],
                ),
            )
            if not isinstance(icon_trims, (list, tuple)):
                icon_trims = [icon_trims] * 6
            right_trim = profile_res.get(
                "anchor_right_trim_pct", res_cfg.get("anchor_right_trim_pct", 0.02)
            )

            top = y + int(top_pct * panel_h)
            height = int(height_pct * panel_h)
            regions = _fallback_rois_from_slice(
                x,
                panel_w,
                top,
                height,
                icon_trims,
                right_trim,
                required_icons,
            )

        missing = [name for name in required_icons if name not in regions]

    if not missing:
        regions = _recalibrate_low_variance(frame, regions, required_icons, cache_obj)
        missing = [name for name in required_icons if name not in regions]

    if missing:
        raise common.ResourceReadError(
            "Resource icon(s) not located on HUD: " + ", ".join(missing)
        )
    regions = _remove_overlaps(regions, required_icons)

    return regions


__all__ = [
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
    "ResourcePanelCfg",
]
