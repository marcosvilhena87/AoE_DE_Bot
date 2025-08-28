"""Resource-related helpers extracted from :mod:`script.common`."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import cv2
import pytesseract

from .template_utils import find_template
from .config_utils import load_config
from . import screen_utils, common, input_utils

ROOT = Path(__file__).resolve().parent.parent

CFG = load_config()
logger = logging.getLogger(__name__)

# Order in which resource icons appear on the HUD
RESOURCE_ICON_ORDER = [
    "wood_stockpile",
    "food_stockpile",
    "gold_stockpile",
    "stone_stockpile",
    "population_limit",
    "idle_villager",
]

@dataclass
class ResourceCache:
    """In-memory cache for resource detection state."""

    last_icon_bounds: dict = field(default_factory=dict)
    last_resource_values: dict = field(default_factory=dict)
    last_resource_ts: dict = field(default_factory=dict)
    resource_failure_counts: dict = field(default_factory=dict)


# Shared cache instance used by default
RESOURCE_CACHE = ResourceCache()

# Icons fulfilled from cache on the most recent read
_LAST_READ_FROM_CACHE = set()

# Resource names whose available span fell below the configured minimum width
_NARROW_ROIS = set()

# Maximum age (in seconds) for cached resource values
_RESOURCE_CACHE_TTL = CFG.get("resource_cache_ttl", 1.5)
# Optional hard limit on cache age before rejection
_RESOURCE_CACHE_MAX_AGE = CFG.get("resource_cache_max_age")

# Track last set of regions returned to invalidate cached values
_LAST_REGION_BOUNDS = None
# Track last available spans for each region
_LAST_REGION_SPANS = {}


def detect_hud(frame):
    """Locate the resource panel and return its bounding box."""

    tmpl = screen_utils.HUD_TEMPLATE
    if tmpl is None:
        return None

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
                return None
        else:
            return None

    return box


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
    """Compute resource ROIs from detected icon bounds.

    Parameters
    ----------
    panel_left, panel_right : int
        Horizontal bounds of the resource panel in screen coordinates.
    top, height : int
        Vertical position and height for the ROI regions.
    pad_left, pad_right : Sequence[int]
        Padding to apply after trimming each icon pair.
    icon_trims : Sequence[float | int]
        Trim values applied to icon widths prior to padding. Values in ``[0,1]``
        are treated as percentages.
    max_width : int
        Maximum width for each ROI.
    min_widths : Sequence[int]
        Minimum acceptable width for each ROI before being flagged as narrow.
    min_requireds : Sequence[int] | None, optional
        Minimum width to allocate for each ROI. When provided, each computed
        ``width`` will be at least the corresponding value, bounded by the
        available span.
    detected : dict[str, tuple[int, int, int, int]] | None
        Mapping of icon names to bounding boxes relative to the panel ``(x, y, w, h)``.

    Returns
    -------
    tuple[dict, dict, dict]
        ``(regions, spans, narrow)`` where ``regions`` is a mapping of resource
        names to ROI tuples ``(left, top, width, height)``, ``spans`` contains the
        available left/right span for each resource with a valid ROI, and
        ``narrow`` flags resources whose available span was smaller than the
        configured minimum width.
    """

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
            narrow[current] = True
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


def locate_resource_panel(frame, cache: ResourceCache = RESOURCE_CACHE):
    """Locate the resource panel and return bounding boxes for each value."""

    box = detect_hud(frame)
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
            cache.last_icon_bounds[name] = (best[1][0], best[1][1], bw, bh)
        elif name in cache.last_icon_bounds:
            logger.info(
                "Using previous position for icon '%s'; score=%.3f", name, best[0]
            )
            detected[name] = cache.last_icon_bounds[name]
        else:
            logger.warning("Icon '%s' not matched; score=%.3f", name, best[0])

    if "population_limit" not in detected and "idle_villager" in detected:
        xi, yi, wi, hi = detected["idle_villager"]
        prev = cache.last_icon_bounds.get("population_limit")
        if prev:
            pw, ph = prev[2], prev[3]
        else:
            pw, ph = wi, hi
        px = max(0, xi - pw)
        detected["population_limit"] = (px, yi, pw, ph)
        cache.last_icon_bounds["population_limit"] = (px, yi, pw, ph)

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

    global _NARROW_ROIS, _LAST_REGION_SPANS
    _NARROW_ROIS = set(narrow.keys())
    _LAST_REGION_SPANS = spans.copy()

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

    global _LAST_REGION_BOUNDS
    if _LAST_REGION_BOUNDS != regions:
        _LAST_REGION_BOUNDS = regions.copy()
        cache.last_resource_values.clear()
        cache.last_resource_ts.clear()

    return regions


def _ocr_digits_better(gray):
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    bilateral = CFG.get("ocr_bilateral", [7, 60, 60])
    if bilateral:
        if isinstance(bilateral, dict):
            d = bilateral.get("d", 7)
            sc = bilateral.get("sigmaColor", bilateral.get("sc", 60))
            ss = bilateral.get("sigmaSpace", bilateral.get("ss", 60))
            gray = cv2.bilateralFilter(gray, d, sc, ss)
        elif isinstance(bilateral, (list, tuple)) and len(bilateral) >= 3:
            gray = cv2.bilateralFilter(gray, bilateral[0], bilateral[1], bilateral[2])
        else:
            gray = cv2.bilateralFilter(gray, 7, 60, 60)
    orig = gray.copy()
    equalize = CFG.get("ocr_equalize_hist", True)
    if equalize:
        if isinstance(equalize, dict):
            clip = equalize.get("clipLimit", equalize.get("clip_limit", 2.0))
            tile = tuple(equalize.get("tileGridSize", equalize.get("tile_grid_size", (8, 8))))
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
            gray = clahe.apply(gray)
        else:
            gray = cv2.equalizeHist(gray)

    kernel_size = CFG.get("ocr_kernel_size", 2)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    psms = list(
        dict.fromkeys(CFG.get("ocr_psm_list", []) + [6, 7, 8, 10, 13])
    )

    debug = CFG.get("ocr_debug")
    debug_dir = ROOT / "debug" if debug else None
    ts = int(time.time() * 1000) if debug else None

    def _run_masks(masks, start_idx=0):
        results = []
        for idx, mask in enumerate(masks, start=start_idx):
            if debug:
                debug_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(debug_dir / f"ocr_mask_{ts}_{idx}.png"), mask)
            for psm in psms:
                data = pytesseract.image_to_data(
                    mask,
                    config=f"--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789",
                    output_type=pytesseract.Output.DICT,
                )
                text = "".join(data.get("text", [])).strip()
                digits = "".join(filter(str.isdigit, text))
                results.append((digits, data, mask))
        results.sort(key=lambda r: len(r[0]), reverse=True)
        return results[0]

    def _is_nearly_empty(mask, tol=0.01):
        if mask.size == 0:
            return True
        ratio = cv2.countNonZero(mask) / mask.size
        return ratio < tol or ratio > 1 - tol

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    if _is_nearly_empty(thresh):
        _otsu_ret, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    primary = _run_masks([thresh, cv2.bitwise_not(thresh)], 0)
    if primary[0]:
        return primary

    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    adaptive = cv2.dilate(adaptive, kernel, iterations=1)
    if _is_nearly_empty(adaptive):
        _otsu_ret, adaptive = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    digits, data, mask = _run_masks([adaptive, cv2.bitwise_not(adaptive)], 2)

    if not digits:
        hsv = cv2.cvtColor(cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        digits, data, mask = _run_masks([white_mask, cv2.bitwise_not(white_mask)], 4)

    if not digits:
        variance = float(np.var(orig))
        if variance < CFG.get("ocr_zero_variance", 15.0):
            return "0", {}, mask

    return digits, data, mask


def _auto_calibrate_from_icons(frame, cache: ResourceCache = RESOURCE_CACHE):
    """Locate resource icons directly on the frame to derive ROI regions.

    This acts as a fallback when the resource panel template cannot be
    matched. Icon templates are searched across the full frame and the spans
    between successive icons are converted into numeric regions via
    :func:`compute_resource_rois`.
    """

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
            cache.last_icon_bounds[name] = (best[1][0], best[1][1], bw, bh)
        elif name in cache.last_icon_bounds:
            detected[name] = cache.last_icon_bounds[name]

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

    global _NARROW_ROIS, _LAST_REGION_SPANS, _LAST_REGION_BOUNDS
    _NARROW_ROIS = set(narrow.keys())
    _LAST_REGION_SPANS = spans.copy()

    if "idle_villager" in detected_rel:
        xi, yi, wi, hi = detected_rel["idle_villager"]
        extra = cfg.idle_roi_extra_width
        left = panel_left + xi
        width = wi + extra
        right = left + width
        if right > panel_right:
            width = panel_right - left
        regions["idle_villager"] = (panel_left + xi, panel_top + yi, width, hi)

    if _LAST_REGION_BOUNDS != regions:
        _LAST_REGION_BOUNDS = regions.copy()
        cache.last_resource_values.clear()
        cache.last_resource_ts.clear()

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
    """Construct resource ROIs from generic slice bounds.

    Parameters
    ----------
    left, width : int
        Horizontal starting position and total width of the resource panel.
    top, height : int
        Vertical position and height for the ROI regions.
    icon_trims : Sequence[float | int]
        Percentage trims to apply on the left side of each icon slice.
    right_trim : float | int
        Percentage trim to apply on the right side of each icon slice.
    required_icons : Sequence[str]
        Resource icons that must be present; used to add the idle villager ROI.
    """

    global _NARROW_ROIS, _LAST_REGION_SPANS

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

    _NARROW_ROIS = set(narrow.keys())

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

    _LAST_REGION_SPANS = spans.copy()

    return regions


def detect_resource_regions(
    frame, required_icons, cache: ResourceCache = RESOURCE_CACHE
):
    """Detect resource value regions on the HUD."""

    global _NARROW_ROIS, _LAST_REGION_SPANS

    regions = locate_resource_panel(frame, cache)
    if "idle_villager" not in regions:
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
    custom_names = [
        "wood_stockpile",
        "food_stockpile",
        "gold_stockpile",
        "stone_stockpile",
        "population_limit",
    ]
    W, H = input_utils._screen_size()
    for name in custom_names:
        cfg = CFG.get(f"{name}_roi")
        if not cfg:
            continue
        left = int(cfg.get("left_pct", 0) * W)
        top = int(cfg.get("top_pct", 0) * H)
        width = int(cfg.get("width_pct", 0) * W)
        height = int(cfg.get("height_pct", 0) * H)
        regions[name] = (left, top, width, height)
        logger.debug("Custom ROI applied for %s: %s", name, regions[name])
    missing = [name for name in required_icons if name not in regions]

    if missing:
        calibrated = _auto_calibrate_from_icons(frame, cache)
        if calibrated:
            regions = calibrated
            # Reapply any custom ROI overrides on top of calibrated values
            for name in custom_names:
                cfg = CFG.get(f"{name}_roi")
                if not cfg:
                    continue
                left = int(cfg.get("left_pct", 0) * W)
                top = int(cfg.get("top_pct", 0) * H)
                width = int(cfg.get("width_pct", 0) * W)
                height = int(cfg.get("height_pct", 0) * H)
                regions[name] = (left, top, width, height)
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
            # Fallback: estimate resource bar from HUD anchor
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

    if missing:
        raise common.ResourceReadError(
            "Resource icon(s) not located on HUD: " + ", ".join(missing)
        )

    return regions


def preprocess_roi(roi):
    """Convert ROI to a blurred grayscale image."""

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    kernel = CFG.get("ocr_blur_kernel", 3)
    if kernel:
        gray = cv2.medianBlur(gray, kernel)
    return gray


def execute_ocr(gray, conf_threshold=None, allow_fallback=True):
    """Run OCR on a preprocessed grayscale image.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale ROI image.
    conf_threshold : int | None, optional
        Confidence threshold for accepting digits.
    allow_fallback : bool, optional
        When ``True`` (default) an additional Tesseract ``image_to_string``
        pass is attempted if no digits are detected. Sliding-window retries
        disable this to avoid excessive fallback calls.
    """

    if conf_threshold is None:
        conf_threshold = CFG.get("ocr_conf_threshold", 60)

    digits, data, mask = _ocr_digits_better(gray)

    confidences = [
        int(c)
        for c in data.get("conf", [])
        if c not in ("-1", "") and int(c) >= 0
    ]
    low_conf = False
    if digits and confidences:
        mean_conf = sum(confidences) / len(confidences)
        max_conf = max(confidences)
        if mean_conf == 0 or max_conf == 0:
            logger.debug(
                "Discarding zero-confidence OCR result: mean=%.1f max=%.1f digits=%s",
                mean_conf,
                max_conf,
                digits,
            )
            digits = ""
            low_conf = True
        elif mean_conf < conf_threshold or max_conf < conf_threshold:
            logger.debug(
                "Clearing low-confidence OCR result: mean=%.1f max=%.1f digits=%s",
                mean_conf,
                max_conf,
                digits,
            )
            digits = ""
            low_conf = True
    if low_conf:
        alt_gray = cv2.bitwise_not(gray)
        digits2, data2, mask2 = _ocr_digits_better(alt_gray)
        confidences2 = [
            int(c)
            for c in data2.get("conf", [])
            if c not in ("-1", "") and int(c) >= 0
        ]
        low_conf2 = False
        if digits2 and confidences2:
            mean_conf2 = sum(confidences2) / len(confidences2)
            max_conf2 = max(confidences2)
            if mean_conf2 == 0 or max_conf2 == 0:
                logger.debug(
                    "Discarding zero-confidence OCR result (second attempt): "
                    "mean=%.1f max=%.1f digits=%s",
                    mean_conf2,
                    max_conf2,
                    digits2,
                )
                digits2 = ""
                low_conf2 = True
            elif mean_conf2 < conf_threshold or max_conf2 < conf_threshold:
                logger.debug(
                    "Clearing low-confidence OCR result (second attempt): "
                    "mean=%.1f max=%.1f digits=%s",
                    mean_conf2,
                    max_conf2,
                    digits2,
                )
                digits2 = ""
                low_conf2 = True
        if digits2:
            return digits2, data2, mask2
        digits, data, mask = digits2, data2, mask2
        low_conf = low_conf2

    if not digits and allow_fallback:
        text = pytesseract.image_to_string(
            gray,
            config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789",
        )
        fallback = "".join(filter(str.isdigit, text))
        if fallback:
            digits = fallback
            data = {"text": [text.strip()], "conf": []}
            mask = gray
    if not digits:
        debug_dir = ROOT / "debug"
        debug_dir.mkdir(exist_ok=True)
        ts = int(time.time() * 1000)
        mask_path = None
        if mask is not None:
            mask_path = debug_dir / f"ocr_fail_mask_{ts}.png"
            cv2.imwrite(str(mask_path), mask)
        text_path = debug_dir / f"ocr_fail_text_{ts}.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write("\n".join(data.get("text", [])))
        if mask_path is not None:
            logger.warning(
                "OCR returned no digits; mask saved to %s; text output saved to %s",
                mask_path,
                text_path,
            )
        else:
            logger.warning(
                "OCR returned no digits; text output saved to %s", text_path
            )
    return digits, data, mask


def handle_ocr_failure(
    frame,
    regions,
    results,
    required_icons,
    cache=None,
    retry_limit=None,
    debug_images=None,
):
    """Handle OCR failures by saving debug images and applying fallbacks.

    Parameters
    ----------
    frame : np.ndarray
        Screenshot frame used for OCR.
    regions : dict
        Mapping of icon names to bounding boxes.
    results : dict
        OCR results with ``None`` for failed icons.
    required_icons : Iterable[str]
        Icons that should trigger an exception if OCR fails.
    cache : ResourceCache, optional
        Cache instance tracking failure counts and cached values.
    retry_limit : int, optional
        Number of consecutive failures before falling back to a default value.
    debug_images : dict, optional
        Mapping of icon names to ``(gray, thresh)`` images for debug output.
    """

    if cache is None:
        cache = RESOURCE_CACHE
    if retry_limit is None:
        retry_limit = CFG.get("ocr_retry_limit", 3)

    debug_success = CFG.get("ocr_debug_success")
    failed = [name for name, v in results.items() if v is None]
    if not failed and not debug_success:
        return

    # Apply fallback for resources that have exceeded the retry limit
    for name in list(failed):
        count = cache.resource_failure_counts.get(name, 0)
        if count >= retry_limit:
            fallback = cache.last_resource_values.get(name, 0)
            results[name] = fallback
            logger.warning(
                "Using fallback value %s=%d after %d OCR failures",
                name,
                fallback,
                count,
            )
            failed.remove(name)

    if not failed and not debug_success:
        return

    required_set = set(required_icons)
    required_failed = [f for f in failed if f in required_set]
    optional_failed = [f for f in failed if f not in required_set]

    def _annotate(names):
        return [
            f"{n} (narrow ROI span)" if n in _NARROW_ROIS else n for n in names
        ]

    annotated_required = _annotate(required_failed)
    annotated_optional = _annotate(optional_failed)

    h_full, w_full = frame.shape[:2]
    debug_dir = ROOT / "debug"
    debug_dir.mkdir(exist_ok=True)
    ts = int(time.time() * 1000)

    panel_path = None
    if failed:
        annotated = frame.copy()
        for name, (x, y, w, h) in regions.items():
            if name in failed:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 1)
        panel_path = debug_dir / f"resource_panel_fail_{ts}.png"
        cv2.imwrite(str(panel_path), annotated)

    roi_paths = []
    roi_logs = []
    debug_targets = set(failed)
    if debug_success:
        debug_targets.update(regions.keys())
    for name in debug_targets:
        x, y, w, h = regions[name]
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, w_full), min(y + h, h_full)
        roi = frame[y1:y2, x1:x2]
        if roi.shape[0] != h or roi.shape[1] != w:
            padded = np.zeros((h, w, 3), dtype=frame.dtype)
            pad_y = y1 - y
            pad_x = x1 - x
            padded[pad_y:pad_y + roi.shape[0], pad_x:pad_x + roi.shape[1]] = roi
            roi = padded
        roi_path = debug_dir / f"resource_{name}_roi_{ts}.png"
        cv2.imwrite(str(roi_path), roi)

        gray = mask = None
        if debug_images and name in debug_images:
            gray, mask = debug_images[name]
        else:
            gray = preprocess_roi(roi)
            _, mask = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        gray_path = debug_dir / f"resource_{name}_gray_{ts}.png"
        cv2.imwrite(str(gray_path), gray)
        if mask is not None:
            thresh_path = debug_dir / f"resource_{name}_thresh_{ts}.png"
            cv2.imwrite(str(thresh_path), mask)

        roi_paths.append(str(roi_path))
        roi_logs.append(
            f"{name}: (x={x}, y={y}, w={w}, h={h}) -> {roi_path}"
        )

    if required_failed:
        logger.error(
            "Resource panel OCR failed for %s; panel saved to %s; rois: %s",
            ", ".join(annotated_required),
            panel_path,
            ", ".join(roi_logs),
        )
        tess_path = pytesseract.pytesseract.tesseract_cmd
        paths_str = ", ".join([str(panel_path)] + roi_paths)
        failed_regions = {k: regions[k] for k in required_failed}
        raise common.ResourceReadError(
            "OCR failed to read resource values for "
            + ", ".join(annotated_required)
            + f" (regions={failed_regions}, tesseract_cmd={tess_path}, debug_images={paths_str})",
        )

    if optional_failed:
        logger.warning(
            "Resource panel OCR failed for optional %s; panel saved to %s; rois: %s",
            ", ".join(annotated_optional),
            panel_path,
            ", ".join(roi_logs),
        )


def _extract_population(frame, regions, results, pop_required, conf_threshold=None):
    cur_pop = pop_cap = None
    if "population_limit" in regions:
        x, y, w, h = regions["population_limit"]
        roi = frame[y : y + h, x : x + w]
        try:
            try:
                cur_pop, pop_cap = _read_population_from_roi(
                    roi, conf_threshold=conf_threshold
                )
            except TypeError:  # compatibility with patches lacking new arg
                cur_pop, pop_cap = _read_population_from_roi(roi)
            if results is not None:
                results["population_limit"] = cur_pop
        except common.PopulationReadError:
            if results is not None:
                results["population_limit"] = None
            if pop_required:
                raise
    else:
        if results is not None:
            results["population_limit"] = None
        if pop_required:
            raise common.ResourceReadError("population_limit region not detected")
    return cur_pop, pop_cap

def _read_resources(
    frame,
    required_icons,
    icons_to_read,
    cache: ResourceCache = RESOURCE_CACHE,
    max_cache_age: float | None = None,
    conf_threshold: int | None = None,
):
    """Core routine for reading resource values from a frame."""

    required_icons = list(required_icons)
    icons_to_read = list(icons_to_read)
    required_set = set(required_icons)

    regions = detect_resource_regions(frame, icons_to_read, cache)

    if max_cache_age is None:
        max_cache_age = _RESOURCE_CACHE_MAX_AGE
    if conf_threshold is None:
        conf_threshold = CFG.get("ocr_conf_threshold", 60)

    resource_icons = [n for n in icons_to_read if n != "population_limit"]
    results = {}
    cache_hits = set()
    debug_images = {}
    for name in resource_icons:
        if name not in regions:
            if name in required_set:
                results[name] = None
            continue
        x, y, w, h = regions[name]
        if w <= 0 or h <= 0:
            logger.error(
                "ROI for '%s' has invalid dimensions w=%d h=%d", name, w, h
            )
            if name in required_set:
                raise common.ResourceReadError(
                    f"{name} region has non-positive size"
                )
            continue
        roi = frame[y : y + h, x : x + w]
        gray = preprocess_roi(roi)
        if name == "idle_villager":
            data = pytesseract.image_to_data(
                gray,
                config="--psm 7 -c tessedit_char_whitelist=0123456789",
                output_type=pytesseract.Output.DICT,
            )
            texts = [t for t in data.get("text", []) if t.strip()]
            confidences = [
                float(c)
                for c in data.get("conf", [])
                if c not in ("-1", "")
            ]
            if texts and confidences and all(c >= conf_threshold for c in confidences):
                digits = "".join(filter(str.isdigit, "".join(texts)))
            else:
                digits = None
            mask = gray
        else:
            digits, data, mask = execute_ocr(gray, conf_threshold=conf_threshold)
        if not digits:
            expand_px = CFG.get("ocr_roi_expand_px", 0)
            if expand_px:
                x0 = max(0, x - expand_px)
                y0 = max(0, y - expand_px)
                x1 = min(frame.shape[1], x + w + expand_px)
                y1 = min(frame.shape[0], y + h + expand_px)
                roi_expanded = frame[y0:y1, x0:x1]
                gray_expanded = preprocess_roi(roi_expanded)
                digits_exp, data_exp, mask_exp = execute_ocr(
                    gray_expanded, conf_threshold=conf_threshold
                )
                if digits_exp:
                    digits, data, mask = digits_exp, data_exp, mask_exp
                    roi, gray = roi_expanded, gray_expanded
                    x, y = x0, y0
                    w, h = x1 - x0, y1 - y0

        if not digits:
            span_left, span_right = _LAST_REGION_SPANS.get(name, (x, x + w))
            span_width = span_right - span_left
            cand_widths = [min(w, span_width)]
            cand_widths += [min(span_width, cw) for cw in (64, 56, 48)]
            cand_widths = list(dict.fromkeys(cand_widths))
            for cand_w in cand_widths:
                for anchor in ("left", "center", "right"):
                    if anchor == "left":
                        cand_x = span_left
                    elif anchor == "center":
                        cand_x = span_left + (span_width - cand_w) // 2
                    else:
                        cand_x = span_right - cand_w
                    cand_x = max(span_left, min(cand_x, span_right - cand_w))
                    roi_retry = frame[y : y + h, cand_x : cand_x + cand_w]
                    gray_retry = preprocess_roi(roi_retry)
                    digits_retry, data_retry, mask_retry = execute_ocr(
                        gray_retry,
                        conf_threshold=conf_threshold,
                        allow_fallback=False,
                    )
                    if digits_retry:
                        digits, data, mask = digits_retry, data_retry, mask_retry
                        roi, gray = roi_retry, gray_retry
                        x, w = cand_x, cand_w
                        break
                if digits:
                    break
        debug_images[name] = (gray, mask)
        if CFG.get("ocr_debug"):
            debug_dir = ROOT / "debug"
            debug_dir.mkdir(exist_ok=True)
            ts = int(time.time() * 1000)
            cv2.imwrite(str(debug_dir / f"resource_{name}_roi_{ts}.png"), roi)
            if mask is not None:
                cv2.imwrite(str(debug_dir / f"resource_{name}_thresh_{ts}.png"), mask)
        if not digits:
            logger.warning(
                "OCR failed for %s; raw boxes=%s", name, data.get("text")
            )
            debug_dir = ROOT / "debug"
            debug_dir.mkdir(exist_ok=True)
            ts = int(time.time() * 1000)
            roi_path = debug_dir / f"resource_{name}_roi_{ts}.png"
            cv2.imwrite(str(roi_path), roi)
            logger.warning("Saved ROI image to %s", roi_path)
            if mask is not None:
                thresh_path = debug_dir / f"resource_{name}_thresh_{ts}.png"
                cv2.imwrite(str(thresh_path), mask)
                logger.warning("Saved threshold image to %s", thresh_path)
            ts_cache = cache.last_resource_ts.get(name)
            failure_count = cache.resource_failure_counts.get(name, 0)
            use_cache = False
            if name in cache.last_resource_values and ts_cache is not None:
                age = time.time() - ts_cache
                if age < _RESOURCE_CACHE_TTL:
                    logger.warning(
                        "Using cached value for %s after OCR failure", name
                    )
                    use_cache = True
                elif failure_count >= 1 and (
                    max_cache_age is None or age <= max_cache_age
                ):
                    logger.warning(
                        "Using cached value for %s despite expired TTL (%.2fs)",
                        name,
                        age,
                    )
                    use_cache = True
            if use_cache:
                results[name] = cache.last_resource_values[name]
                cache_hits.add(name)
            else:
                results[name] = None
            cache.resource_failure_counts[name] = failure_count + 1
        else:
            value = int(digits)
            results[name] = value
            cache.last_resource_values[name] = value
            cache.last_resource_ts[name] = time.time()
            cache.resource_failure_counts[name] = 0
            logger.info("Detected %s=%d", name, value)

    filtered_regions = {n: regions[n] for n in resource_icons if n in regions}
    required_for_ocr = [n for n in required_icons if n != "population_limit"]
    handle_ocr_failure(
        frame,
        filtered_regions,
        results,
        required_for_ocr,
        cache,
        debug_images=debug_images,
    )

    cur_pop = pop_cap = None
    if "population_limit" in icons_to_read:
        pop_required = "population_limit" in required_set
        cur_pop, pop_cap = _extract_population(
            frame, regions, results, pop_required, conf_threshold=conf_threshold
        )

    global _LAST_READ_FROM_CACHE
    _LAST_READ_FROM_CACHE = cache_hits

    return results, (cur_pop, pop_cap)

def read_resources_from_hud(
    required_icons: Iterable[str] | None = None,
    icons_to_read: Iterable[str] | None = None,
    force_delay: float | None = None,
    max_cache_age: float | None = None,
    conf_threshold: int | None = None,
    cache: ResourceCache = RESOURCE_CACHE,
):
    """Read resource values displayed on the HUD."""

    if force_delay is not None:
        time.sleep(force_delay)

    frame = screen_utils._grab_frame()

    if required_icons is None:
        required_icons = list(RESOURCE_ICON_ORDER)
    else:
        required_icons = list(required_icons)

    if icons_to_read is None:
        icons_to_read = list(required_icons)
    else:
        icons_to_read = list(set(required_icons).union(icons_to_read))

    return _read_resources(
        frame,
        required_icons,
        icons_to_read,
        cache,
        max_cache_age,
        conf_threshold,
    )


def read_population_from_roi(
    roi_bbox,
    retries: int = 1,
    conf_threshold: int | None = None,
    save_failed_roi: bool = False,
):
    """Read population values from the screen using the provided ROI bbox.

    Parameters
    ----------
    roi_bbox : dict
        Bounding box with ``left``, ``top``, ``width`` and ``height`` keys.
    retries : int, optional
        Number of OCR attempts before giving up.
    conf_threshold : int | None, optional
        Minimum confidence accepted for OCR characters. When ``None`` the
        configuration value is used.
    save_failed_roi : bool, optional
        Force saving of the failed ROI even when debugging is disabled.
    """

    last_exc = None
    for attempt in range(retries):
        roi = screen_utils._grab_frame(roi_bbox)
        try:
            return _read_population_from_roi(
                roi,
                conf_threshold=conf_threshold,
                save_debug=(
                    attempt == retries - 1
                    and (CFG.get("debug") or save_failed_roi)
                ),
            )
        except common.PopulationReadError as exc:
            last_exc = exc
            if attempt < retries - 1:
                logger.debug("OCR attempt %s failed: %s", attempt + 1, exc)
                time.sleep(0.1)

    raise last_exc


def _read_population_from_roi(roi, conf_threshold=None, save_debug=True):
    """Read current and capacity population values from a ROI.

    Parameters
    ----------
    roi : np.ndarray
        Image region containing the population text.
    conf_threshold : int | None, optional
        Minimum confidence value accepted for OCR characters. When ``None``
        the value from configuration is used.
    save_debug : bool, optional
        When ``True`` failed OCR attempts will write debug images to disk.
    """

    if conf_threshold is None:
        conf_threshold = CFG.get("ocr_conf_threshold", 60)

    if roi.size == 0:
        raise common.PopulationReadError("Population ROI has zero size")

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    data = pytesseract.image_to_data(
        thresh,
        config="--psm 7 -c tessedit_char_whitelist=0123456789/",
        output_type=pytesseract.Output.DICT,
    )
    text = "".join(data.get("text", [])).replace(" ", "")
    confidences = [int(c) for c in data.get("conf", []) if c != "-1"]
    parts = [p for p in text.split("/") if p]
    if len(parts) >= 2 and (not confidences or min(confidences) >= conf_threshold):
        cur = int("".join(filter(str.isdigit, parts[0])) or 0)
        cap = int("".join(filter(str.isdigit, parts[1])) or 0)
        return cur, cap

    logger.warning(
        "OCR failed for population; text='%s', conf=%s", text, confidences
    )

    if save_debug:
        debug_dir = ROOT / "debug"
        debug_dir.mkdir(exist_ok=True)
        ts = int(time.time() * 1000)
        cv2.imwrite(str(debug_dir / f"population_roi_{ts}.png"), roi)
        cv2.imwrite(str(debug_dir / f"population_thresh_{ts}.png"), thresh)
    raise common.PopulationReadError(
        f"Failed to read population from HUD: text='{text}', confs={confidences}"
    )


def gather_hud_stats(
    force_delay=None,
    required_icons=None,
    optional_icons=None,
    max_cache_age: float | None = None,
    conf_threshold: int | None = None,
    cache: ResourceCache = RESOURCE_CACHE,
):
    """Capture a single frame and read resources and population."""

    if force_delay is not None:
        time.sleep(force_delay)

    frame = screen_utils._grab_frame()

    icon_cfg = CFG.get("hud_icons", {})
    provided_lists = not (required_icons is None and optional_icons is None)
    if required_icons is None:
        required_icons = icon_cfg.get(
            "required",
            [
                "wood_stockpile",
                "food_stockpile",
                "gold_stockpile",
                "stone_stockpile",
                "population_limit",
                "idle_villager",
            ],
        )
    if optional_icons is None:
        optional_icons = icon_cfg.get("optional", [])

    required_icons = list(required_icons)
    optional_icons = list(optional_icons)
    all_icons = list(dict.fromkeys(required_icons + optional_icons))

    if not provided_lists:
        regions = detect_resource_regions(frame, all_icons, cache)
        required_icons = [n for n in required_icons if n in regions]
        all_icons = list(regions.keys())

    return _read_resources(
        frame,
        required_icons,
        all_icons,
        cache,
        max_cache_age,
        conf_threshold,
    )
