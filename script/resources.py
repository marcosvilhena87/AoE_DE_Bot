"""Resource-related helpers extracted from :mod:`script.common`."""

import logging
import time
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

# Cache of last detected icon positions
_LAST_ICON_BOUNDS = {}

# Cache of last successfully read resource values
_LAST_RESOURCE_VALUES = {}
# Timestamp of last update for each cached resource value
_LAST_RESOURCE_TS = {}
# Consecutive OCR failure counts for each resource
_RESOURCE_FAILURE_COUNTS = {}

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
    detected,
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
    detected : dict[str, tuple[int, int, int, int]]
        Mapping of icon names to bounding boxes relative to the panel ``(x, y, w, h)``.

    Returns
    -------
    tuple[dict, dict, dict]
        ``(regions, spans, narrow)`` where ``regions`` is a mapping of resource
        names to ROI tuples ``(left, top, width, height)``, ``spans`` contains the
        available left/right span for each resource, and ``narrow`` flags resources
        whose available span was smaller than the configured minimum width.
    """

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
        spans[current] = (left, right)

        if right <= left:
            logger.warning(
                "Skipping ROI for icon '%s' due to non-positive span (left=%d, right=%d)",
                current,
                left,
                right,
            )
            continue

        available_width = right - left
        width = min(max_width, available_width)

        min_w = min_widths[idx] if idx < len(min_widths) else min_widths[-1]
        if available_width < min_w:
            width = available_width
            narrow[current] = True
            logger.warning(
                "ROI estreita para '%s': disp=%d min=%d",
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



def locate_resource_panel(frame):
    """Locate the resource panel and return bounding boxes for each value."""

    box = detect_hud(frame)
    if not box:
        return {}

    x, y, w, h = box
    panel_gray = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)

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
    pad_left = pad_left if isinstance(pad_left, (list, tuple)) else [pad_left] * 6
    pad_right = pad_right if isinstance(pad_right, (list, tuple)) else [pad_right] * 6
    icon_trims = profile_res.get(
        "icon_trim_pct",
        res_cfg.get("icon_trim_pct", [0, 0, 0, 0, 0, 0]),
    )
    icon_trims = (
        icon_trims if isinstance(icon_trims, (list, tuple)) else [icon_trims] * 6
    )
    max_width = res_cfg.get("max_width", 160)
    min_width_cfg = res_cfg.get("min_width", 90)
    min_widths = (
        min_width_cfg
        if isinstance(min_width_cfg, (list, tuple))
        else [min_width_cfg] * 6
    )
    top_pct = profile_res.get("top_pct", res_cfg.get("top_pct", 0.08))
    height_pct = profile_res.get("height_pct", res_cfg.get("height_pct", 0.84))
    screen_utils._load_icon_templates()

    detected = {}
    for name in RESOURCE_ICON_ORDER:
        icon = screen_utils.ICON_TEMPLATES.get(name)
        if icon is None:
            continue
        best = (0, None, None)
        for scale in scales:
            icon_scaled = cv2.resize(icon, None, fx=scale, fy=scale)
            result = cv2.matchTemplate(panel_gray, icon_scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best[0]:
                best = (max_val, max_loc, icon_scaled.shape[::-1])
        if best[0] >= match_threshold and best[1] is not None:
            bw, bh = best[2]
            detected[name] = (best[1][0], best[1][1], bw, bh)
            _LAST_ICON_BOUNDS[name] = (best[1][0], best[1][1], bw, bh)
        elif name in _LAST_ICON_BOUNDS:
            logger.info(
                "Using previous position for icon '%s'; score=%.3f", name, best[0]
            )
            detected[name] = _LAST_ICON_BOUNDS[name]
        else:
            logger.warning("Icon '%s' not matched; score=%.3f", name, best[0])

    if "population_limit" not in detected and "idle_villager" in detected:
        xi, yi, wi, hi = detected["idle_villager"]
        prev = _LAST_ICON_BOUNDS.get("population_limit")
        if prev:
            pw, ph = prev[2], prev[3]
        else:
            pw, ph = wi, hi
        px = max(0, xi - pw)
        detected["population_limit"] = (px, yi, pw, ph)
        _LAST_ICON_BOUNDS["population_limit"] = (px, yi, pw, ph)

    top = y + int(top_pct * h)
    height = int(height_pct * h)

    regions, spans, narrow = compute_resource_rois(
        x,
        x + w,
        top,
        height,
        pad_left,
        pad_right,
        icon_trims,
        max_width,
        min_widths,
        detected,
    )

    global _NARROW_ROIS, _LAST_REGION_SPANS
    _NARROW_ROIS = set(narrow.keys())
    _LAST_REGION_SPANS = spans.copy()

    if "idle_villager" in detected:
        xi, yi, wi, hi = detected["idle_villager"]
        extra = res_cfg.get("idle_roi_extra_width", 0)
        left = x + xi
        width = wi + extra
        right = left + width
        if right > x + w:
            width = (x + w) - left
        regions["idle_villager"] = (left, y + yi, width, hi)
        logger.debug(
            "ROI for 'idle_villager': icon=(%d,%d) width=%d", left, y + yi, width
        )

    global _LAST_REGION_BOUNDS, _LAST_RESOURCE_VALUES, _LAST_RESOURCE_TS
    if _LAST_REGION_BOUNDS != regions:
        _LAST_REGION_BOUNDS = regions.copy()
        _LAST_RESOURCE_VALUES.clear()
        _LAST_RESOURCE_TS.clear()

    return regions


def _ocr_digits_better(gray):
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    gray = cv2.bilateralFilter(gray, 7, 60, 60)

    kernel_size = CFG.get("ocr_kernel_size", 2)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    psms = CFG.get("ocr_psm_list", [6, 7, 8, 10, 13])

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

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    primary = _run_masks([thresh, cv2.bitwise_not(thresh)], 0)
    if primary[0]:
        return primary

    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    adaptive = cv2.dilate(adaptive, kernel, iterations=1)
    return _run_masks([adaptive, cv2.bitwise_not(adaptive)], 2)


def detect_resource_regions(frame, required_icons):
    """Detect resource value regions on the HUD."""

    global _NARROW_ROIS, _LAST_REGION_SPANS

    regions = locate_resource_panel(frame)
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
                "Custom ROI aplicada para idle_villager: %s", regions["idle_villager"]
            )
    missing = [name for name in required_icons if name not in regions]

    if missing and common.HUD_ANCHOR:
        if common.HUD_ANCHOR.get("asset") == "assets/resources.png":
            x = common.HUD_ANCHOR["left"]
            y = common.HUD_ANCHOR["top"]
            w = common.HUD_ANCHOR["width"]
            h = common.HUD_ANCHOR["height"]

            slice_w = w / 6
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

            detected = {
                name: (int(idx * slice_w), 0, 0, 0)
                for idx, name in enumerate(RESOURCE_ICON_ORDER)
            }
            pad_left_fallback = [
                int(
                    round(
                        (icon_trims_cfg[idx] if idx < len(icon_trims_cfg) else icon_trims_cfg[-1])
                        * slice_w
                    )
                )
                for idx in range(len(RESOURCE_ICON_ORDER))
            ]
            pad_right_fallback = [int(round(right_trim * slice_w))] * len(
                RESOURCE_ICON_ORDER
            )
            icon_trims_zero = [0] * len(RESOURCE_ICON_ORDER)
            min_widths = [90] * len(RESOURCE_ICON_ORDER)
            regions, spans, narrow = compute_resource_rois(
                x,
                x + w,
                top,
                height,
            pad_left_fallback,
            pad_right_fallback,
            icon_trims_zero,
            w,
            min_widths,
            detected,
            )
            _NARROW_ROIS = set(narrow.keys())
            for name in RESOURCE_ICON_ORDER[:-1]:
                if name in regions:
                    l, t, width, hgt = regions[name]
                    if width < 90:
                        width = 90
                        regions[name] = (l, t, width, hgt)
                    spans[name] = (l, l + width)
            if "idle_villager" in required_icons:
                idx_iv = RESOURCE_ICON_ORDER.index("idle_villager")
                left_iv = x + int(idx_iv * slice_w + pad_left_fallback[idx_iv])
                right_iv = x + int(w - pad_right_fallback[idx_iv])
                width_iv = max(90, right_iv - left_iv)
                regions["idle_villager"] = (left_iv, top, width_iv, height)
                spans["idle_villager"] = (left_iv, left_iv + width_iv)
            _LAST_REGION_SPANS = spans.copy()
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

            slice_w = panel_w / 6
            top = y + int(top_pct * panel_h)
            height = int(height_pct * panel_h)
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
            pad_right_fallback = [int(round(right_trim * slice_w))] * len(
                RESOURCE_ICON_ORDER
            )
            icon_trims_zero = [0] * len(RESOURCE_ICON_ORDER)
            min_widths = [90] * len(RESOURCE_ICON_ORDER)
            regions, spans, narrow = compute_resource_rois(
                x,
                x + panel_w,
                top,
                height,
            pad_left_fallback,
            pad_right_fallback,
            icon_trims_zero,
            panel_w,
            min_widths,
            detected,
            )
            if "idle_villager" in required_icons:
                idx_iv = RESOURCE_ICON_ORDER.index("idle_villager")
                left_iv = x + int(idx_iv * slice_w + pad_left_fallback[idx_iv])
                right_iv = x + int(panel_w - pad_right_fallback[idx_iv])
                width_iv = max(90, right_iv - left_iv)
                regions["idle_villager"] = (left_iv, top, width_iv, height)
                spans["idle_villager"] = (left_iv, left_iv + width_iv)
            _NARROW_ROIS = set(narrow.keys())
            for name in RESOURCE_ICON_ORDER[:-1]:
                if name in regions:
                    l, t, width, hgt = regions[name]
                    if width < 90:
                        width = 90
                        regions[name] = (l, t, width, hgt)
                    spans[name] = (l, l + width)
            _LAST_REGION_SPANS = spans.copy()

        missing = [name for name in required_icons if name not in regions]

    if missing:
        raise common.ResourceReadError(
            "Resource icon(s) not located on HUD: " + ", ".join(missing)
        )

    return regions


def preprocess_roi(roi):
    """Convert ROI to a blurred grayscale image."""

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.medianBlur(gray, 3)


def execute_ocr(gray, conf_threshold=None):
    """Run OCR on a preprocessed grayscale image."""

    if conf_threshold is None:
        conf_threshold = CFG.get("ocr_conf_threshold", 60)

    digits, data, mask = _ocr_digits_better(gray)

    confidences = [int(c) for c in data.get("conf", []) if c != "-1"]
    low_conf = False
    if digits and confidences:
        mean_conf = sum(confidences) / len(confidences)
        max_conf = max(confidences)
        if mean_conf < conf_threshold or max_conf < conf_threshold:
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
        confidences2 = [int(c) for c in data2.get("conf", []) if c != "-1"]
        low_conf2 = False
        if digits2 and confidences2:
            mean_conf2 = sum(confidences2) / len(confidences2)
            max_conf2 = max(confidences2)
            if mean_conf2 < conf_threshold or max_conf2 < conf_threshold:
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

    if not digits:
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


def handle_ocr_failure(frame, regions, results, required_icons):
    """Handle OCR failures by saving debug images.

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
    """

    failed = [name for name, v in results.items() if v is None]
    if not failed:
        return

    required_set = set(required_icons)
    required_failed = [f for f in failed if f in required_set]
    optional_failed = [f for f in failed if f not in required_set]

    h_full, w_full = frame.shape[:2]
    debug_dir = ROOT / "debug"
    debug_dir.mkdir(exist_ok=True)
    ts = int(time.time() * 1000)

    annotated = frame.copy()
    for name, (x, y, w, h) in regions.items():
        if name in failed:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 1)
    panel_path = debug_dir / f"resource_panel_fail_{ts}.png"
    cv2.imwrite(str(panel_path), annotated)

    roi_paths = []
    roi_logs = []
    for name in failed:
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
        roi_paths.append(str(roi_path))
        roi_logs.append(f"{name}:{regions[name]} -> {roi_path}")

    if required_failed:
        logger.error(
            "Resource panel OCR failed for %s; panel saved to %s; rois: %s",
            ", ".join(required_failed),
            panel_path,
            ", ".join(roi_logs),
        )
        tess_path = pytesseract.pytesseract.tesseract_cmd
        paths_str = ", ".join([str(panel_path)] + roi_paths)
        failed_regions = {k: regions[k] for k in required_failed}
        raise common.ResourceReadError(
            "OCR failed to read resource values for "
            + ", ".join(required_failed)
            + f" (regions={failed_regions}, tesseract_cmd={tess_path}, debug_images={paths_str})",
        )

    if optional_failed:
        logger.warning(
            "Resource panel OCR failed for optional %s; panel saved to %s; rois: %s",
            ", ".join(optional_failed),
            panel_path,
            ", ".join(roi_logs),
        )


def _extract_population(frame, regions, results, pop_required):
    cur_pop = pop_cap = None
    if "population_limit" in regions:
        x, y, w, h = regions["population_limit"]
        roi = frame[y : y + h, x : x + w]
        try:
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

def read_resources_from_hud(
    required_icons: Iterable[str] | None = None,
    icons_to_read: Iterable[str] | None = None,
    force_delay: float | None = None,
    max_cache_age: float | None = None,
):
    """Read resource values displayed on the HUD.

    Parameters
    ----------
    required_icons : Iterable[str] | None, optional
        Icons that must be successfully read or a
        :class:`common.ResourceReadError` is raised. When ``None`` the
        default set of resource icons is used.
    icons_to_read : Iterable[str] | None, optional
        Additional icons to attempt reading beyond those required. If
        ``None`` only ``required_icons`` are read.
    force_delay : float | None, optional
        When provided, sleep for the given amount of seconds before
        grabbing a frame from the screen. This is useful when a hotkey has
        just been pressed and the HUD may need a short time to update.
    max_cache_age : float | None, optional
        Maximum age (in seconds) for any cached value to be considered when
        multiple consecutive OCR failures occur. ``None`` disables the limit.
    """

    if force_delay is not None:
        time.sleep(force_delay)

    frame = screen_utils._grab_frame()
    if required_icons is None:
        required_icons = [
            "wood_stockpile",
            "food_stockpile",
            "gold_stockpile",
            "stone_stockpile",
            "population_limit",
            "idle_villager",
        ]
    else:
        required_icons = list(required_icons)

    if icons_to_read is None:
        icons_to_read = list(required_icons)
    else:
        icons_to_read = list(set(required_icons).union(icons_to_read))

    required_set = set(required_icons)

    regions = detect_resource_regions(frame, required_icons)

    if max_cache_age is None:
        max_cache_age = _RESOURCE_CACHE_MAX_AGE

    resource_icons = [n for n in icons_to_read if n != "population_limit"]
    results = {}
    cache_hits = set()
    for name in resource_icons:
        if name not in regions:
            continue
        x, y, w, h = regions[name]
        roi = frame[y : y + h, x : x + w]
        gray = preprocess_roi(roi)
        if name == "idle_villager":
            text = pytesseract.image_to_string(
                gray,
                config="--psm 7 -c tessedit_char_whitelist=0123456789",
            )
            digits = "".join(filter(str.isdigit, text))
            data = {"text": [text.strip()], "conf": []}
            mask = gray
        else:
            digits, data, mask = execute_ocr(gray)
        if not digits:
            expand = 24
            span_left, span_right = _LAST_REGION_SPANS.get(name, (x, x + w))
            expand_left = min(expand, max(0, x - span_left))
            expand_right = min(expand, max(0, span_right - (x + w)))
            x1 = max(0, x - expand_left)
            x2 = min(frame.shape[1], x + w + expand_right)
            roi_retry = frame[y : y + h, x1:x2]
            gray_retry = preprocess_roi(roi_retry)
            digits_retry, data_retry, mask_retry = execute_ocr(gray_retry)
            if digits_retry:
                digits, data, mask = digits_retry, data_retry, mask_retry
                roi, gray = roi_retry, gray_retry
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
            ts_cache = _LAST_RESOURCE_TS.get(name)
            failure_count = _RESOURCE_FAILURE_COUNTS.get(name, 0)
            use_cache = False
            if name in _LAST_RESOURCE_VALUES and ts_cache is not None:
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
                results[name] = _LAST_RESOURCE_VALUES[name]
                cache_hits.add(name)
            else:
                results[name] = None
            _RESOURCE_FAILURE_COUNTS[name] = failure_count + 1
        else:
            value = int(digits)
            results[name] = value
            _LAST_RESOURCE_VALUES[name] = value
            _LAST_RESOURCE_TS[name] = time.time()
            _RESOURCE_FAILURE_COUNTS[name] = 0
            logger.info("Detected %s=%d", name, value)

    filtered_regions = {n: regions[n] for n in resource_icons if n in regions}
    required_for_ocr = [n for n in required_icons if n != "population_limit"]
    handle_ocr_failure(frame, filtered_regions, results, required_for_ocr)
    cur_pop = pop_cap = None
    if "population_limit" in icons_to_read:
        pop_required = "population_limit" in required_set
        cur_pop, pop_cap = _extract_population(frame, regions, results, pop_required)
    global _LAST_READ_FROM_CACHE
    _LAST_READ_FROM_CACHE = cache_hits
    return results, (cur_pop, pop_cap)


def _read_population_from_roi(roi, conf_threshold=None):
    """Read current and capacity population values from a ROI.

    Parameters
    ----------
    roi : np.ndarray
        Image region containing the population text.
    conf_threshold : int | None, optional
        Minimum confidence value accepted for OCR characters. When ``None``
        the value from configuration is used.
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
    debug_dir = ROOT / "debug"
    debug_dir.mkdir(exist_ok=True)
    ts = int(time.time() * 1000)
    cv2.imwrite(str(debug_dir / f"population_roi_{ts}.png"), roi)
    cv2.imwrite(str(debug_dir / f"population_thresh_{ts}.png"), thresh)
    raise common.PopulationReadError(
        f"Falha ao ler população da HUD: texto='{text}', confs={confidences}"
    )


def gather_hud_stats(
    force_delay=None,
    required_icons=None,
    optional_icons=None,
    max_cache_age: float | None = None,
):
    """Capture a single frame and read resources and population.

    Parameters
    ----------
    force_delay : float | None, optional
        Seconds to sleep before grabbing the screen.
    required_icons : Iterable[str] | None, optional
        Icons that must be read successfully. When ``None`` the value is
        pulled from configuration.
    optional_icons : Iterable[str] | None, optional
        Icons that should be read when available but are not required.
    max_cache_age : float | None, optional
        Maximum age (in seconds) for any cached value to be considered when
        multiple consecutive OCR failures occur. ``None`` disables the limit.

    Returns
    -------
    tuple
        ``(resources, (current_pop, pop_cap))``
    """

    if force_delay is not None:
        time.sleep(force_delay)

    frame = screen_utils._grab_frame()

    icon_cfg = CFG.get("hud_icons", {})
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

    regions = detect_resource_regions(frame, all_icons)

    if max_cache_age is None:
        max_cache_age = _RESOURCE_CACHE_MAX_AGE

    resource_icons = [name for name in all_icons if name != "population_limit"]

    results = {}
    cache_hits = set()
    for name in resource_icons:
        if name not in regions:
            if name in required_icons:
                results[name] = None
            continue
        x, y, w, h = regions[name]
        roi = frame[y : y + h, x : x + w]
        gray = preprocess_roi(roi)
        digits, data, mask = execute_ocr(gray)
        if not digits:
            expand = 24
            span_left, span_right = _LAST_REGION_SPANS.get(name, (x, x + w))
            expand_left = min(expand, max(0, x - span_left))
            expand_right = min(expand, max(0, span_right - (x + w)))
            x1 = max(0, x - expand_left)
            x2 = min(frame.shape[1], x + w + expand_right)
            roi_retry = frame[y : y + h, x1:x2]
            gray_retry = preprocess_roi(roi_retry)
            digits_retry, data_retry, mask_retry = execute_ocr(gray_retry)
            if digits_retry:
                digits, data, mask = digits_retry, data_retry, mask_retry
                roi, gray = roi_retry, gray_retry
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
            cv2.imwrite(str(debug_dir / f"resource_{name}_roi_{ts}.png"), roi)
            if mask is not None:
                cv2.imwrite(str(debug_dir / f"resource_{name}_thresh_{ts}.png"), mask)
            ts_cache = _LAST_RESOURCE_TS.get(name)
            failure_count = _RESOURCE_FAILURE_COUNTS.get(name, 0)
            use_cache = False
            if name in _LAST_RESOURCE_VALUES and ts_cache is not None:
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
                results[name] = _LAST_RESOURCE_VALUES[name]
                cache_hits.add(name)
            else:
                results[name] = None
            _RESOURCE_FAILURE_COUNTS[name] = failure_count + 1
        else:
            value = int(digits)
            results[name] = value
            _LAST_RESOURCE_VALUES[name] = value
            _LAST_RESOURCE_TS[name] = time.time()
            _RESOURCE_FAILURE_COUNTS[name] = 0
            logger.info("Detected %s=%d", name, value)

    filtered_regions = {n: regions[n] for n in resource_icons if n in regions}
    required_for_ocr = [n for n in required_icons if n != "population_limit"]
    handle_ocr_failure(frame, filtered_regions, results, required_for_ocr)

    cur_pop = pop_cap = None
    if "population_limit" in all_icons:
        pop_required = "population_limit" in required_icons
        cur_pop, pop_cap = _extract_population(frame, regions, results, pop_required)

    global _LAST_READ_FROM_CACHE
    _LAST_READ_FROM_CACHE = cache_hits

    return results, (cur_pop, pop_cap)
