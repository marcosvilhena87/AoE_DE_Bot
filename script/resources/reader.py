"""Functions for reading resource values from the HUD."""

import time
from typing import Iterable

import cv2
import numpy as np
import pytesseract

from . import CFG, ROOT, cache, common, logger, screen_utils, RESOURCE_ICON_ORDER
from .panel import detect_resource_regions
from . import ocr

# Re-export cache utilities
ResourceCache = cache.ResourceCache
RESOURCE_CACHE = cache.RESOURCE_CACHE
_LAST_READ_FROM_CACHE = cache._LAST_READ_FROM_CACHE
_NARROW_ROIS = cache._NARROW_ROIS
_NARROW_ROI_DEFICITS = cache._NARROW_ROI_DEFICITS
_RESOURCE_CACHE_TTL = cache._RESOURCE_CACHE_TTL
_RESOURCE_DEBUG_COOLDOWN = cache._RESOURCE_DEBUG_COOLDOWN
_LAST_REGION_BOUNDS = cache._LAST_REGION_BOUNDS
_LAST_REGION_SPANS = cache._LAST_REGION_SPANS

# Track last OCR failure reasons
_LAST_LOW_CONFIDENCE: set[str] = set()
_LAST_NO_DIGITS: set[str] = set()

# OCR helpers
preprocess_roi = ocr.preprocess_roi
_ocr_digits_better = ocr._ocr_digits_better
execute_ocr = ocr.execute_ocr
handle_ocr_failure = ocr.handle_ocr_failure
_read_population_from_roi = ocr._read_population_from_roi
read_population_from_roi = ocr.read_population_from_roi
_extract_population = ocr._extract_population

def _read_resources(
    frame,
    required_icons,
    icons_to_read,
    cache_obj: ResourceCache = RESOURCE_CACHE,
    max_cache_age: float | None = None,
    conf_threshold: int | None = None,
):
    required_icons = list(required_icons)
    icons_to_read = list(icons_to_read)
    required_set = set(required_icons)

    regions = detect_resource_regions(frame, icons_to_read, cache_obj)

    if max_cache_age is None:
        max_cache_age = cache._RESOURCE_CACHE_MAX_AGE
    if conf_threshold is None:
        conf_threshold = CFG.get(
            "population_limit_ocr_conf_threshold",
            CFG.get("ocr_conf_threshold", 60),
        )

    resource_icons = [n for n in icons_to_read if n != "population_limit"]
    results: dict[str, int | None] = {}
    cache_hits = set()
    debug_images = {}
    low_confidence = set()
    no_digits = set()
    for name in resource_icons:
        res_conf_threshold = CFG.get(f"{name}_ocr_conf_threshold", conf_threshold)
        if name not in regions:
            if name in required_set:
                results[name] = None
                no_digits.add(name)
            continue
        x, y, w, h = regions[name]
        deficit = cache._NARROW_ROI_DEFICITS.get(name)
        if deficit:
            expand_left = deficit // 2
            expand_right = deficit - expand_left
            orig_x = x
            orig_w = w
            x = max(0, orig_x - expand_left)
            right = min(frame.shape[1], orig_x + orig_w + expand_right)
            w = right - x
            regions[name] = (x, y, w, h)
            logger.debug(
                "Expanding narrow ROI for %s by %dpx (left=%d right=%d)",
                name,
                deficit,
                expand_left,
                expand_right,
            )
        if w <= 0 or h <= 0:
            logger.error(
                "ROI for '%s' has invalid dimensions w=%d h=%d", name, w, h
            )
            if name in required_set:
                raise common.ResourceReadError(
                    f"{name} region has non-positive size"
                )
            continue
        failure_count = cache_obj.resource_failure_counts.get(name, 0)
        roi = frame[y : y + h, x : x + w]
        gray = preprocess_roi(roi)
        top_crop = CFG.get("ocr_top_crop", 2)
        overrides = CFG.get("ocr_top_crop_overrides", {})
        if name in overrides:
            top_crop = overrides[name]
        if top_crop > 0 and gray.shape[0] > top_crop:
            gray = gray[top_crop:, :]
        if name == "idle_villager":
            data = pytesseract.image_to_data(
                gray,
                config="--psm 7 -c tessedit_char_whitelist=0123456789",
                output_type=pytesseract.Output.DICT,
            )
            texts = [t for t in data.get("text", []) if t.strip()]
            confidences = ocr.parse_confidences(data)
            if texts and confidences and all(
                c >= res_conf_threshold for c in confidences
            ):
                digits = "".join(filter(str.isdigit, "".join(texts)))
            else:
                digits = None
            mask = gray
            low_conf = False
        else:
            try:
                digits, data, mask, low_conf = execute_ocr(
                    gray,
                    color=roi,
                    conf_threshold=res_conf_threshold,
                    roi=(x, y, w, h),
                    resource=name,
                )
                logger.info(
                    "OCR %s: digits=%s conf=%s low_conf=%s",
                    name,
                    digits,
                    data.get("conf"),
                    low_conf,
                )
            except TypeError:
                digits, data, mask, low_conf = execute_ocr(
                    gray,
                    conf_threshold=res_conf_threshold,
                    resource=name,
                )
                logger.info(
                    "OCR %s: digits=%s conf=%s low_conf=%s",
                    name,
                    digits,
                    data.get("conf"),
                    low_conf,
                )
            if (
                name == "wood_stockpile"
                and low_conf
                and digits
                and name not in cache_obj.last_resource_values
            ):
                min_conf = CFG.get("ocr_conf_min", 0)
                if res_conf_threshold > min_conf:
                    try:
                        digits_retry, data_retry, mask_retry, low_conf = execute_ocr(
                            gray,
                            color=roi,
                            conf_threshold=min_conf,
                            roi=(x, y, w, h),
                            resource=name,
                        )
                        logger.info(
                            "Retry OCR %s: digits=%s conf=%s low_conf=%s",
                            name,
                            digits_retry,
                            data_retry.get("conf"),
                            low_conf,
                        )
                    except TypeError:
                        digits_retry, data_retry, mask_retry, low_conf = execute_ocr(
                            gray,
                            conf_threshold=min_conf,
                            resource=name,
                        )
                        logger.info(
                            "Retry OCR %s: digits=%s conf=%s low_conf=%s",
                            name,
                            digits_retry,
                            data_retry.get("conf"),
                            low_conf,
                        )
                    if digits_retry:
                        digits, data, mask = digits_retry, data_retry, mask_retry
        if not digits:
            base_expand = CFG.get("ocr_roi_expand_px", 0)
            step = CFG.get("ocr_roi_expand_step", 0)
            growth = CFG.get("ocr_roi_expand_growth", 1.0)
            expand_px = int(
                round(
                    base_expand
                    + step * ((failure_count + 1) ** growth - 1)
                )
            )
            if expand_px > 0:
                x0 = max(0, x - expand_px)
                y0 = max(0, y - expand_px)
                x1 = min(frame.shape[1], x + w + expand_px)
                y1 = min(frame.shape[0], y + h + expand_px)
                logger.debug(
                    "Expanding ROI for %s after %d failures by %dpx to x=%d y=%d w=%d h=%d",
                    name,
                    failure_count,
                    expand_px,
                    x0,
                    y0,
                    x1 - x0,
                    y1 - y0,
                )
                roi_expanded = frame[y0:y1, x0:x1]
                gray_expanded = preprocess_roi(roi_expanded)
                if top_crop > 0 and gray_expanded.shape[0] > top_crop:
                    gray_expanded = gray_expanded[top_crop:, :]
                try:
                    digits_exp, data_exp, mask_exp, low_conf = execute_ocr(
                        gray_expanded,
                        color=roi_expanded,
                        conf_threshold=res_conf_threshold,
                        roi=(x0, y0, x1 - x0, y1 - y0),
                        resource=name,
                    )
                    logger.info(
                        "OCR %s: digits=%s conf=%s low_conf=%s",
                        name,
                        digits_exp,
                        data_exp.get("conf"),
                        low_conf,
                    )
                except TypeError:
                    digits_exp, data_exp, mask_exp, low_conf = execute_ocr(
                        gray_expanded,
                        conf_threshold=res_conf_threshold,
                        resource=name,
                    )
                    logger.info(
                        "OCR %s: digits=%s conf=%s low_conf=%s",
                        name,
                        digits_exp,
                        data_exp.get("conf"),
                        low_conf,
                    )
                if digits_exp:
                    digits, data, mask = digits_exp, data_exp, mask_exp
                    roi, gray = roi_expanded, gray_expanded
                    x, y = x0, y0
                    w, h = x1 - x0, y1 - y0

        if not digits:
            span_left, span_right = cache._LAST_REGION_SPANS.get(name, (x, x + w))
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
                    if top_crop > 0 and gray_retry.shape[0] > top_crop:
                        gray_retry = gray_retry[top_crop:, :]
                    try:
                        digits_retry, data_retry, mask_retry, low_conf = execute_ocr(
                            gray_retry,
                            color=roi_retry,
                            conf_threshold=res_conf_threshold,
                            allow_fallback=False,
                            roi=(cand_x, y, cand_w, h),
                            resource=name,
                        )
                        logger.info(
                            "OCR %s: digits=%s conf=%s low_conf=%s",
                            name,
                            digits_retry,
                            data_retry.get("conf"),
                            low_conf,
                        )
                    except TypeError:
                        digits_retry, data_retry, mask_retry, low_conf = execute_ocr(
                            gray_retry,
                            conf_threshold=res_conf_threshold,
                            allow_fallback=False,
                            resource=name,
                        )
                        logger.info(
                            "OCR %s: digits=%s conf=%s low_conf=%s",
                            name,
                            digits_retry,
                            data_retry.get("conf"),
                            low_conf,
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
            ts_cache = cache_obj.last_resource_ts.get(name)
            use_cache = False
            if name in cache_obj.last_resource_values and ts_cache is not None:
                age = time.time() - ts_cache
                if age < cache._RESOURCE_CACHE_TTL:
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
                results[name] = cache_obj.last_resource_values[name]
                cache_hits.add(name)
            else:
                results[name] = None
                if low_conf:
                    low_confidence.add(name)
                else:
                    no_digits.add(name)
            cache_obj.resource_failure_counts[name] = failure_count + 1
        else:
            value = int(digits)
            if (
                low_conf
                and CFG.get("treat_low_conf_as_failure", True)
                and (name != "wood_stockpile" or name in cache_obj.last_resource_values)
            ):
                fallback_key = f"{name}_low_conf_fallback"
                if CFG.get(fallback_key, False) and name in cache_obj.last_resource_values:
                    results[name] = cache_obj.last_resource_values[name]
                    cache_hits.add(name)
                    logger.warning(
                        "Using cached value for %s due to low-confidence OCR", name
                    )
                else:
                    results[name] = None
                    logger.warning(
                        "Discarding %s=%d due to low-confidence OCR",
                        name,
                        value,
                    )
                low_confidence.add(name)
            else:
                results[name] = value
                if not low_conf:
                    cache_obj.last_resource_values[name] = value
                    cache_obj.last_resource_ts[name] = time.time()
                    cache_obj.resource_failure_counts[name] = 0
                else:
                    low_confidence.add(name)
            if results.get(name) is not None:
                logger.info("Detected %s=%d", name, value)

    filtered_regions = {n: regions[n] for n in resource_icons if n in regions}
    required_for_ocr = [n for n in required_icons if n != "population_limit"]
    handle_ocr_failure(
        frame,
        filtered_regions,
        results,
        required_for_ocr,
        cache_obj,
        debug_images=debug_images,
        low_confidence=low_confidence,
    )

    cur_pop = pop_cap = None
    if "population_limit" in icons_to_read:
        pop_required = "population_limit" in required_set
        pop_conf_threshold = CFG.get(
            "population_limit_ocr_conf_threshold", conf_threshold
        )
        cur_pop, pop_cap = _extract_population(
            frame,
            regions,
            results,
            pop_required,
            conf_threshold=pop_conf_threshold,
        )

    cache._LAST_READ_FROM_CACHE = cache_hits

    # Record failure reasons for later inspection
    global _LAST_LOW_CONFIDENCE, _LAST_NO_DIGITS
    _LAST_LOW_CONFIDENCE = set(low_confidence)
    _LAST_NO_DIGITS = set(no_digits)

    logger.info("Resumo de recursos detectados: %s", results)

    return results, (cur_pop, pop_cap)


def read_resources_from_hud(
    required_icons: Iterable[str] | None = None,
    icons_to_read: Iterable[str] | None = None,
    *,
    force_delay=None,
    max_cache_age: float | None = None,
    conf_threshold: int | None = None,
    cache: ResourceCache = RESOURCE_CACHE,
):
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


def gather_hud_stats(
    force_delay=None,
    required_icons=None,
    optional_icons=None,
    max_cache_age: float | None = None,
    conf_threshold: int | None = None,
    cache: ResourceCache = RESOURCE_CACHE,
):
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


def validate_starting_resources(
    current: dict[str, int],
    expected: dict[str, int] | None,
    *,
    tolerance: int = 10,
    raise_on_error: bool = False,
    frame: np.ndarray | None = None,
    rois: dict[str, tuple[int, int, int, int]] | None = None,
) -> None:
    if not expected:
        return
    errors: list[str] = []
    for name, exp in expected.items():
        actual = current.get(name)
        if actual is None:
            if name in _LAST_LOW_CONFIDENCE:
                msg = f"Low-confidence OCR for '{name}'"
            else:
                msg = f"Missing OCR reading for '{name}'"
            if raise_on_error:
                logger.error(msg)
                errors.append(msg)
            else:
                logger.warning(msg)
            continue

        if abs(actual - exp) > tolerance:
            roi_path = None
            if frame is not None and rois and name in rois:
                x, y, w, h = rois[name]
                debug_dir = ROOT / "debug"
                debug_dir.mkdir(exist_ok=True)
                ts = int(time.time() * 1000)
                roi_img = frame[y : y + h, x : x + w]
                roi_path = debug_dir / f"resource_roi_{name}_{ts}.png"
                cv2.imwrite(str(roi_path), roi_img)

            msg = (
                f"{name} reading {actual} deviates from expected {exp} "
                f"(±{tolerance})"
            )
            if roi_path is not None:
                msg += f"; ROI saved to {roi_path}"
            if raise_on_error:
                logger.error(msg)
                errors.append(msg)
            else:
                logger.warning(msg)

    if errors and raise_on_error:
        raise ValueError("; ".join(errors))



__all__ = [
    "ResourceCache",
    "RESOURCE_CACHE",
    "_LAST_READ_FROM_CACHE",
    "_NARROW_ROIS",
    "_NARROW_ROI_DEFICITS",
    "_RESOURCE_CACHE_TTL",
    "_RESOURCE_DEBUG_COOLDOWN",
    "_LAST_REGION_BOUNDS",
    "_LAST_REGION_SPANS",
    "preprocess_roi",
    "_ocr_digits_better",
    "execute_ocr",
    "handle_ocr_failure",
    "_read_population_from_roi",
    "read_population_from_roi",
    "_read_resources",
    "read_resources_from_hud",
    "gather_hud_stats",
    "validate_starting_resources",
]
