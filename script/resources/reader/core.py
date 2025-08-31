from __future__ import annotations

import time
from typing import Iterable

import cv2
import numpy as np
import pytesseract

from .. import CFG, ROOT, cache, common, logger, screen_utils, RESOURCE_ICON_ORDER
from ..panel import detect_resource_regions
from ..ocr.preprocess import preprocess_roi
from ..ocr.confidence import parse_confidences
from ..ocr.executor import (
    execute_ocr,
    handle_ocr_failure,
    _read_population_from_roi,
    read_population_from_roi,
    _extract_population,
)
from .cache_utils import (
    ResourceCache,
    RESOURCE_CACHE,
    _LAST_READ_FROM_CACHE,
    _NARROW_ROIS,
    _NARROW_ROI_DEFICITS,
    _RESOURCE_CACHE_TTL,
    _RESOURCE_CACHE_MAX_AGE,
    _RESOURCE_DEBUG_COOLDOWN,
    _LAST_REGION_BOUNDS,
    _LAST_REGION_SPANS,
)
from .roi import prepare_roi, expand_roi_after_failure

_LAST_LOW_CONFIDENCE: set[str] = set()
_LAST_NO_DIGITS: set[str] = set()


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
    low_conf_counts = getattr(cache_obj, "resource_low_conf_counts", {})
    cache_obj.resource_low_conf_counts = low_conf_counts
    for name in resource_icons:
        res_conf_threshold = CFG.get(f"{name}_ocr_conf_threshold", conf_threshold)
        if name not in regions:
            if name in required_set:
                results[name] = None
                no_digits.add(name)
            continue
        roi_info = prepare_roi(frame, regions, name, required_set, cache_obj)
        if roi_info is None:
            continue
        x, y, w, h, roi, gray, top_crop, failure_count = roi_info
        if name == "idle_villager":
            data = pytesseract.image_to_data(
                gray,
                config="--psm 7 -c tessedit_char_whitelist=0123456789",
                output_type=pytesseract.Output.DICT,
            )
            texts = [t for t in data.get("text", []) if t.strip()]
            raw_digits = "".join(filter(str.isdigit, "".join(texts))) if texts else None
            confidences = parse_confidences(data)
            digits = None
            low_conf = False
            if raw_digits:
                if confidences and any(c >= res_conf_threshold for c in confidences):
                    digits = raw_digits
                elif confidences and any(c > 0 for c in confidences):
                    digits = None
                else:
                    digits = raw_digits
                    low_conf = True
            mask = gray
            if digits and low_conf:
                low_conf_counts[name] = low_conf_counts.get(name, 0) + 1
            else:
                low_conf_counts[name] = 0
        else:
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
                parse_confidences(data),
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
                        parse_confidences(data_retry),
                        low_conf,
                    )
                    if digits_retry:
                        digits, data, mask = digits_retry, data_retry, mask_retry
        if not digits:
            expansion = expand_roi_after_failure(
                frame,
                name,
                x,
                y,
                w,
                h,
                roi,
                gray,
                top_crop,
                failure_count,
                res_conf_threshold,
            )
            if expansion:
                (
                    digits,
                    data,
                    mask,
                    roi,
                    gray,
                    x,
                    y,
                    w,
                    h,
                    low_conf,
                ) = expansion
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
                        parse_confidences(data_retry),
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
            if name == "idle_villager":
                count = low_conf_counts.get(name, 0)
                if low_conf:
                    threshold = CFG.get("idle_villager_low_conf_streak", 3)
                    if count >= threshold and name in cache_obj.last_resource_values:
                        results[name] = cache_obj.last_resource_values[name]
                        cache_hits.add(name)
                        logger.warning(
                            "Using cached value for %s due to %d consecutive low-confidence OCR results",
                            name,
                            count,
                        )
                    else:
                        results[name] = value
                    low_confidence.add(name)
                else:
                    results[name] = value
                    cache_obj.last_resource_values[name] = value
                    cache_obj.last_resource_ts[name] = time.time()
                    cache_obj.resource_failure_counts[name] = 0
                if results.get(name) is not None:
                    logger.info("Detected %s=%d", name, results[name])
                continue
            treat_low_conf_as_failure = (
                CFG.get("treat_low_conf_as_failure", True)
                and not CFG.get("allow_low_conf_digits", False)
            )
            if (
                low_conf
                and treat_low_conf_as_failure
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
            cache_obj=cache_obj,
        )

    cache._LAST_READ_FROM_CACHE = cache_hits

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
    tolerances: dict[str, int] | None = None,
    raise_on_error: bool = False,
    frame: np.ndarray | None = None,
    rois: dict[str, tuple[int, int, int, int]] | None = None,
) -> None:
    if not expected:
        return
    errors: list[str] = []

    def _save_debug(name: str):
        if frame is None or rois is None or name not in rois:
            return None
        x, y, w, h = rois[name]
        debug_dir = ROOT / "debug"
        debug_dir.mkdir(exist_ok=True)
        ts = int(time.time() * 1000)
        roi_img = frame[y : y + h, x : x + w]
        roi_path = debug_dir / f"resource_roi_{name}_{ts}.png"
        gray = preprocess_roi(roi_img)
        gray_path = debug_dir / f"resource_gray_{name}_{ts}.png"
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_path = debug_dir / f"resource_thresh_{name}_{ts}.png"
        cv2.imwrite(str(roi_path), roi_img)
        cv2.imwrite(str(gray_path), gray)
        cv2.imwrite(str(thresh_path), mask)
        return x, y, w, h, roi_path, gray_path, thresh_path

    for name, exp in expected.items():
        actual = current.get(name)

        if actual is None:
            if name in _LAST_LOW_CONFIDENCE:
                debug = _save_debug(name)
                if debug is not None:
                    x, y, w, h, roi_path, gray_path, thresh_path = debug
                    msg = (
                        f"Low-confidence OCR for '{name}' at ROI (x={x}, y={y}, w={w}, h={h}); "
                        f"ROI saved to {roi_path}; gray saved to {gray_path}; threshold saved to {thresh_path}"
                    )
                else:
                    msg = f"Low-confidence OCR for '{name}'"
            else:
                msg = f"Missing OCR reading for '{name}'"
            if raise_on_error:
                logger.error(msg)
                errors.append(msg)
            else:
                logger.warning(msg)
            continue

        tol = tolerance if tolerances is None else tolerances.get(name, tolerance)
        deviation = abs(actual - exp)
        need_debug = name in _LAST_LOW_CONFIDENCE or deviation > tol
        debug = _save_debug(name) if need_debug else None

        debug_info = ""
        if debug is not None:
            x, y, w, h, roi_path, gray_path, thresh_path = debug
            debug_info = (
                f"; ROI (x={x}, y={y}, w={w}, h={h}) saved to {roi_path}; "
                f"gray saved to {gray_path}; threshold saved to {thresh_path}"
            )

        if deviation > tol:
            msg = (
                f"{name} reading {actual} deviates from expected {exp} "
                f"(±{tol}){debug_info}"
            )
            if raise_on_error:
                logger.error(msg)
                errors.append(msg)
            else:
                logger.warning(msg)
        elif name in _LAST_LOW_CONFIDENCE:
            msg = f"Low-confidence OCR for '{name}'{debug_info}"
            if raise_on_error:
                logger.error(msg)
                errors.append(msg)
            else:
                logger.warning(msg)

    if errors and raise_on_error:
        raise ValueError("; ".join(errors))


__all__ = [
    "_read_resources",
    "read_resources_from_hud",
    "gather_hud_stats",
    "validate_starting_resources",
]
