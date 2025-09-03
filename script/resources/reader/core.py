from __future__ import annotations

import time
from typing import Iterable, Callable

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


class ResourceValidationError(ValueError):
    """Exception raised when resource validation fails.

    Attributes
    ----------
    failing_keys:
        Set of resource names that did not meet the expected values.
    """

    def __init__(self, errors: list[str], failing_keys: set[str]):
        super().__init__("; ".join(errors))
        self.failing_keys = failing_keys


def _ocr_resource(
    name: str,
    roi: np.ndarray,
    gray: np.ndarray,
    res_conf_threshold: int,
    roi_bbox: tuple[int, int, int, int],
    cache_obj: ResourceCache,
) -> tuple[str | None, dict, np.ndarray | None, bool]:
    """Run OCR for a single resource ROI.

    Parameters
    ----------
    name:
        Name of the resource being read.
    roi:
        Color ROI extracted from the HUD.
    gray:
        Grayscale and preprocessed version of ``roi``.
    res_conf_threshold:
        Minimum confidence required for OCR digits.
    roi_bbox:
        Bounding box of the ROI as ``(x, y, w, h)``.
    cache_obj:
        Resource cache used for lookups and statistics.

    Returns
    -------
    tuple[str | None, dict, np.ndarray | None, bool]
        Detected digits, raw Tesseract data, threshold mask and a flag
        indicating low-confidence detection.
    """

    x, y, w, h = roi_bbox
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
        logger.info(
            "OCR %s: digits=%s conf=%s low_conf=%s",
            name,
            digits,
            confidences,
            low_conf,
        )
        return digits, data, mask, low_conf

    digits, data, mask, low_conf = execute_ocr(
        gray,
        color=roi,
        conf_threshold=res_conf_threshold,
        roi=roi_bbox,
        resource=name,
    )
    if data.get("zero_conf") and not CFG.get("allow_zero_confidence_digits"):
        low_conf = True
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
                roi=roi_bbox,
                resource=name,
            )
            if data_retry.get("zero_conf") and not CFG.get("allow_zero_confidence_digits"):
                low_conf = True
            logger.info(
                "Retry OCR %s: digits=%s conf=%s low_conf=%s",
                name,
                digits_retry,
                parse_confidences(data_retry),
                low_conf,
            )
            if digits_retry:
                digits, data, mask = digits_retry, data_retry, mask_retry
    treat_low_conf_as_failure = (
        CFG.get("treat_low_conf_as_failure", True)
        and not CFG.get("allow_low_conf_digits", False)
    )
    if low_conf and treat_low_conf_as_failure:
        digits = None
    return digits, data, mask, low_conf


def _retry_ocr(
    frame: np.ndarray,
    name: str,
    digits: str | None,
    data: dict,
    mask: np.ndarray | None,
    roi: np.ndarray,
    gray: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    top_crop: int,
    failure_count: int,
    res_conf_threshold: int,
    low_conf: bool,
) -> tuple[str | None, dict, np.ndarray | None, np.ndarray, np.ndarray, int, int, int, int, bool]:
    """Retry OCR with ROI expansion and alternative widths.

    Returns updated OCR results and ROI information.
    """

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
            if data.get("zero_conf") and not CFG.get("allow_zero_confidence_digits"):
                low_conf = True
    if not digits:
        span_left, span_right = cache._LAST_REGION_SPANS.get(name, (x, x + w))
        span_width = span_right - span_left
        cand_widths = [min(w, span_width)]
        cand_widths += [min(span_width, cw) for cw in (64, 56, 48)]
        cand_widths = list(dict.fromkeys(cand_widths))
        for idx, cand_w in enumerate(cand_widths):
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
                if data_retry.get("zero_conf") and not CFG.get("allow_zero_confidence_digits"):
                    low_conf = True
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
            if idx == 0 and not digits:
                expanded_w = min(span_width, cand_w + 10)
                if expanded_w > cand_w:
                    cand_x = span_left
                    cand_x = max(span_left, min(cand_x, span_right - expanded_w))
                    roi_retry = frame[y : y + h, cand_x : cand_x + expanded_w]
                    gray_retry = preprocess_roi(roi_retry)
                    if top_crop > 0 and gray_retry.shape[0] > top_crop:
                        gray_retry = gray_retry[top_crop:, :]
                    digits_retry, data_retry, mask_retry, low_conf = execute_ocr(
                        gray_retry,
                        color=roi_retry,
                        conf_threshold=res_conf_threshold,
                        allow_fallback=False,
                        roi=(cand_x, y, expanded_w, h),
                        resource=name,
                    )
                    if data_retry.get("zero_conf") and not CFG.get("allow_zero_confidence_digits"):
                        low_conf = True
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
                        x, w = cand_x, expanded_w
                        break
    return digits, data, mask, roi, gray, x, y, w, h, low_conf


def _handle_cache_and_fallback(
    name: str,
    digits: str | None,
    low_conf: bool,
    data: dict,
    roi: np.ndarray,
    mask: np.ndarray | None,
    failure_count: int,
    *,
    cache_obj: ResourceCache,
    max_cache_age: float | None,
    low_conf_counts: dict[str, int],
) -> tuple[int | None, bool, bool, bool]:
    """Resolve OCR output using cache and fallback strategies.

    Returns the final value, whether a cache hit occurred and flags for
    low-confidence or missing digits.
    """

    cache_hit = False
    low_conf_flag = False
    no_digit_flag = False

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
            cached_val = cache_obj.last_resource_values[name]
            tol = CFG.get("resource_cache_tolerance", 100)
            expected = CFG.get("starting_resources", {}).get(name)
            if expected is not None and abs(cached_val - expected) > tol:
                logger.warning(
                    "Discarding cached value for %s due to mismatch with expected %d",
                    name,
                    expected,
                )
                use_cache = False
        if use_cache:
            value = cache_obj.last_resource_values[name]
            cache_hit = True
        else:
            value = None
            if low_conf:
                low_conf_flag = True
            else:
                no_digit_flag = True
        cache_obj.resource_failure_counts[name] = failure_count + 1
        return value, cache_hit, low_conf_flag, no_digit_flag

    value = int(digits)
    count = low_conf_counts.get(name, 0)
    if low_conf and name != "idle_villager":
        threshold = CFG.get(
            f"{name}_low_conf_streak",
            CFG.get("resource_low_conf_streak", 3),
        )
        if count >= threshold and name in cache_obj.last_resource_values:
            cache_hit = True
            logger.warning(
                "Using cached value for %s due to %d consecutive low-confidence OCR results",
                name,
                count,
            )
            low_conf_flag = True
            return (
                cache_obj.last_resource_values[name],
                cache_hit,
                low_conf_flag,
                False,
            )

    if name == "idle_villager":
        if low_conf:
            threshold = CFG.get("idle_villager_low_conf_streak", 5)
            if count >= threshold and name in cache_obj.last_resource_values:
                cache_hit = True
                logger.warning(
                    "Using cached value for %s due to %d consecutive low-confidence OCR results",
                    name,
                    count,
                )
                return (
                    cache_obj.last_resource_values[name],
                    cache_hit,
                    True,
                    False,
                )
            cache_obj.last_resource_values[name] = value
            cache_obj.last_resource_ts[name] = time.time()
            low_conf_flag = True
        else:
            cache_obj.last_resource_values[name] = value
            cache_obj.last_resource_ts[name] = time.time()
            cache_obj.resource_failure_counts[name] = 0
        return value, cache_hit, low_conf_flag, no_digit_flag

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
            cached_val = cache_obj.last_resource_values[name]
            tol = CFG.get("resource_cache_tolerance", 100)
            expected = CFG.get("starting_resources", {}).get(name)
            compare = value if digits is not None else expected
            if compare is not None and abs(cached_val - compare) > tol:
                logger.warning(
                    "Discarding cached value for %s due to mismatch with %d",
                    name,
                    compare,
                )
                cache_obj.last_resource_values[name] = compare if compare is not None else cached_val
                cache_obj.last_resource_ts[name] = time.time()
                cache_obj.resource_failure_counts[name] = 0
                return compare, False, False, no_digit_flag
            cache_hit = True
            value = cached_val
            logger.warning(
                "Using cached value for %s due to low-confidence OCR", name
            )
        else:
            logger.warning(
                "Discarding %s=%d due to low-confidence OCR",
                name,
                value,
            )
            value = None
            no_digit_flag = False
        low_conf_flag = True
        return value, cache_hit, low_conf_flag, no_digit_flag

    if (
        name == "wood_stockpile"
        and low_conf
        and name not in cache_obj.last_resource_values
    ):
        logger.warning(
            "Discarding %s=%d due to low-confidence OCR", name, value
        )
        value = None
        low_conf_flag = True
        no_digit_flag = False
        cache_obj.resource_failure_counts[name] = failure_count + 1
        return value, cache_hit, low_conf_flag, no_digit_flag

    if not low_conf:
        cache_obj.last_resource_values[name] = value
        cache_obj.last_resource_ts[name] = time.time()
        cache_obj.resource_failure_counts[name] = 0
    else:
        low_conf_flag = True
    return value, cache_hit, low_conf_flag, no_digit_flag


def _process_resource(
    frame: np.ndarray,
    name: str,
    roi_info: tuple[int, int, int, int, np.ndarray, np.ndarray, int, int],
    *,
    cache_obj: ResourceCache,
    res_conf_threshold: int,
    max_cache_age: float | None,
    low_conf_counts: dict[str, int],
) -> tuple[int | None, bool, bool, bool, tuple[np.ndarray, np.ndarray]]:
    """Read and resolve a single resource value from the HUD.

    Parameters
    ----------
    frame:
        Full game frame from which the ROI was extracted.
    name:
        Name of the resource being processed.
    roi_info:
        Tuple returned by :func:`prepare_roi` describing the ROI and
        preprocessing results.
    cache_obj:
        Cache used for lookups and statistics.
    res_conf_threshold:
        Minimum OCR confidence required for this resource.
    max_cache_age:
        Optional maximum age for cached values when falling back.
    low_conf_counts:
        Counter tracking consecutive low-confidence results.

    Returns
    -------
    tuple
        ``(value, cache_hit, low_conf_flag, no_digit_flag, debug_images)``
        where ``debug_images`` contains the grayscale ROI and threshold mask.
    """

    x, y, w, h, roi, gray, top_crop, failure_count = roi_info
    digits, data, mask, low_conf = _ocr_resource(
        name, roi, gray, res_conf_threshold, (x, y, w, h), cache_obj
    )
    digits, data, mask, roi, gray, x, y, w, h, low_conf = _retry_ocr(
        frame,
        name,
        digits,
        data,
        mask,
        roi,
        gray,
        x,
        y,
        w,
        h,
        top_crop,
        failure_count,
        res_conf_threshold,
        low_conf,
    )
    if digits and low_conf:
        low_conf_counts[name] = low_conf_counts.get(name, 0) + 1
    else:
        low_conf_counts[name] = 0
    debug_images = (gray, mask)
    if CFG.get("ocr_debug"):
        debug_dir = ROOT / "debug"
        debug_dir.mkdir(exist_ok=True)
        ts = int(time.time() * 1000)
        cv2.imwrite(str(debug_dir / f"resource_{name}_roi_{ts}.png"), roi)
        if mask is not None:
            cv2.imwrite(str(debug_dir / f"resource_{name}_thresh_{ts}.png"), mask)
    value, cache_hit, low_conf_flag, no_digit_flag = _handle_cache_and_fallback(
        name,
        digits,
        low_conf,
        data,
        roi,
        mask,
        failure_count,
        cache_obj=cache_obj,
        max_cache_age=max_cache_age,
        low_conf_counts=low_conf_counts,
    )
    if value is not None:
        logger.info("Detected %s=%d", name, value)
    return value, cache_hit, low_conf_flag, no_digit_flag, debug_images


def _post_process_population(
    frame: np.ndarray,
    regions: dict[str, tuple[int, int, int, int]],
    icons_to_read: list[str],
    required_set: set[str],
    results: dict[str, int | None],
    *,
    cache_obj: ResourceCache,
    max_cache_age: float | None,
    conf_threshold: int,
    low_conf_counts: dict[str, int],
    low_confidence: set[str],
    cache_hits: set[str],
    prev_idle_val: int | None,
    prev_idle_ts: float | None,
) -> tuple[int | None, int | None]:
    """Handle population readings and idle villager validation.

    Returns the current population and population cap while applying
    fallback logic for inconsistent idle villager counts.
    """

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

    idle_val = results.get("idle_villager")
    if idle_val is not None and cur_pop is not None:
        if idle_val > cur_pop or idle_val < 0:
            logger.warning(
                "Idle villager count %d is invalid for population %d; falling back",
                idle_val,
                cur_pop,
            )
            low_confidence.add("idle_villager")
            low_conf_counts["idle_villager"] = low_conf_counts.get(
                "idle_villager", 0
            ) + 1
            ts_cache = prev_idle_ts
            use_cache = False
            if prev_idle_val is not None and ts_cache is not None:
                age = time.time() - ts_cache
                if age < cache._RESOURCE_CACHE_TTL:
                    use_cache = True
                elif max_cache_age is None or age <= max_cache_age:
                    use_cache = True
            if use_cache:
                results["idle_villager"] = prev_idle_val
                cache_obj.last_resource_values["idle_villager"] = prev_idle_val
                if ts_cache is not None:
                    cache_obj.last_resource_ts["idle_villager"] = ts_cache
                cache_hits.add("idle_villager")
            else:
                results["idle_villager"] = None
                cache_obj.last_resource_values.pop("idle_villager", None)
                cache_obj.last_resource_ts.pop("idle_villager", None)

    return cur_pop, pop_cap


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
    cache_hits: set[str] = set()
    debug_images: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    low_confidence: set[str] = set()
    no_digits: set[str] = set()
    low_conf_counts = getattr(cache_obj, "resource_low_conf_counts", {})
    cache_obj.resource_low_conf_counts = low_conf_counts
    prev_idle_val = cache_obj.last_resource_values.get("idle_villager")
    prev_idle_ts = cache_obj.last_resource_ts.get("idle_villager")

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
        (
            value,
            cache_hit,
            low_conf_flag,
            no_digit_flag,
            dbg,
        ) = _process_resource(
            frame,
            name,
            roi_info,
            cache_obj=cache_obj,
            res_conf_threshold=res_conf_threshold,
            max_cache_age=max_cache_age,
            low_conf_counts=low_conf_counts,
        )
        debug_images[name] = dbg
        results[name] = value
        if cache_hit:
            cache_hits.add(name)
        if low_conf_flag:
            low_confidence.add(name)
        if no_digit_flag:
            no_digits.add(name)

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

    cur_pop, pop_cap = _post_process_population(
        frame,
        regions,
        icons_to_read,
        required_set,
        results,
        cache_obj=cache_obj,
        max_cache_age=max_cache_age,
        conf_threshold=conf_threshold,
        low_conf_counts=low_conf_counts,
        low_confidence=low_confidence,
        cache_hits=cache_hits,
        prev_idle_val=prev_idle_val,
        prev_idle_ts=prev_idle_ts,
    )

    cache._LAST_READ_FROM_CACHE = cache_hits
    cache_obj.last_low_confidence = set(low_confidence)
    cache_obj.last_no_digits = set(no_digits)
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

    def _run_read():
        return _read_resources(
            frame,
            required_icons,
            all_icons,
            cache,
            max_cache_age,
            conf_threshold,
        )

    results, pop = _run_read()

    cur_pop, pop_cap = pop
    idle_val = results.get("idle_villager")

    needs_retry = False
    if pop_cap is not None:
        if cur_pop is not None and cur_pop > pop_cap:
            needs_retry = True
        if idle_val is not None and idle_val > pop_cap:
            needs_retry = True
    if needs_retry:
        logger.warning(
            "Population readings exceed cap (idle=%s cur=%s cap=%s); retrying OCR",
            idle_val,
            cur_pop,
            pop_cap,
        )
        results, pop = _run_read()

    return results, pop


def validate_starting_resources(
    current: dict[str, int],
    expected: dict[str, int] | None,
    *,
    tolerance: int = 10,
    tolerances: dict[str, int] | None = None,
    raise_on_error: bool = False,
    frame: np.ndarray | None = None,
    rois: dict[str, tuple[int, int, int, int]] | None = None,
    cache: ResourceCache = RESOURCE_CACHE,
) -> set[str]:
    if not expected:
        return set()
    errors: list[str] = []
    failing: set[str] = set()

    def _save_debug(name: str):
        if frame is None or rois is None or name not in rois:
            return None
        x, y, w, h = rois[name]
        debug_dir = ROOT / "debug"
        debug_dir.mkdir(exist_ok=True)
        ts = int(time.time() * 1000)
        text = f"{name} {ts}"
        roi_img = frame[y : y + h, x : x + w]
        roi_dbg = roi_img.copy()
        cv2.putText(
            roi_dbg,
            text,
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        roi_path = debug_dir / f"resource_roi_{name}_{ts}.png"
        gray = preprocess_roi(roi_img)
        gray_dbg = gray.copy()
        cv2.putText(
            gray_dbg,
            text,
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            255,
            1,
            cv2.LINE_AA,
        )
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask_dbg = mask.copy()
        cv2.putText(
            mask_dbg,
            text,
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            255,
            1,
            cv2.LINE_AA,
        )
        gray_path = debug_dir / f"resource_gray_{name}_{ts}.png"
        thresh_path = debug_dir / f"resource_thresh_{name}_{ts}.png"
        cv2.imwrite(str(roi_path), roi_dbg)
        cv2.imwrite(str(gray_path), gray_dbg)
        cv2.imwrite(str(thresh_path), mask_dbg)
        return x, y, w, h, roi_path, gray_path, thresh_path

    for name, exp in expected.items():
        actual = current.get(name)

        if actual is None:
            if name in cache.last_low_confidence:
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
                failing.add(name)
            else:
                logger.warning(msg)
                failing.add(name)
            continue

        tol = tolerance if tolerances is None else tolerances.get(name, tolerance)
        deviation = abs(actual - exp)
        debug = None
        if deviation > tol:
            debug = _save_debug(name)
        elif name in cache.last_low_confidence:
            debug = _save_debug(name)

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
                failing.add(name)
            else:
                logger.warning(msg)
                failing.add(name)
        elif name in cache.last_low_confidence:
            msg = f"Low-confidence OCR for '{name}'{debug_info}"
            if raise_on_error:
                logger.error(msg)
                errors.append(msg)
                failing.add(name)
            else:
                logger.warning(msg)
                failing.add(name)

    if errors and raise_on_error:
        raise ResourceValidationError(errors, failing)

    return failing


def validate_population(
    res_vals: dict[str, int],
    cur_pop: int | None,
    pop_cap: int | None,
    *,
    expected_cur: int,
    expected_cap: int,
    retry_fn: Callable[[], tuple[dict[str, int], tuple[int | None, int | None]]] | None = None,
    max_attempts: int = 2,
) -> tuple[dict[str, int], tuple[int | None, int | None]] | None:
    """Validate population readings with optional OCR retries.

    Parameters
    ----------
    cur_pop:
        Current population read from the HUD.
    pop_cap:
        Population cap read from the HUD.
    expected_cur:
        Expected starting population for the scenario.
    expected_cap:
        Expected population cap for the scenario.
    retry_fn:
        Optional callable returning ``(cur_pop, pop_cap)`` for an OCR retry.
    max_attempts:
        Total number of attempts allowed including the initial read.

    Returns
    -------
    tuple[dict[str, int], tuple[int | None, int | None]] | None
        ``(res_vals, (cur_pop, pop_cap))`` if the readings match
        expectations or ``None`` if validation fails after exhausting
        retries.
    """

    attempt = 1
    while True:
        if cur_pop == expected_cur and pop_cap == expected_cap:
            return res_vals, (cur_pop, pop_cap)
        if (
            cur_pop == pop_cap
            and pop_cap is not None
            and pop_cap != expected_cap
            and retry_fn is not None
            and attempt < max_attempts
        ):
            logger.warning(
                "Population cap equals current population (%s) but expected %s; retrying OCR",
                pop_cap,
                expected_cap,
            )
            res_vals, (cur_pop, pop_cap) = retry_fn()
            attempt += 1
            continue
        logger.error(
            "HUD population (%s/%s) does not match expected %s/%s; aborting scenario.",
            cur_pop,
            pop_cap,
            expected_cur,
            expected_cap,
        )
        return None


__all__ = [
    "_read_resources",
    "read_resources_from_hud",
    "gather_hud_stats",
    "validate_starting_resources",
    "validate_population",
    "ResourceValidationError",
]
