from __future__ import annotations

"""OCR execution and fallback helpers."""

import logging
import re
import time

import cv2
import numpy as np
import pytesseract

from .. import CFG, ROOT, screen_utils, common, cache
from .preprocess import preprocess_roi
from . import masks
from .confidence import parse_confidences, _sanitize_digits

logger = logging.getLogger(__name__)


def execute_ocr(
    gray,
    color=None,
    conf_threshold=None,
    allow_fallback=True,
    roi=None,
    resource=None,
    whitelist=None,
):
    min_conf = CFG.get("ocr_conf_min", 0)
    decay = CFG.get("ocr_conf_decay", 1.0)
    if conf_threshold is None:
        conf_threshold = CFG.get("ocr_conf_threshold", 60)
    allow_zero = CFG.get("allow_zero_confidence_digits")
    metric = 0

    if whitelist is None:
        whitelist = "0123456789"

    try:
        if whitelist == "0123456789":
            digits, data, mask = masks._ocr_digits_better(gray, color, resource=resource)
        else:
            digits, data, mask = masks._ocr_digits_better(
                gray, color, resource=resource, whitelist=whitelist
            )
    except TypeError:
        if whitelist == "0123456789":
            digits, data, mask = masks._ocr_digits_better(gray)
        else:
            digits, data, mask = masks._ocr_digits_better(gray, whitelist=whitelist)
    low_conf = False
    zero_conf = False
    best_digits = digits
    best_data = data
    best_mask = mask
    attempts = 0
    max_attempts = CFG.get("ocr_conf_max_attempts", 10)
    while digits:
        confs = parse_confidences(data)
        if confs is None:
            if data.get("conf"):
                zero_conf = True
                if not allow_zero:
                    low_conf = True
            break
        texts = data.get("text", [])
        digit_confs = [c for t, c in zip(texts, confs) if any(ch.isdigit() for ch in t)]
        metric = np.median(digit_confs) if digit_confs else 0
        if digit_confs and metric >= conf_threshold:
            low_conf = False
            break
        low_conf = True
        if metric == 0:
            break
        if conf_threshold <= min_conf:
            digits, data, mask = best_digits, best_data, best_mask
            break
        old_threshold = conf_threshold
        conf_threshold = max(min_conf, conf_threshold * decay)
        logger.debug(
            "Lowering OCR confidence threshold from %d to %d",
            old_threshold,
            conf_threshold,
        )
        attempts += 1
        if conf_threshold == old_threshold or attempts > max_attempts or conf_threshold <= metric:
            digits, data, mask = best_digits, best_data, best_mask
            break
        try:
            if whitelist == "0123456789":
                digits, data, mask = masks._ocr_digits_better(
                    gray, color, resource=resource
                )
            else:
                digits, data, mask = masks._ocr_digits_better(
                    gray, color, resource=resource, whitelist=whitelist
                )
        except TypeError:
            if whitelist == "0123456789":
                digits, data, mask = masks._ocr_digits_better(gray)
            else:
                digits, data, mask = masks._ocr_digits_better(
                    gray, whitelist=whitelist
                )
        if digits:
            best_digits, best_data, best_mask = digits, data, mask
        if attempts == max_attempts:
            logger.debug(
                "Reached OCR confidence iteration cap (%d)", max_attempts
            )

    # Re-evaluate confidence using the chosen metric after any decay loop
    if digits:
        confs = parse_confidences(data)
        if confs is None:
            if data.get("conf"):
                zero_conf = True
                if not allow_zero:
                    low_conf = True
        else:
            texts = data.get("text", [])
            digit_confs = [c for t, c in zip(texts, confs) if any(ch.isdigit() for ch in t)]
            metric = np.median(digit_confs) if digit_confs else 0
            if digit_confs and metric >= conf_threshold:
                low_conf = False
            else:
                low_conf = True

    if not digits and allow_fallback:
        text = pytesseract.image_to_string(
            gray,
            config=f"--psm 6 --oem 3 -c tessedit_char_whitelist={whitelist}",
        )
        fallback = "".join(filter(str.isdigit, text))
        if fallback:
            digits = fallback
            data = {"text": [text.strip()], "conf": []}
            mask = None
            low_conf = True
    if not digits and best_digits:
        best_digits = _sanitize_digits(best_digits)
        if zero_conf:
            best_data["zero_conf"] = True
        logger.debug(
            "Final OCR confidence threshold %d, median confidence %.2f",
            conf_threshold,
            metric,
        )
        return best_digits, best_data, best_mask, True
    if not digits:
        debug_dir = ROOT / "debug"
        debug_dir.mkdir(exist_ok=True)
        ts = int(time.time() * 1000)
        mask_path = None
        if mask is not None:
            mask_path = debug_dir / f"ocr_fail_mask_{ts}.png"
            cv2.imwrite(str(mask_path), mask)
        roi_path = debug_dir / f"ocr_fail_roi_{ts}.png"
        cv2.imwrite(str(roi_path), gray)
        text_path = debug_dir / f"ocr_fail_text_{ts}.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write("\n".join(data.get("text", [])))
        if mask_path is not None:
            if roi is not None:
                x, y, w, h = roi
                logger.warning(
                    "OCR returned no digits at ROI (%d, %d, %d, %d); mask saved to %s; ROI saved to %s; text output saved to %s",
                    x,
                    y,
                    w,
                    h,
                    mask_path,
                    roi_path,
                    text_path,
                )
            else:
                logger.warning(
                    "OCR returned no digits; mask saved to %s; ROI saved to %s; text output saved to %s",
                    mask_path,
                    roi_path,
                    text_path,
                )
        else:
            if roi is not None:
                x, y, w, h = roi
                logger.warning(
                    "OCR returned no digits at ROI (%d, %d, %d, %d); ROI saved to %s; text output saved to %s",
                    x,
                    y,
                    w,
                    h,
                    roi_path,
                    text_path,
                )
            else:
                logger.warning(
                    "OCR returned no digits; ROI saved to %s; text output saved to %s",
                    roi_path,
                    text_path,
                )
        digits = _sanitize_digits(digits)
        if zero_conf:
            data["zero_conf"] = True
        logger.debug(
            "Final OCR confidence threshold %d, median confidence %.2f",
            conf_threshold,
            metric,
        )
        return digits, data, mask, True
    if digits:
        digits = _sanitize_digits(digits)
        if zero_conf:
            data["zero_conf"] = True
    logger.debug(
        "Final OCR confidence threshold %d, median confidence %.2f",
        conf_threshold,
        metric,
    )
    return digits, data, mask, low_conf


def handle_ocr_failure(
    frame,
    regions,
    results,
    required_icons,
    cache_obj=None,
    retry_limit=None,
    debug_images=None,
    low_confidence=None,
):
    if cache_obj is None:
        cache_obj = cache.RESOURCE_CACHE
    if retry_limit is None:
        retry_limit = CFG.get("ocr_retry_limit", 3)

    if low_confidence is None:
        low_confidence = set()

    required_set = set(required_icons)
    debug_success = CFG.get("ocr_debug_success")

    for name in list(low_confidence):
        count = cache_obj.resource_failure_counts.get(name, 0) + 1
        cache_obj.resource_failure_counts[name] = count
        limit = retry_limit
        if name == "idle_villager":
            limit = CFG.get("idle_villager_low_conf_streak", 5)
        if count >= limit:
            fallback = cache_obj.last_resource_values.get(name)
            if fallback is None and name == "idle_villager":
                logger.warning(
                    "No cached value for %s after %d low-confidence OCR results",
                    name,
                    count,
                )
                continue
            if fallback is None:
                if name in required_set:
                    raise common.ResourceReadError(
                        f"No cached value for {name} after {count} low-confidence OCR results"
                    )
                logger.warning(
                    "No cached value for %s after %d low-confidence OCR results",
                    name,
                    count,
                )
                low_confidence.remove(name)
                continue
            results[name] = fallback
            logger.warning(
                "Using fallback value %s=%d after %d low-confidence OCR results",
                name,
                fallback,
                count,
            )
            low_confidence.remove(name)

    failed = [name for name, v in results.items() if v is None]
    if not failed and not debug_success:
        return

    for name in list(failed):
        count = cache_obj.resource_failure_counts.get(name, 0)
        if count >= retry_limit:
            fallback = cache_obj.last_resource_values.get(name)
            if fallback is None and name == "idle_villager":
                logger.warning(
                    "No cached value for %s after %d OCR failures",
                    name,
                    count,
                )
                continue
            if fallback is None:
                if name in required_set:
                    raise common.ResourceReadError(
                        f"No cached value for {name} after {count} OCR failures"
                    )
                logger.warning(
                    "No cached value for %s after %d OCR failures",
                    name,
                    count,
                )
                continue
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

    required_failed = [f for f in failed if f in required_set]
    optional_failed = [f for f in failed if f not in required_set]

    def _annotate(names):
        return [
            f"{n} (narrow ROI span)" if n in cache._NARROW_ROIS else n for n in names
        ]

    annotated_required = _annotate(required_failed)
    annotated_optional = _annotate(optional_failed)
    failure_set = set(failed)
    now = time.time()
    save_debug = True
    if (
        failure_set == cache_obj.last_debug_failure_set
        and cache_obj.last_debug_failure_ts is not None
        and now - cache_obj.last_debug_failure_ts < cache._RESOURCE_DEBUG_COOLDOWN
    ):
        save_debug = False

    panel_path = None
    roi_paths = []
    roi_logs = []

    if save_debug:
        cache_obj.last_debug_failure_set = failure_set
        cache_obj.last_debug_failure_ts = now

        h_full, w_full = frame.shape[:2]
        debug_dir = ROOT / "debug"
        debug_dir.mkdir(exist_ok=True)
        ts = int(time.time() * 1000)

        if failed:
            annotated = frame.copy()
            for name, (x, y, w, h) in regions.items():
                if name in failed:
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 1)
            panel_path = debug_dir / f"resource_panel_fail_{ts}.png"
            cv2.imwrite(str(panel_path), annotated)

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

            cache_obj.last_debug_image_ts[name] = now
            roi_paths.append(str(roi_path))
            roi_logs.append(
                f"{name}: (x={x}, y={y}, w={w}, h={h}) -> {roi_path}"
            )

    if required_failed:
        if save_debug:
            logger.error(
                "Resource panel OCR failed for %s; panel saved to %s; rois: %s",
                ", ".join(annotated_required),
                panel_path,
                ", ".join(roi_logs),
            )
            paths_str = ", ".join([str(panel_path)] + roi_paths)
        else:
            logger.error(
                "Resource panel OCR failed for %s; debug images throttled",
                ", ".join(annotated_required),
            )
            paths_str = "throttled"
    if optional_failed:
        if save_debug:
            logger.warning(
                "Resource panel OCR failed for optional %s; panel saved to %s; rois: %s",
                ", ".join(annotated_optional),
                panel_path,
                ", ".join(roi_logs),
            )
        else:
            logger.warning(
                "Resource panel OCR failed for optional %s; debug images throttled",
                ", ".join(annotated_optional),
            )


def _read_population_from_roi(roi, conf_threshold=None, roi_bbox=None, failure_count=0):
    """Return the population (current, cap) from a HUD region.

    The OCR normally expects a string in the form ``"cur/cap"``. If no slash
    is detected and the OCR output is exactly two digits (e.g. ``"34"``), the
    digits are interpreted as single-digit values for the current and maximum
    population respectively.

    When ``CFG['allow_low_conf_population']`` is true, low-confidence OCR
    results are returned after logging a warning instead of raising
    ``PopulationReadError`` immediately.

    Raises:
        common.PopulationReadError: If the population cannot be parsed.
    """

    if conf_threshold is None:
        conf_threshold = CFG.get(
            "population_limit_ocr_conf_threshold",
            CFG.get("ocr_conf_threshold", 60),
        )

    if roi.size == 0:
        msg = "Population ROI has zero size"
        if roi_bbox is not None:
            x, y, w, h = roi_bbox
            msg = f"Population ROI has zero size at ({x}, {y}, {w}, {h})"
        logger.error(msg)
        raise common.PopulationReadError(msg)

    gray = preprocess_roi(roi)
    digits, data, mask, low_conf = execute_ocr(
        gray,
        color=roi,
        conf_threshold=conf_threshold,
        allow_fallback=False,
        whitelist="0123456789/",
        resource="population_limit",
    )
    raw_text = "".join(filter(None, data.get("text", [])))
    parts = [p for p in raw_text.split("/") if p]
    confidences = parse_confidences(data)
    zero_conf = data.get("zero_conf")
    allow_low_conf = CFG.get("allow_low_conf_population", False)
    retry_limit = CFG.get("ocr_retry_limit", 3)
    low_conf_fallback = CFG.get("population_limit_low_conf_fallback", False)
    pattern_ok = bool(re.fullmatch(r"\d+/\d+", raw_text))
    if len(parts) >= 2:
        cur = int("".join(filter(str.isdigit, parts[0])) or 0)
        cap = int("".join(filter(str.isdigit, parts[1])) or 0)
        if not low_conf:
            return cur, cap
        if zero_conf and pattern_ok and (allow_low_conf or low_conf_fallback):
            logger.warning(
                "Returning zero-confidence population %d/%d; conf=%s",
                cur,
                cap,
                confidences,
            )
            return cur, cap
        if allow_low_conf or (low_conf_fallback and failure_count >= retry_limit - 1):
            logger.warning(
                "Returning low-confidence population %d/%d%s; conf=%s",
                cur,
                cap,
                " after %d attempts" % (failure_count + 1)
                if low_conf_fallback and not allow_low_conf
                else "",
                confidences,
            )
            return cur, cap
        err_msg = (
            f"Low-confidence population OCR: text='{raw_text}', confs={confidences}"
        )
        err = common.PopulationReadError(err_msg)
        err.low_conf = True
        err.low_conf_digits = (cur, cap)
        raise err

    if "/" not in raw_text and raw_text.isdigit():
        if len(raw_text) == 2:
            cur = int(raw_text[0])
            cap = int(raw_text[1])
        elif len(raw_text) == 3:
            cur = cap = None
            # Secondary OCR pass: invert the ROI to try to recover a slash
            inv_gray = cv2.bitwise_not(gray)
            _, alt_data, _, _ = execute_ocr(
                inv_gray,
                color=roi,
                conf_threshold=conf_threshold,
                allow_fallback=False,
                whitelist="0123456789/",
                resource="population_limit",
            )
            alt_text = "".join(filter(None, alt_data.get("text", [])))
            alt_parts = [p for p in alt_text.split("/") if p]
            if len(alt_parts) >= 2:
                cur = int("".join(filter(str.isdigit, alt_parts[0])) or 0)
                cap = int("".join(filter(str.isdigit, alt_parts[1])) or 0)
            else:
                # Treat the middle digit as a potential separator
                candidates = [
                    (int(raw_text[0]), int(raw_text[1:])),
                    (int(raw_text[:2]), int(raw_text[2])),
                ]
                for c, p in candidates:
                    if c <= p:
                        cur, cap = c, p
                        break
        elif len(raw_text) == 4:
            half = len(raw_text) // 2
            cur = int(raw_text[:half])
            cap = int(raw_text[half:])
        else:
            cur = cap = None
        if cur is not None:
            if not low_conf:
                return cur, cap
            if allow_low_conf or (low_conf_fallback and failure_count >= retry_limit - 1):
                logger.warning(
                    "Returning low-confidence population %d/%d from ambiguous OCR '%s'%s; conf=%s",
                    cur,
                    cap,
                    raw_text,
                    " after %d attempts" % (failure_count + 1)
                    if low_conf_fallback and not allow_low_conf
                    else "",
                    confidences,
                )
                return cur, cap
            err_msg = (
                f"Low-confidence population OCR: text='{raw_text}', confs={confidences}"
            )
            err = common.PopulationReadError(err_msg)
            err.low_conf = True
            err.low_conf_digits = (cur, cap)
            raise err

    debug_dir = ROOT / "debug"
    debug_dir.mkdir(exist_ok=True)
    ts = int(time.time() * 1000)
    roi_path = debug_dir / f"population_roi_{ts}.png"
    cv2.imwrite(str(roi_path), roi)
    mask_path = None
    if mask is not None:
        mask_path = debug_dir / f"population_mask_{ts}.png"
        cv2.imwrite(str(mask_path), mask)

    if roi_bbox is not None:
        x, y, w, h = roi_bbox
    else:
        h, w = roi.shape[:2]
        x = y = 0

    base_warning = (
        "OCR failed for population at ROI (%d, %d, %d, %d); attempt=%d; "
        "conf_threshold=%s; text='%s', conf=%s; ROI saved to %s"
    )
    if mask_path is not None:
        logger.warning(
            base_warning + "; mask saved to %s",
            x,
            y,
            w,
            h,
            failure_count,
            conf_threshold,
            raw_text,
            confidences,
            roi_path,
            mask_path,
        )
        msg = (
            f"Failed to read population from HUD at ROI ({x}, {y}, {w}, {h}): "
            f"text='{raw_text}', confs={confidences}; conf_threshold={conf_threshold}; "
            f"attempt={failure_count}; ROI saved to {roi_path}; mask saved to {mask_path}"
        )
    else:
        logger.warning(
            base_warning,
            x,
            y,
            w,
            h,
            failure_count,
            conf_threshold,
            raw_text,
            confidences,
            roi_path,
        )
        msg = (
            f"Failed to read population from HUD at ROI ({x}, {y}, {w}, {h}): "
            f"text='{raw_text}', confs={confidences}; conf_threshold={conf_threshold}; "
            f"attempt={failure_count}; ROI saved to {roi_path}"
        )
    err = common.PopulationReadError(msg)
    if low_conf:
        err.low_conf = True
        err.low_conf_digits = None
    raise err


def read_population_from_roi(
    roi_bbox,
    retries: int = 1,
    conf_threshold: int | None = None,
    save_failed_roi: bool = False,
    cache_obj: cache.ResourceCache = cache.RESOURCE_CACHE,
):
    from ..reader.roi import expand_population_roi_after_failure

    x = roi_bbox["left"]
    y = roi_bbox["top"]
    w = roi_bbox["width"]
    h = roi_bbox["height"]
    attempt_errors: list[common.PopulationReadError] = []
    last_exc = None
    for attempt in range(retries):
        roi = screen_utils._grab_frame({"left": x, "top": y, "width": w, "height": h})
        try:
            cur_pop, pop_cap = _read_population_from_roi(
                roi,
                conf_threshold=conf_threshold,
                roi_bbox=(x, y, w, h),
                failure_count=attempt,
            )
            cache_obj.resource_failure_counts["population_limit"] = 0
            return cur_pop, pop_cap
        except common.PopulationReadError as exc:
            exc.attempt = attempt + 1
            attempt_errors.append(exc)
            last_exc = exc
            failure_count = cache_obj.resource_failure_counts.get(
                "population_limit", 0
            )
            frame_full = screen_utils._grab_frame()
            expansion = expand_population_roi_after_failure(
                frame_full,
                x,
                y,
                w,
                h,
                roi,
                failure_count,
                conf_threshold,
                max_right=x + w,
            )
            if expansion:
                cur_pop, pop_cap, _, x, y, w, h = expansion
                cache_obj.resource_failure_counts["population_limit"] = 0
                return cur_pop, pop_cap
            cache_obj.resource_failure_counts["population_limit"] = (
                failure_count + 1
            )
            if attempt < retries - 1:
                logger.debug("OCR attempt %s failed: %s", attempt + 1, exc)
                time.sleep(0.1)

    if last_exc is not None:
        final_attempt = getattr(last_exc, "attempt", retries)
        logger.debug(
            "Final OCR attempt %s failed: %s", final_attempt, last_exc
        )
        msg = f"OCR failed after {final_attempt} attempts: {last_exc}"
        err = common.PopulationReadError(msg)
        err.attempt = final_attempt
        err.attempt_errors = attempt_errors
        if hasattr(last_exc, "low_conf"):
            err.low_conf = last_exc.low_conf
        if hasattr(last_exc, "low_conf_digits"):
            err.low_conf_digits = last_exc.low_conf_digits
        raise err

    raise RuntimeError("read_population_from_roi failed without exception")


def _extract_population(
    frame,
    regions,
    results,
    pop_required,
    conf_threshold=None,
    cache_obj: cache.ResourceCache = cache.RESOURCE_CACHE,
):
    from ..reader.roi import expand_population_roi_after_failure

    cur_pop = pop_cap = None
    if "population_limit" in regions:
        x, y, w, h = regions["population_limit"]
        roi = frame[y : y + h, x : x + w]
        failure_count = cache_obj.resource_failure_counts.get("population_limit", 0)
        low_conf_counts = getattr(cache_obj, "resource_low_conf_counts", {})
        cache_obj.resource_low_conf_counts = low_conf_counts
        low_conf_count = low_conf_counts.get("population_limit", 0)
        try:
            cur_pop, pop_cap = _read_population_from_roi(
                roi,
                conf_threshold=conf_threshold,
                roi_bbox=(x, y, w, h),
                failure_count=low_conf_count,
            )
            if results is not None:
                results["population_limit"] = cur_pop
            cache_obj.resource_failure_counts["population_limit"] = 0
            low_conf_counts["population_limit"] = 0
            cache_obj.last_resource_values["population_limit"] = (cur_pop, pop_cap)
        except common.PopulationReadError as exc:
            low_conf = getattr(exc, "low_conf", False)
            low_conf_digits = getattr(exc, "low_conf_digits", None)
            max_right = (
                regions["idle_villager"][0] - CFG.get("population_idle_padding", 6)
                if "idle_villager" in regions
                else None
            )
            expansion = expand_population_roi_after_failure(
                frame,
                x,
                y,
                w,
                h,
                roi,
                failure_count,
                conf_threshold,
                max_right=max_right,
            )
            if expansion:
                cur_pop, pop_cap, _, x, y, w, h = expansion
                if results is not None:
                    results["population_limit"] = cur_pop
                cache_obj.resource_failure_counts["population_limit"] = 0
                low_conf_counts["population_limit"] = 0
                cache_obj.last_resource_values["population_limit"] = (cur_pop, pop_cap)
            else:
                if results is not None:
                    results["population_limit"] = None
                cache_obj.resource_failure_counts["population_limit"] = (
                    failure_count + 1
                )
                if low_conf:
                    low_conf_counts["population_limit"] = low_conf_count + 1
                else:
                    low_conf_counts["population_limit"] = 0
                if (
                    low_conf
                    and CFG.get("population_limit_low_conf_fallback", False)
                    and low_conf_counts["population_limit"]
                    >= CFG.get("ocr_retry_limit", 3)
                ):
                    if low_conf_digits is not None:
                        cur_pop, pop_cap = low_conf_digits
                        source = "low-confidence"
                    else:
                        cur_pop, pop_cap = cache_obj.last_resource_values.get(
                            "population_limit", (None, None)
                        )
                        source = "cached"
                    if cur_pop is not None:
                        if results is not None:
                            results["population_limit"] = cur_pop
                        logger.warning(
                            "Using %s population %d/%d after %d low-confidence attempts",
                            source,
                            cur_pop,
                            pop_cap,
                            low_conf_counts["population_limit"],
                        )
                        cache_obj.last_resource_values["population_limit"] = (
                            cur_pop,
                            pop_cap,
                        )
                        cache_obj.resource_failure_counts["population_limit"] = 0
                        low_conf_counts["population_limit"] = 0
                        return cur_pop, pop_cap
                if pop_required:
                    raise
    else:
        if results is not None:
            results["population_limit"] = None
        if pop_required:
            raise common.ResourceReadError("population_limit region not detected")
    return cur_pop, pop_cap


__all__ = [
    "execute_ocr",
    "handle_ocr_failure",
    "_read_population_from_roi",
    "read_population_from_roi",
    "_extract_population",
]
