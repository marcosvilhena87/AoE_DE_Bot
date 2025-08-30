"""OCR helpers for reading HUD resources."""
import logging
import time

import cv2
import numpy as np
import pytesseract

from . import CFG, ROOT
from . import screen_utils, common
from . import cache

logger = logging.getLogger(__name__)


def parse_confidences(data):
    """Convert OCR confidence values to floats, ignoring negatives."""

    confs = []
    for c in data.get("conf", []):
        try:
            val = float(c)
        except (ValueError, TypeError):
            continue
        if val >= 0:
            confs.append(val)
    return confs


def preprocess_roi(roi):
    """Convert ROI to a blurred grayscale image."""

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    kernel = CFG.get("ocr_blur_kernel", 1)
    if kernel and kernel > 1 and gray.shape[0] >= 30:
        gray = cv2.medianBlur(gray, kernel)
    return gray


def _ocr_digits_better(gray):
    variance = float(np.var(gray))
    if variance < CFG.get("ocr_zero_variance", 15.0):
        return "0", {"zero_variance": True}, None

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

    contrast = CFG.get("ocr_contrast_stretch")
    if contrast:
        if isinstance(contrast, dict):
            alpha = contrast.get("alpha", 0)
            beta = contrast.get("beta", 255)
            norm_type = getattr(cv2, contrast.get("norm_type", "NORM_MINMAX"), cv2.NORM_MINMAX)
            gray = cv2.normalize(gray, None, alpha=alpha, beta=beta, norm_type=norm_type)
        else:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

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
        yellow_mask = cv2.inRange(hsv, np.array([20, 100, 180]), np.array([40, 255, 255]))
        gray_mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 50, 220]))
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)
        color_mask = cv2.bitwise_or(color_mask, gray_mask)
        digits, data, mask = _run_masks([color_mask, cv2.bitwise_not(color_mask)], 4)
        if not digits:
            closed = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            digits, data, mask = _run_masks([closed, cv2.bitwise_not(closed)], 6)

    if not digits:
        variance = float(np.var(orig))
        if variance < CFG.get("ocr_zero_variance", 15.0):
            return "0", {"zero_variance": True}, mask

    return digits, data, mask


def execute_ocr(
    gray,
    conf_threshold=None,
    allow_fallback=True,
    roi=None,
    resource=None,
):
    min_conf = CFG.get("ocr_conf_min", 0)
    decay = CFG.get("ocr_conf_decay", 1.0)
    if conf_threshold is None:
        conf_threshold = CFG.get("ocr_conf_threshold", 60)

    digits, data, mask = _ocr_digits_better(gray)
    low_conf = False
    best_digits = digits
    best_data = data
    best_mask = mask
    attempts = 0
    max_attempts = CFG.get("ocr_conf_max_attempts", 10)
    while digits and data.get("conf") and attempts < max_attempts:
        confs = parse_confidences(data)
        if confs and min(confs) >= conf_threshold:
            low_conf = False
            break
        low_conf = True
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
    else:
        if attempts >= max_attempts:
            logger.debug("Reached OCR confidence iteration cap (%d)", max_attempts)

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
            return digits, data, mask, True
    if not digits and best_digits:
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
        return digits, data, mask, True
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

    debug_success = CFG.get("ocr_debug_success")

    for name in list(low_confidence):
        count = cache_obj.resource_failure_counts.get(name, 0) + 1
        cache_obj.resource_failure_counts[name] = count
        if count >= retry_limit:
            fallback = cache_obj.last_resource_values.get(name, 0)
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
            fallback = cache_obj.last_resource_values.get(name, 0)
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


def _read_population_from_roi(roi, conf_threshold=None, save_debug=True):
    if conf_threshold is None:
        conf_threshold = CFG.get(
            "population_limit_ocr_conf_threshold",
            CFG.get("ocr_conf_threshold", 60),
        )

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
    confidences = [c for c in map(int, data.get("conf", [])) if c >= 0]
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


def read_population_from_roi(
    roi_bbox,
    retries: int = 1,
    conf_threshold: int | None = None,
    save_failed_roi: bool = False,
):
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


def _extract_population(frame, regions, results, pop_required, conf_threshold=None):
    cur_pop = pop_cap = None
    if "population_limit" in regions:
        x, y, w, h = regions["population_limit"]
        roi = frame[y : y + h, x : x + w]
        try:
            cur_pop, pop_cap = _read_population_from_roi(
                roi, conf_threshold=conf_threshold
            )
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


__all__ = [
    "preprocess_roi",
    "_ocr_digits_better",
    "execute_ocr",
    "handle_ocr_failure",
    "_read_population_from_roi",
    "read_population_from_roi",
    "_extract_population",
]
