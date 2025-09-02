from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .. import CFG, logger, common
from ..ocr.preprocess import preprocess_roi
from ..ocr.executor import execute_ocr, _read_population_from_roi
from .cache_utils import (
    get_narrow_roi_deficit,
    get_failure_count,
    ResourceCache,
)


def prepare_roi(
    frame: np.ndarray,
    regions: Dict[str, Tuple[int, int, int, int]],
    name: str,
    required_set: set[str],
    cache_obj: ResourceCache,
) -> tuple[int, int, int, int, np.ndarray, np.ndarray, int, int] | None:
    """Prepare the region of interest for OCR.

    Returns the ROI coordinates, the color and grayscale crops, the amount of
    top cropping applied, and the current failure count. ``None`` is returned
    when the region has invalid dimensions and is not required.
    """
    x, y, w, h = regions[name]
    deficit = get_narrow_roi_deficit(name)
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
    if name == "wood_stockpile":
        orig_x = x
        orig_w = w
        expand_left = 4
        expand_right = 4
        x = max(0, orig_x - expand_left)
        right = min(frame.shape[1], orig_x + orig_w + expand_right)
        w = right - x
        regions[name] = (x, y, w, h)
        logger.debug(
            "Expanding ROI for %s by %dpx on each side",
            name,
            expand_left,
        )
    if w <= 0 or h <= 0:
        logger.error("ROI for '%s' has invalid dimensions w=%d h=%d", name, w, h)
        if name in required_set:
            raise common.ResourceReadError(f"{name} region has non-positive size")
        return None
    failure_count = get_failure_count(cache_obj, name)
    roi = frame[y : y + h, x : x + w]
    gray = preprocess_roi(roi)
    top_crop = CFG.get("ocr_top_crop", 2)
    overrides = CFG.get("ocr_top_crop_overrides", {})
    if name in overrides:
        top_crop = overrides[name]
    if top_crop > 0 and gray.shape[0] > top_crop:
        gray = gray[top_crop:, :]
    return x, y, w, h, roi, gray, top_crop, failure_count


def expand_roi_after_failure(
    frame: np.ndarray,
    name: str,
    x: int,
    y: int,
    w: int,
    h: int,
    roi: np.ndarray,
    gray: np.ndarray,
    top_crop: int,
    failure_count: int,
    res_conf_threshold: int,
) -> tuple[str, dict, np.ndarray | None, np.ndarray, np.ndarray, int, int, int, int, bool] | None:
    """Expand a resource ROI after a failed OCR attempt."""
    base_expand = CFG.get("ocr_roi_expand_base", CFG.get("ocr_roi_expand_px", 1))
    step = CFG.get("ocr_roi_expand_step", 0)
    growth = CFG.get("ocr_roi_expand_growth", 1.0)
    expand_px = int(round(base_expand + step * ((failure_count + 1) ** growth - 1)))
    if expand_px <= 0:
        return None
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
    digits_exp, data_exp, mask_exp, low_conf = execute_ocr(
        gray_expanded,
        color=roi_expanded,
        conf_threshold=res_conf_threshold,
        roi=(x0, y0, x1 - x0, y1 - y0),
        resource=name,
    )
    if digits_exp:
        return (
            digits_exp,
            data_exp,
            mask_exp,
            roi_expanded,
            gray_expanded,
            x0,
            y0,
            x1 - x0,
            y1 - y0,
            low_conf,
        )
    return None


def expand_population_roi_after_failure(
    frame,
    x: int,
    y: int,
    w: int,
    h: int,
    roi,
    failure_count: int,
    res_conf_threshold: int | None,
    max_right: int | None = None,
):
    """Expand the population ROI after a failed OCR attempt.

    Args:
        frame: Full frame image.
        x, y, w, h: Original ROI bounding box.
        roi: Color ROI image.
        failure_count: Number of consecutive OCR failures.
        res_conf_threshold: Confidence threshold for OCR.
        max_right: Optional right boundary to prevent expansion past the
            idle-villager region.
    """
    base_expand = CFG.get(
        "population_ocr_roi_expand_base",
        CFG.get("population_ocr_roi_expand_px", 1),
    )
    step = CFG.get("population_ocr_roi_expand_step", 0)
    growth = CFG.get("population_ocr_roi_expand_growth", 1.0)
    expand_px = int(round(base_expand + step * ((failure_count + 1) ** growth - 1)))
    if expand_px <= 0:
        return None
    x0 = max(0, x - expand_px)
    y0 = max(0, y - expand_px)
    orig_right = x + w
    x1 = min(frame.shape[1], orig_right + expand_px)
    if max_right is not None:
        x1 = min(max_right, x1)
    y1 = min(frame.shape[0], y + h + expand_px)
    logger.debug(
        "Expanding population ROI after %d failures by %dpx to x=%d y=%d w=%d h=%d",
        failure_count,
        expand_px,
        x0,
        y0,
        x1 - x0,
        y1 - y0,
    )
    roi_expanded = frame[y0:y1, x0:x1]
    try:
        cur_pop, pop_cap = _read_population_from_roi(
            roi_expanded,
            conf_threshold=res_conf_threshold,
            roi_bbox=(x0, y0, x1 - x0, y1 - y0),
            failure_count=failure_count + 1,
        )
        return cur_pop, pop_cap, roi_expanded, x0, y0, x1 - x0, y1 - y0
    except common.PopulationReadError:
        return None

__all__ = [
    "prepare_roi",
    "expand_roi_after_failure",
    "expand_population_roi_after_failure",
]
