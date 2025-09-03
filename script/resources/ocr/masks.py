"""Masking logic for OCR digit recognition."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytesseract

from .. import CFG, ROOT
from .colors import color_mask_sets
from .confidence import parse_confidences


def _run_masks(
    masks,
    psms,
    debug,
    debug_dir,
    ts,
    start_idx=0,
    whitelist="0123456789",
    resource=None,
):
    results = []
    for idx, mask in enumerate(masks, start=start_idx):
        if debug and debug_dir is not None:
            debug_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(debug_dir / f"ocr_mask_{ts}_{idx}.png"), mask)
        for psm in psms:
            data = pytesseract.image_to_data(
                mask,
                config=f"--psm {psm} --oem 3 -c tessedit_char_whitelist={whitelist}",
                output_type=pytesseract.Output.DICT,
            )
            text = "".join(data.get("text", [])).strip()
            digits = "".join(filter(str.isdigit, text))
            results.append((digits, data, mask, text))
    if not results:
        return "", {"text": [], "conf": []}, None

    scored = []
    for digits, data, mask, text in results:
        confs = parse_confidences(data)
        metric = float(np.median(confs)) if confs else 0.0
        if digits:
            has_slash = resource == "population_limit" and "/" in text
            scored.append((digits, data, mask, metric, has_slash))

    if not scored:
        first = results[0]
        return first[0], first[1], first[2]

    scored.sort(key=lambda r: (-int(r[4]), -len(r[0]), -r[3]))
    best = scored[0]
    return best[0], best[1], best[2]


def _is_nearly_empty(mask, tol=0.01):
    if mask.size == 0:
        return True
    ratio = cv2.countNonZero(mask) / mask.size
    return ratio < tol or ratio > 1 - tol


def _ocr_digits_better(gray, color=None, resource=None, whitelist="0123456789"):
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
    psms = list(dict.fromkeys(CFG.get("ocr_psm_list", []) + [6, 7, 8, 10, 13]))

    debug = CFG.get("ocr_debug")
    debug_dir = ROOT / "debug" if debug else None
    ts = int(time.time() * 1000) if debug else None

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    ws_dilate_var = CFG.get("ocr_ws_dilate_var", 50)
    ws_should_dilate = (
        resource == "wood_stockpile"
        and (
            CFG.get("wood_stockpile_dilate", False)
            or (ws_dilate_var > 0 and variance < ws_dilate_var)
        )
    )
    if ws_should_dilate:
        # Avoid an aggressive cross-shaped close which can break digit loops.
        ws_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.dilate(thresh, ws_dilate_kernel, iterations=1)
    if _is_nearly_empty(thresh):
        _otsu_ret, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    digits, data, mask = _run_masks(
        [thresh, cv2.bitwise_not(thresh)],
        psms,
        debug,
        debug_dir,
        ts,
        0,
        whitelist=whitelist,
        resource=resource,
    )
    threshold = CFG.get("ocr_conf_threshold", 60)
    if resource:
        threshold = CFG.get(f"{resource}_ocr_conf_threshold", threshold)
    confs = parse_confidences(data)

    if (
        resource in {"wood_stockpile", "food_stockpile"}
        and color is not None
        and CFG.get(f"{resource}_color_pass", True)
    ):
        hsv_res = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(
            hsv_res, np.array([0, 0, 170]), np.array([180, 80, 255])
        )
        gray = cv2.inRange(
            hsv_res, np.array([0, 0, 120]), np.array([180, 80, 220])
        )
        mask_color = cv2.bitwise_or(white, gray)
        color_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_color = cv2.morphologyEx(
            mask_color, cv2.MORPH_CLOSE, color_kernel, iterations=1
        )
        if ws_should_dilate:
            mask_color = cv2.dilate(mask_color, color_kernel, iterations=1)
        alt_digits, alt_data, alt_mask = _run_masks(
            [mask_color, cv2.bitwise_not(mask_color)],
            psms,
            debug,
            debug_dir,
            ts,
            4,
            whitelist=whitelist,
            resource=resource,
        )
        alt_confs = parse_confidences(alt_data)
        if alt_digits and (
            not digits or (alt_confs and max(alt_confs) >= max(confs or [0]))
        ):
            digits, data, mask = alt_digits, alt_data, alt_mask
            confs = alt_confs

    if digits and confs and max(confs) >= threshold:
        return digits, data, mask

    block_size = 21 if resource == "population_limit" else 11
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 2
    )
    if resource == "population_limit":
        pop_close_cfg = CFG.get("population_morph_close", True)
        do_pop_close = bool(pop_close_cfg)
        pop_kernel_size = (1, 3)
        var_thresh = None
        if isinstance(pop_close_cfg, dict):
            pop_kernel_size = tuple(pop_close_cfg.get("kernel", pop_close_cfg.get("kernel_size", pop_kernel_size)))
            var_thresh = pop_close_cfg.get("variance_threshold")
            do_pop_close = pop_close_cfg.get("enabled", True)
        if do_pop_close and (var_thresh is None or variance <= var_thresh):
            edges = cv2.Canny(adaptive, 50, 150)
            edge_density = (
                cv2.countNonZero(edges) / edges.size if edges.size else 0.0
            )
            slash_likely = edge_density > CFG.get(
                "population_slash_edge_density", 0.15
            )
            if not slash_likely:
                pop_kernel = cv2.getStructuringElement(
                    cv2.MORPH_CROSS, pop_kernel_size
                )
                adaptive = cv2.morphologyEx(
                    adaptive, cv2.MORPH_CLOSE, pop_kernel, iterations=1
                )
        _otsu_ret, otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        digits, data, mask = _run_masks(
            [otsu, cv2.bitwise_not(otsu)],
            psms,
            debug,
            debug_dir,
            ts,
            2,
            whitelist=whitelist,
            resource=resource,
        )
        if digits:
            return digits, data, mask
        # Morphological opening with a cross-shaped kernel would remove
        # diagonal strokes (e.g. the '/' separating current and maximum
        # population). Skip this step to preserve such characters.
        # pl_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        # adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, pl_kernel, iterations=1)
    else:
        adaptive = cv2.dilate(adaptive, kernel, iterations=1)
        if resource == "wood_stockpile":
            ws_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, ws_kernel, iterations=1)
            if ws_should_dilate:
                ws_dilate_kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (3, 3)
                )
                adaptive = cv2.dilate(adaptive, ws_dilate_kernel, iterations=1)
    if _is_nearly_empty(adaptive):
        _otsu_ret, adaptive = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    adaptive_start = (
        6
        if resource in {"wood_stockpile", "food_stockpile"}
        else (4 if resource == "population_limit" else 2)
    )
    digits, data, mask = _run_masks(
        [adaptive, cv2.bitwise_not(adaptive)],
        psms,
        debug,
        debug_dir,
        ts,
        adaptive_start,
        whitelist=whitelist,
        resource=resource,
    )

    confs = parse_confidences(data)
    if digits and confs and max(confs) >= threshold:
        return digits, data, mask

    if not digits:
        if color is not None:
            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        else:
            hsv = cv2.cvtColor(cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
        base_masks, closed_masks = color_mask_sets(hsv, resource, kernel)
        digits, data, mask = _run_masks(
            base_masks,
            psms,
            debug,
            debug_dir,
            ts,
            adaptive_start + 2,
            whitelist=whitelist,
            resource=resource,
        )
        if not digits:
            digits, data, mask = _run_masks(
                closed_masks,
                psms,
                debug,
                debug_dir,
                ts,
                adaptive_start + 4,
                whitelist=whitelist,
                resource=resource,
            )

    if not digits:
        variance = float(np.var(orig))
        if variance < CFG.get("ocr_zero_variance", 15.0):
            return "0", {"zero_variance": True}, mask

    return digits, data, mask


__all__ = ["_ocr_digits_better", "_run_masks", "_is_nearly_empty"]
