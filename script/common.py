"""Utility functions for interacting with the game screen.

This module bundles screen-capture helpers, HUD detection and OCR routines.
Resource reading first tries a thresholded OCR via :func:`_ocr_digits_better`.
If no digits are detected it falls back to ``pytesseract.image_to_string`` on
the raw region before reporting a failure.
"""

import logging
import time
from pathlib import Path
import os

import numpy as np
import cv2
import pyautogui as pg
import pytesseract
from .template_utils import find_template
from .config_utils import CFG
from . import screen_utils

# =========================
# CONFIGURAÇÃO
# =========================
pg.PAUSE = 0.05
pg.FAILSAFE = True  # mouse no canto sup-esq aborta instantaneamente

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

tesseract_cmd = os.environ.get("TESSERACT_CMD") or CFG.get("tesseract_path")
if tesseract_cmd:
    tesseract_path = Path(tesseract_cmd)
    if not (tesseract_path.is_file() and os.access(tesseract_path, os.X_OK)):
        raise RuntimeError(
            "Invalid Tesseract OCR path. Install Tesseract or update 'tesseract_path' in config.json."
        )
    pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)

logging.basicConfig(
    level=logging.DEBUG if CFG.get("verbose_logging") else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Contadores internos de população
CURRENT_POP = 0
POP_CAP = 0
TARGET_POP = 0

# Posição detectada do HUD usada apenas como referência
HUD_ANCHOR = None
# Últimas posições detectadas dos ícones de recurso
_LAST_ICON_BOUNDS = {}

class PopulationReadError(RuntimeError):
    """Raised when population values cannot be extracted from the HUD."""


class ResourceReadError(RuntimeError):
    """Raised when resource values cannot be extracted from the HUD."""

# =========================
# RECURSOS NA HUD
# =========================

def locate_resource_panel(frame):
    """Locate the resource panel and return bounding boxes for each value."""

    tmpl = screen_utils.HUD_TEMPLATES.get("assets/resources.png")
    if tmpl is None:
        return {}

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
        logging.warning(
            "Resource panel template not matched; score=%.3f", score
        )
        _save_debug(frame, heat)
        fallback = CFG.get("threshold_fallback")
        if fallback is not None:
            box, score, heat = find_template(
                frame, tmpl, threshold=fallback, scales=CFG["scales"]
            )
            if not box:
                logging.warning(
                    "Resource panel template not matched with fallback; score=%.3f",
                    score,
                )
                _save_debug(frame, heat)
                return {}
        else:
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
    min_width = res_cfg.get("min_width", 60)
    top_pct = profile_res.get("top_pct", res_cfg.get("top_pct", 0.08))
    height_pct = profile_res.get("height_pct", res_cfg.get("height_pct", 0.84))
    screen_utils._load_icon_templates()
    detections = []
    global _LAST_ICON_BOUNDS
    for name in screen_utils.ICON_NAMES:
        icon = screen_utils.ICON_TEMPLATES.get(name)
        if icon is None:
            continue
        best = (-1, None, None)  # score, loc, (w,h)
        for scale in scales:
            icon_scaled = cv2.resize(icon, None, fx=scale, fy=scale)
            if (
                icon_scaled.shape[0] > panel_gray.shape[0]
                or icon_scaled.shape[1] > panel_gray.shape[1]
            ):
                continue
            result = cv2.matchTemplate(panel_gray, icon_scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best[0]:
                best = (max_val, max_loc, icon_scaled.shape[::-1])
        if best[0] >= match_threshold and best[1] is not None:
            (bw, bh) = best[2]
            detections.append((name, best[1][0], best[1][1], bw, bh))
            _LAST_ICON_BOUNDS[name] = (best[1][0], best[1][1], bw, bh)
        elif name in _LAST_ICON_BOUNDS:
            logging.info(
                "Using previous position for icon '%s'; score=%.3f", name, best[0]
            )
            detections.append((name, *_LAST_ICON_BOUNDS[name]))
        else:
            logging.warning("Icon '%s' not matched; score=%.3f", name, best[0])

    detections.sort(key=lambda d: d[1])  # sort by x position
    top = y + int(top_pct * h)
    height = int(height_pct * h)
    regions = {}
    for idx, (name, xi, yi, wi, hi) in enumerate(detections):
        if name == "idle_villager":
            left = x + xi + pad_left
            right = x + xi + wi - pad_right
            top_i = y + yi
            height_i = hi
            width = max(1, right - left)
        else:
            left = x + xi + wi + pad_left
            if idx + 1 < len(detections):
                right = x + detections[idx + 1][1] - pad_right
            else:
                right = x + w - pad_right
            width = max(min_width, right - left)
            top_i = top
            height_i = height
        regions[name] = (left, top_i, width, height_i)

    return regions


def _ocr_digits_better(gray):
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

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

def read_resources_from_hud():
    frame = screen_utils._grab_frame()
    h_full, w_full = frame.shape[:2]
    regions = locate_resource_panel(frame)

    required_icons = [
        "wood_stockpile",
        "food_stockpile",
        "gold",
        "stone",
        "population",
        "idle_villager",
    ]

    missing = [name for name in required_icons if name not in regions]

    if missing and HUD_ANCHOR:
        if HUD_ANCHOR.get("asset") == "assets/resources.png":
            x = HUD_ANCHOR["left"]
            y = HUD_ANCHOR["top"]
            w = HUD_ANCHOR["width"]
            h = HUD_ANCHOR["height"]

            slice_w = w / 6
            res_cfg = CFG.get("resource_panel", {})
            profile = CFG.get("profile")
            profile_res = CFG.get("profiles", {}).get(profile, {}).get(
                "resource_panel", {}
            )
            top_pct = profile_res.get("top_pct", res_cfg.get("top_pct", 0.08))
            height_pct = profile_res.get("height_pct", res_cfg.get("height_pct", 0.84))
            icon_trims = profile_res.get(
                "icon_trim_pct",
                res_cfg.get(
                    "icon_trim_pct",
                    [0.25, 0.20, 0.20, 0.20, 0.20, 0.20],
                ),
            )
            if not isinstance(icon_trims, (list, tuple)):
                icon_trims = [icon_trims] * 6
            right_trim = profile_res.get(
                "right_trim_pct", res_cfg.get("right_trim_pct", 0.02)
            )

            top = y + int(top_pct * h)
            height = int(height_pct * h)
            regions = {}
            for idx, name in enumerate(required_icons):
                icon_trim = icon_trims[idx] if idx < len(icon_trims) else icon_trims[-1]
                left = x + int(idx * slice_w + icon_trim * slice_w)
                right_limit = x + int((idx + 1) * slice_w - right_trim * slice_w)
                width = max(60, right_limit - left)
                regions[name] = (left, top, width, height)
        else:
            # Fallback: estimate the resource bar position from the previously
            # detected HUD anchor.  The anchor is assumed to sit at the top of the
            # screen and the resource panel spans to its right.  The original
            # template ``resources.png`` measured 568x59 px on a
            # 1920x1080 display.  Offsets are configurable via
            # ``resource_panel`` entries in ``config.json``.
            W, H = _screen_size()
            margin = int(0.01 * W)  # ~1% horizontal gap between anchor and panel
            panel_w = int(568 / 1920 * W)
            panel_h = int(59 / 1080 * H)
            x = HUD_ANCHOR["left"] + HUD_ANCHOR["width"] + margin
            y = HUD_ANCHOR["top"]

            res_cfg = CFG.get("resource_panel", {})
            profile = CFG.get("profile")
            profile_res = CFG.get("profiles", {}).get(profile, {}).get(
                "resource_panel", {}
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
            regions = {}
            for idx, name in enumerate(required_icons):
                icon_trim = icon_trims[idx] if idx < len(icon_trims) else icon_trims[-1]
                left = x + int(idx * slice_w + icon_trim * slice_w)
                right_limit = x + int((idx + 1) * slice_w - right_trim * slice_w)
                width = max(60, right_limit - left)
                regions[name] = (left, top, width, height)

        missing = [name for name in required_icons if name not in regions]

    if missing:
        raise ResourceReadError(
            "Resource icon(s) not located on HUD: " + ", ".join(missing)
        )

    results = {}
    for name, (x, y, w, h) in regions.items():
        roi = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)

        digits, data, mask = _ocr_digits_better(gray)
        if CFG.get("ocr_debug"):
            debug_dir = ROOT / "debug"
            debug_dir.mkdir(exist_ok=True)
            ts = int(time.time() * 1000)
            cv2.imwrite(str(debug_dir / f"resource_{name}_roi_{ts}.png"), roi)
            if mask is not None:
                cv2.imwrite(str(debug_dir / f"resource_{name}_thresh_{ts}.png"), mask)
        if not digits:
            text = pytesseract.image_to_string(
                gray,
                config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789",
            )
            fallback = "".join(filter(str.isdigit, text))
            if fallback:
                digits = fallback
                data = {"text": [text.strip()]}
                mask = gray
        if not digits:
            logging.warning(
                "OCR failed for %s; raw boxes=%s", name, data.get("text")
            )
            debug_dir = ROOT / "debug"
            debug_dir.mkdir(exist_ok=True)
            ts = int(time.time() * 1000)
            cv2.imwrite(str(debug_dir / f"resource_{name}_roi_{ts}.png"), roi)
            if mask is not None:
                cv2.imwrite(
                    str(debug_dir / f"resource_{name}_thresh_{ts}.png"), mask
                )
            results[name] = None
        else:
            results[name] = int(digits)

    failed = [name for name, v in results.items() if v is None]
    if failed:
        debug_dir = ROOT / "debug"
        debug_dir.mkdir(exist_ok=True)
        ts = int(time.time() * 1000)

        # Annotate a copy of the full frame with the ROI bounding boxes
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

        # Log ROI coordinates alongside their debug image paths for easier inspection
        logging.error(
            "Resource panel OCR failed for %s; panel saved to %s; rois: %s",
            ", ".join(failed),
            panel_path,
            ", ".join(roi_logs),
        )

        tess_path = pytesseract.pytesseract.tesseract_cmd
        paths_str = ", ".join([str(panel_path)] + roi_paths)
        failed_regions = {k: regions[k] for k in failed}
        raise ResourceReadError(
            "OCR failed to read resource values for "
            + ", ".join(failed)
            + f" (regions={failed_regions}, tesseract_cmd={tess_path}, debug_images={paths_str})"
        )

    return results

# =========================
# AÇÕES DE JOGO BÁSICAS
# =========================

def _screen_size():
    return screen_utils.MONITOR["width"], screen_utils.MONITOR["height"]

def _to_px(nx, ny):
    W, H = _screen_size()
    return int(nx * W), int(ny * H)

def _click_norm(nx, ny, button="left"):
    x, y = _to_px(nx, ny)
    try:
        pg.click(x, y, button=button)
    except pg.FailSafeException:
        logging.warning(
            "Fail-safe triggered during click at (%s, %s). Moving cursor to center.",
            x,
            y,
        )
        _move_cursor_safe()
        pg.click(x, y, button=button)

def _move_cursor_safe():
    W, H = _screen_size()
    failsafe_state = pg.FAILSAFE
    pg.FAILSAFE = False
    pg.moveTo(W // 2, H // 2)
    pg.FAILSAFE = failsafe_state

def _press_key_safe(key, pause):
    try:
        pg.press(key)
        time.sleep(pause)
    except pg.FailSafeException:
        logging.warning(
            "Fail-safe triggered while pressing '%s'. Moving cursor to center.",
            key,
        )
        _move_cursor_safe()
        pg.press(key)
        time.sleep(pause)
