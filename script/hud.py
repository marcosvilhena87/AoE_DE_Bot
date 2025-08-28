"""HUD-related helpers extracted from :mod:`script.common`.

This module provides routines for waiting until the game HUD is detected on
screen and reading the population values directly from the HUD.  It relies on
utilities from :mod:`screen_utils` and configuration values from
:mod:`config_utils`.
"""

import logging
import time
from pathlib import Path

import cv2
import pytesseract

from .template_utils import find_template
from .config_utils import load_config
from . import screen_utils
from . import common, resources, input_utils

ROOT = Path(__file__).resolve().parent.parent

CFG = load_config()
logger = logging.getLogger(__name__)


def wait_hud(timeout=60):
    logger.info("Waiting for HUD for up to %ss...", timeout)
    t0 = time.time()
    tmpl = screen_utils.HUD_TEMPLATE
    asset = "assets/resources.png"
    last_score = -1.0
    while time.time() - t0 < timeout:
        frame = screen_utils._grab_frame()
        if tmpl is None:
            break
        box, score, heat = find_template(
            frame, tmpl, threshold=CFG["threshold"], scales=CFG["scales"]
        )
        if box:
            if CFG["debug"]:
                cv2.imwrite(f"debug_hud_{asset}.png", frame)
            x, y, w, h = box
            logger.info("HUD detected with template '%s'", asset)
            common.HUD_ANCHOR = {
                "left": x,
                "top": y,
                "width": w,
                "height": h,
                "asset": asset,
            }
            return common.HUD_ANCHOR, asset
        if score > last_score:
            last_score = score
            logger.debug("HUD template '%s' score improved to %.3f", asset, score)
        time.sleep(0.25)
    logger.debug(
        "HUD search ended. Highest score=%.3f using template '%s'",
        last_score,
        asset,
    )
    logger.error(
        "HUD not found. Best score=%.3f on template '%s'. "
        "Consider recapturing icon templates or checking resolution/SCALING 100%%.",
        last_score,
        asset,
    )
    raise RuntimeError(
        f"HUD not found. Best score={last_score:.3f} on template '{asset}'. "
        "Consider recapturing icon templates or checking resolution/SCALING 100%.",
    )


def read_population_from_hud(retries=1, conf_threshold=None, save_failed_roi=False):
    if conf_threshold is None:
        conf_threshold = CFG.get("ocr_conf_threshold", 60)

    frame_full = screen_utils._grab_frame()
    regions = resources.locate_resource_panel(frame_full)
    pop_cfg = CFG.get("population_limit_roi")
    if pop_cfg:
        W, H = input_utils._screen_size()
        left = int(pop_cfg.get("left_pct", 0) * W)
        top = int(pop_cfg.get("top_pct", 0) * H)
        width = int(pop_cfg.get("width_pct", 0) * W)
        height = int(pop_cfg.get("height_pct", 0) * H)
        regions["population_limit"] = (left, top, width, height)
    roi_bbox = None
    if "population_limit" in regions:
        x, y, w, h = regions["population_limit"]
        roi_bbox = {"left": x, "top": y, "width": w, "height": h}
    else:
        x, y, w, h = CFG["areas"]["pop_box"]
        screen_width, screen_height = input_utils._screen_size()
        abs_left = int(x * screen_width)
        abs_top = int(y * screen_height)
        pw = int(w * screen_width)
        ph = int(h * screen_height)
        if pw <= 0 or ph <= 0:
            raise common.PopulationReadError(
                f"Population ROI has non-positive dimensions after scaling: width={pw}, height={ph}. "
                "Recalibrate areas.pop_box in config.json."
            )
        abs_right = abs_left + pw
        abs_bottom = abs_top + ph
        if (
            abs_left < 0
            or abs_top < 0
            or abs_right > screen_width
            or abs_bottom > screen_height
        ):
            raise common.PopulationReadError(
                "Population ROI out of screen bounds: "
                f"left={abs_left}, top={abs_top}, width={pw}, height={ph}, "
                f"screen={screen_width}x{screen_height}. "
                "Recalibrate areas.pop_box in config.json or use template anchoring."
            )
        roi_bbox = {"left": abs_left, "top": abs_top, "width": pw, "height": ph}

    last_roi = None
    last_thresh = None
    last_text = ""
    last_confidences = []

    for attempt in range(retries):
        roi = screen_utils._grab_frame(roi_bbox)
        if roi.size == 0:
            logger.warning("Population ROI has zero size")
            continue
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
        last_roi = roi
        last_thresh = thresh
        last_text = text
        last_confidences = confidences
        parts = [p for p in text.split("/") if p]
        if len(parts) >= 2 and (not confidences or min(confidences) >= conf_threshold):
            cur = int("".join(filter(str.isdigit, parts[0])) or 0)
            limit = int("".join(filter(str.isdigit, parts[1])) or 0)
            return cur, limit
        logger.debug(
            "OCR attempt %s failed: text='%s', conf=%s",
            attempt + 1,
            text,
            confidences,
        )
        time.sleep(0.1)

    logger.warning(
        "Failed to read population from HUD after %s attempts; last text='%s', conf=%s",
        retries,
        last_text,
        last_confidences,
    )
    if (CFG.get("debug") or save_failed_roi) and last_roi is not None:
        ts = int(time.time() * 1000)
        cv2.imwrite(str(ROOT / f"debug_pop_roi_{ts}.png"), last_roi)
        cv2.imwrite(str(ROOT / f"debug_pop_thresh_{ts}.png"), last_thresh)
        logger.info(
            "ROI saved; extracted text: '%s'; conf=%s",
            last_text,
            last_confidences,
        )
    logger.info(
        "Triggering fallback to read population via resources.read_resources_from_hud"
    )
    try:
        _, (cur, limit) = resources.read_resources_from_hud(
            ["population_limit"], force_delay=0.1, conf_threshold=conf_threshold
        )
        if cur is not None and limit is not None:
            logger.info("Population fallback succeeded: %s/%s", cur, limit)
            return cur, limit
    except Exception as exc:  # pragma: no cover - log but ignore any failure
        logger.debug("Fallback failed: %s", exc)

    raise common.PopulationReadError(
        f"Failed to read population from HUD after {retries} attempts. Text='{last_text}', confs={last_confidences}"
    )
