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

from .template_utils import find_template
from .config_utils import load_config
from . import screen_utils
from . import common, resources, input_utils
from .resources import reader as resource_reader

ROOT = Path(__file__).resolve().parent.parent

CFG = load_config()
logger = logging.getLogger(__name__)


def wait_hud(timeout=60):
    logger.info("Waiting for HUD for up to %ss...", timeout)
    t0 = time.time()
    tmpl = screen_utils.HUD_TEMPLATE
    asset = "assets/resources.png"
    if tmpl is None:
        raise RuntimeError(
            f"HUD template '{asset}' not loaded; ensure {asset} exists"
        )
    last_score = -1.0
    while time.time() - t0 < timeout:
        frame = screen_utils.grab_frame()
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


def calculate_population_roi(frame_full) -> dict:
    """Compute the population ROI from the given frame.

    This routine encapsulates all logic for determining the region of interest
    for population OCR. It considers configured overrides, detected icons and
    clamps the ROI to avoid overlapping with other HUD elements.

    Parameters
    ----------
    frame_full:
        Full screen capture used for locating HUD elements.

    Returns
    -------
    dict
        Bounding box with ``left``, ``top``, ``width`` and ``height`` keys.
    """

    regions = resources.locate_resource_panel(frame_full)
    pop_cfg = CFG.get("population_limit_roi")
    if pop_cfg:
        W, H = input_utils._screen_size()
        left = int(pop_cfg.get("left_pct", 0) * W)
        top = int(pop_cfg.get("top_pct", 0) * H)
        width = int(pop_cfg.get("width_pct", 0) * W)
        height = int(pop_cfg.get("height_pct", 0) * H)
        regions["population_limit"] = (left, top, width, height)
    idle_bounds = regions.get("idle_villager")
    if idle_bounds and "population_limit" in regions:
        left, top, width, height = regions["population_limit"]
        idle_left = idle_bounds[0]
        clamp_right = idle_left - CFG.get("population_idle_padding", 0)
        if clamp_right < left + width:
            width = max(0, clamp_right - left)
            regions["population_limit"] = (left, top, width, height)
        logger.debug(
            "Population ROI after clamp: x=%d y=%d w=%d h=%d",
            left,
            top,
            width,
            height,
        )
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

    logger.info(
        "Population limit ROI for OCR: left=%d top=%d width=%d height=%d",
        roi_bbox["left"],
        roi_bbox["top"],
        roi_bbox["width"],
        roi_bbox["height"],
    )
    return roi_bbox


def read_population_from_hud(retries=1, conf_threshold=None, save_failed_roi=False):
    if conf_threshold is None:
        conf_threshold = CFG.get("ocr_conf_threshold", 60)

    frame_full = screen_utils.grab_frame()
    roi_bbox = calculate_population_roi(frame_full)
    try:
        cur, limit, low_conf = resources.read_population_from_roi(
            roi_bbox,
            retries=retries,
            conf_threshold=conf_threshold,
            save_failed_roi=save_failed_roi,
        )
        if low_conf and CFG.get("treat_low_conf_as_failure", False):
            err = common.PopulationReadError("Low-confidence population OCR")
            err.low_conf = True
            err.low_conf_digits = (cur, limit)
            raise err
        return cur, limit, low_conf
    except common.PopulationReadError as primary_exc:
        logger.info(
            "Triggering fallback to read population via resources.read_resources_from_hud after %s",
            primary_exc,
        )
        fallback_exc = None
        try:
            _, (cur, limit) = resource_reader.read_resources_from_hud(
                ["population_limit"], force_delay=0.1, conf_threshold=conf_threshold
            )
            if cur is not None and limit is not None:
                logger.info("Population fallback succeeded: %s/%s", cur, limit)
                return cur, limit, False
        except (common.ResourceReadError, common.PopulationReadError) as exc:
            # pragma: no cover - log but ignore expected OCR failures
            logger.debug("Fallback failed: %s", exc)
            fallback_exc = exc

        if fallback_exc is not None:
            raise common.PopulationReadError(
                f"{primary_exc}; fallback failed: {fallback_exc}"
            ) from fallback_exc
        raise

