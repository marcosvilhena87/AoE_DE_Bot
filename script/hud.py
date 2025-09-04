"""HUD-related helpers extracted from :mod:`script.common`.

This module provides routines for waiting until the game HUD is detected on
screen. Population values are sourced from the internal :data:`STATE` rather
than performing OCR on the HUD.
"""

import logging
import time

from config import Config

from .template_utils import find_template
from . import screen_utils, common
from .common import STATE

CFG: Config = STATE.config

logger = logging.getLogger(__name__)


def wait_hud(cfg: Config = CFG, timeout=60):
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
        frame = screen_utils.screen_capture.grab_frame()
        box, score, heat = find_template(
            frame, tmpl, threshold=cfg["threshold"], scales=cfg["scales"]
        )
        if box:
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


def read_population_from_hud(
    cfg: Config = CFG,
    retries: int = 1,
    conf_threshold: int | None = None,
    save_failed_roi: bool = False,
):
    """Return current and maximum population using cached state values.

    Parameters are accepted for backward compatibility but are ignored. The
    function performs no screen capture or OCR.
    """

    return STATE.current_pop, STATE.pop_cap, False

