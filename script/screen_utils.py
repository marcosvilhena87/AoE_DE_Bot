"""Screen capture and template utilities."""

import logging
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import cv2
from mss import mss

from .config_utils import load_config

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

CFG = load_config()
logger = logging.getLogger(__name__)

SCT = None
MONITOR = None


@contextmanager
def mss_session():
    """Provide an ``mss`` instance and ensure it is closed."""
    sct = mss()
    try:
        yield sct
    finally:
        close = getattr(sct, "close", None)
        if close:
            close()


def init_sct():
    """Initialise global screen capture resources."""
    global SCT, MONITOR
    if SCT is None:
        SCT = mss()
        MONITOR = SCT.monitors[1]


def teardown_sct():
    """Release global screen capture resources."""
    global SCT, MONITOR
    if SCT is not None:
        close = getattr(SCT, "close", None)
        if close:
            close()
        SCT = None
        MONITOR = None


def get_monitor():
    """Return the active monitor, initialising resources if needed."""
    if MONITOR is None:
        init_sct()
    return MONITOR


def _grab_frame(bbox=None):
    """Capture a frame from the screen."""
    init_sct()
    region = bbox or get_monitor()
    img = np.array(SCT.grab(region))[:, :, :3]
    return img


def _load_gray(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        logger.warning("Asset não encontrado: %s", path)
        return None
    return im


HUD_TEMPLATE = _load_gray(ROOT / "assets/resources.png")

ICON_NAMES = [
    "wood_stockpile",
    "food_stockpile",
    "gold",
    "stone",
    "population",
    "idle_villager",
]
ICON_TEMPLATES = {}


def _load_icon_templates():
    icons_dir = ASSETS / "icons"
    for name in ICON_NAMES:
        if name in ICON_TEMPLATES:
            continue
        path = icons_dir / f"{name}.png"
        icon = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if icon is None:
            logger.warning("Icon asset missing: %s", path)
            continue
        ICON_TEMPLATES[name] = icon
