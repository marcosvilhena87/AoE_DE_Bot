"""Screen capture and template utilities."""

import logging
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import cv2
from mss import mss

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Manage ``mss`` screen capture resources.

    This class lazily creates an ``mss`` instance and stores the active
    monitor. It can be used as a context manager to ensure resources are
    released.
    """

    def __init__(self):
        self._sct = None
        self._monitor = None

    def __enter__(self):
        self.init_sct()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.teardown_sct()

    def init_sct(self):
        """Initialise screen capture resources."""
        if self._sct is None:
            self._sct = mss()
            self._monitor = self._sct.monitors[1]

    def teardown_sct(self):
        """Release screen capture resources."""
        if self._sct is not None:
            close = getattr(self._sct, "close", None)
            if close:
                close()
            self._sct = None
            self._monitor = None

    def get_monitor(self):
        """Return the active monitor, initialising resources if needed."""
        if self._monitor is None:
            self.init_sct()
        return self._monitor

    def _grab_frame(self, bbox=None):
        """Capture a frame from the screen."""
        self.init_sct()
        region = bbox or self.get_monitor()
        img = np.array(self._sct.grab(region))[:, :, :3]
        return img

    def grab_frame(self, bbox=None):
        """Public wrapper around :meth:`_grab_frame`."""
        return self._grab_frame(bbox)


# Default screen capture manager used by module-level helpers
screen_capture = ScreenCapture()


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
    screen_capture.init_sct()


def teardown_sct():
    screen_capture.teardown_sct()


def get_monitor():
    return screen_capture.get_monitor()


def get_screen_size():
    """Return the width and height of the active monitor.

    Returns
    -------
    tuple[int, int]
        ``(width, height)`` of the current monitor in pixels.
    """

    monitor = get_monitor()
    return monitor["width"], monitor["height"]


def grab_frame(bbox=None):
    return screen_capture.grab_frame(bbox)


def _load_gray(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        logger.warning("Asset not found: %s", path)
        return None
    return im


HUD_TEMPLATE = _load_gray(ROOT / "assets/resources.png")

ICONS_DIR = ASSETS / "icons"
ICON_NAMES = [p.stem for p in sorted(ICONS_DIR.glob("*.png"))]
if "idle_villager" not in ICON_NAMES:
    ICON_NAMES.append("idle_villager")
ICON_TEMPLATES = {}


def load_icon_templates():
    for name in ICON_NAMES:
        if name in ICON_TEMPLATES:
            continue
        path = ICONS_DIR / f"{name}.png"
        icon = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if icon is None:
            logger.warning("Icon asset missing: %s", path)
            continue
        ICON_TEMPLATES[name] = icon


__all__ = [
    "ScreenCapture",
    "screen_capture",
    "mss_session",
    "init_sct",
    "teardown_sct",
    "get_monitor",
    "get_screen_size",
    "grab_frame",
    "load_icon_templates",
    "HUD_TEMPLATE",
    "ICON_NAMES",
    "ICON_TEMPLATES",
]
