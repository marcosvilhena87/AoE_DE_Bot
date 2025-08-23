"""Utility functions for interacting with the game screen.

This module bundles screen-capture helpers, HUD detection and OCR routines.
"""

import logging
import time
from pathlib import Path
import os

import pyautogui as pg
import pytesseract
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


class PopulationReadError(RuntimeError):
    """Raised when population values cannot be extracted from the HUD."""


class ResourceReadError(RuntimeError):
    """Raised when resource values cannot be extracted from the HUD."""


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
