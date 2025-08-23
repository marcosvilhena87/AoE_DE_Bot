"""Utility functions for interacting with the game screen.

This module bundles screen-capture helpers, HUD detection and OCR routines.
"""

import logging
from pathlib import Path
import os

import pytesseract
from .config_utils import load_config

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

CFG = load_config()

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


