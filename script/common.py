"""Utility functions for interacting with the game screen.

This module bundles screen-capture helpers, HUD detection and OCR routines.
"""

from pathlib import Path
import os
import shutil
import logging

import pytesseract
from .config_utils import load_config

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

CFG = load_config()
logger = logging.getLogger(__name__)

tesseract_cmd = os.environ.get("TESSERACT_CMD") or CFG.get("tesseract_path")
path_lookup = shutil.which("tesseract") if not tesseract_cmd else None
if tesseract_cmd:
    tesseract_path = Path(tesseract_cmd)
    if tesseract_path.is_file() and os.access(tesseract_path, os.X_OK):
        pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)
    elif path_lookup:
        logger.warning(
            "Configured Tesseract path '%s' is invalid. Using '%s' found on PATH.",
            tesseract_cmd,
            path_lookup,
        )
        pytesseract.pytesseract.tesseract_cmd = path_lookup
    else:
        raise RuntimeError(
            "Invalid Tesseract OCR path. Install Tesseract or update 'tesseract_path' in config.json."
        )
elif path_lookup:
    pytesseract.pytesseract.tesseract_cmd = path_lookup
else:
    raise RuntimeError(
        "Invalid Tesseract OCR path. Install Tesseract or update 'tesseract_path' in config.json."
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


