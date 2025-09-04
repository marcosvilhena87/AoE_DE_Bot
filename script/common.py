"""Utility functions and shared state for interacting with the game screen."""

from pathlib import Path
import os
import shutil
import logging
from dataclasses import dataclass, field
from typing import Any, Dict

import pytesseract
from .config_utils import load_config

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


@dataclass
class BotState:
    """Container for global bot state."""

    config: Dict[str, Any] = field(default_factory=dict)
    current_pop: int = 0
    pop_cap: int = 0
    target_pop: int = 0


# Default shared state instance
STATE = BotState()
logger = logging.getLogger(__name__)

# Backwards compatibility for modules/tests importing ``common.CFG``
CFG = STATE.config


def init_common(path: str | Path | None = None, state: BotState | None = None) -> BotState:
    """Load configuration and configure Tesseract.

    Parameters
    ----------
    path:
        Optional path to the configuration file. Defaults to the standard
        configuration file when ``None``.
    state:
        Optional :class:`BotState` instance to update. If ``None`` the module
        level :data:`STATE` is used.

    Returns
    -------
    BotState
        The updated state instance.
    """

    if state is None:
        state = STATE

    state.config.clear()
    state.config.update(load_config(path))

    tesseract_cmd = os.environ.get("TESSERACT_CMD") or state.config.get("tesseract_path")
    path_lookup = shutil.which("tesseract")
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
                "Invalid Tesseract OCR path. Install Tesseract or update 'tesseract_path' in config.json.",
            )
    elif path_lookup:
        pytesseract.pytesseract.tesseract_cmd = path_lookup
    else:
        raise RuntimeError(
            "Invalid Tesseract OCR path. Install Tesseract or update 'tesseract_path' in config.json.",
        )

    return state


# Posição detectada do HUD usada apenas como referência
HUD_ANCHOR = None


class PopulationReadError(RuntimeError):
    """Raised when population values cannot be extracted from the HUD."""


class ResourceReadError(RuntimeError):
    """Raised when resource values cannot be extracted from the HUD."""

