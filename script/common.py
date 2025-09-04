"""Utility functions and shared state for interacting with the game screen."""

from pathlib import Path
import os
import shutil
import logging
from dataclasses import dataclass, field
from typing import Any

from config import Config

import pytesseract
from .config_utils import load_config

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


@dataclass
class BotState:
    """Container for global bot state."""

    config: Config = field(default_factory=Config)
    current_pop: int = 0
    pop_cap: int = 0
    target_pop: int = 0


# Default shared state instance
STATE = BotState()
logger = logging.getLogger(__name__)

# Backwards compatibility for modules/tests importing ``common.CFG``
CFG: Config = STATE.config


def resolve_tesseract_path(cfg: Config) -> str:
    """Return a valid Tesseract executable path.

    The path is resolved using the ``TESSERACT_CMD`` environment variable,
    the ``tesseract_path`` value from the provided configuration, or by
    falling back to ``shutil.which``. A :class:`RuntimeError` is raised if no
    usable path can be found.
    """

    tesseract_cmd = os.environ.get("TESSERACT_CMD") or cfg.get("tesseract_path")
    path_lookup = shutil.which("tesseract")

    if tesseract_cmd:
        tesseract_path = Path(tesseract_cmd)
        if tesseract_path.is_file() and os.access(tesseract_path, os.X_OK):
            return str(tesseract_path)
        if path_lookup:
            logger.warning(
                "Configured Tesseract path '%s' is invalid. Using '%s' found on PATH.",
                tesseract_cmd,
                path_lookup,
            )
            return path_lookup

    if path_lookup:
        return path_lookup

    raise RuntimeError(
        "Invalid Tesseract OCR path. Install Tesseract or update 'tesseract_path' in config.json."
    )


def init_common(path: str | Path | None = None, state: BotState | None = None) -> BotState:
    """Load configuration and configure Tesseract."""

    if state is None:
        state = STATE

    state.config.clear()
    state.config.update(load_config(path))

    pytesseract.pytesseract.tesseract_cmd = resolve_tesseract_path(state.config)

    return state


# Posição detectada do HUD usada apenas como referência
HUD_ANCHOR = None


class PopulationReadError(RuntimeError):
    """Raised when population values cannot be extracted from the HUD."""


class ResourceReadError(RuntimeError):
    """Raised when resource values cannot be extracted from the HUD."""

