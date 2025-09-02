"""Configuration helpers and scenario parsing utilities.

This module centralises loading and validation of ``config.json`` as well as
helpers for reading simple scenario information.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

ROOT = Path(__file__).resolve().parent.parent
_CFG_CACHE: dict[str, Any] | None = None

logger = logging.getLogger(__name__)


def validate_config(cfg: dict[str, Any]) -> None:
    """Ensure mandatory config sections and coordinates are present.

    Raises
    ------
    RuntimeError
        If required sections or fields are missing, suggesting how to fix it.
    """

    if "areas" not in cfg:
        raise RuntimeError(
            "Missing mandatory 'areas' section in config.json. "
            "Copy values from config.sample.json or run the calibration tools."
        )

    required_coords = [
        "house_spot",
        "granary_spot",
        "storage_spot",
        "wood_spot",
        "food_spot",
        "pop_box",
    ]
    areas = cfg.get("areas", {})
    missing = [name for name in required_coords if name not in areas]
    if missing:
        raise RuntimeError(
            "Missing required coordinate(s) in 'areas': "
            + ", ".join(missing)
            + ". Copy them from config.sample.json or run the calibration tools."
        )

    if "keys" not in cfg:
        raise RuntimeError(
            "Missing mandatory 'keys' section in config.json. "
            "Copy values from config.sample.json."
        )

    required_keys = ["idle_vill", "build_menu", "house", "select_tc", "train_vill"]
    keys = cfg.get("keys", {})
    missing_keys = [k for k in required_keys if not keys.get(k)]
    if missing_keys:
        raise RuntimeError(
            "Missing required hotkey(s) in 'keys': "
            + ", ".join(missing_keys)
            + ". Copy them from config.sample.json."
        )

    allow_low_conf = cfg.get("allow_low_conf_digits")
    if allow_low_conf is not None and not isinstance(allow_low_conf, bool):
        raise RuntimeError("'allow_low_conf_digits' must be a boolean")

    allow_low_conf_pop = cfg.get("allow_low_conf_population")
    if allow_low_conf_pop is not None and not isinstance(allow_low_conf_pop, bool):
        raise RuntimeError("'allow_low_conf_population' must be a boolean")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config.json`` and validate its contents.

    Parameters
    ----------
    path:
        Optional path to the configuration file. When ``None`` the project's
        default ``config.json`` is used. The configuration is cached so repeated
        calls return the same dictionary instance.

    Raises
    ------
    RuntimeError
        If the file is missing, contains invalid JSON or fails validation.
    """

    global _CFG_CACHE
    cfg_path = Path(path) if path is not None else ROOT / "config.json"
    if path is None and _CFG_CACHE is not None:
        return _CFG_CACHE
    try:
        with open(cfg_path, encoding="utf-8") as cfg_file:
            cfg = json.load(cfg_file)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Configuration file not found: {cfg_path}. "
            "Copy config.sample.json and adjust it for your setup."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON in {cfg_path}: {exc}"
        ) from exc

    # Merge profile overrides with base settings
    profiles = cfg.get("profiles", {})
    base_cfg = {k: v for k, v in cfg.items() if k != "profiles"}
    for name, override in profiles.items():
        profiles[name] = _deep_merge(base_cfg, override)
    cfg["profiles"] = profiles

    validate_config(cfg)
    if path is None:
        _CFG_CACHE = cfg
    return cfg


@dataclass
class ScenarioInfo:
    starting_villagers: int = 0
    starting_idle_villagers: int = 0
    population_limit: int = 0
    objective_villagers: int = 0
    # Expected starting resources keyed by resource name used in OCR routines
    starting_resources: dict[str, int] | None = None


def parse_scenario_info(path: str | Path) -> ScenarioInfo:
    """Parse basic scenario information from a text file."""
    info = ScenarioInfo()
    in_objectives = False
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    in_objectives = False
                    continue
                lower = line.lower()
                if lower.startswith("population limit"):
                    m = re.search(r"(\d+)", line)
                    if m:
                        info.population_limit = int(m.group(1))
                elif lower.startswith("starting units"):
                    m = re.search(r"(\d+)\s+villagers", line, re.IGNORECASE)
                    if m:
                        info.starting_villagers = int(m.group(1))
                elif lower.startswith("starting idle villagers"):
                    m = re.search(r"(\d+)", line)
                    if m:
                        info.starting_idle_villagers = int(m.group(1))
                elif lower.startswith("starting resources"):
                    res = {}
                    for name in ("wood", "food", "gold", "stone"):
                        m = re.search(rf"(\d+)\s+{name}", line, re.IGNORECASE)
                        if m:
                            res[f"{name}_stockpile"] = int(m.group(1))
                    info.starting_resources = res or None
                elif lower.startswith("objectives"):
                    in_objectives = True
                    continue
                elif in_objectives:
                    m = re.search(r"(\d+)\s+villagers", line, re.IGNORECASE)
                    if m:
                        info.objective_villagers = int(m.group(1))
                        in_objectives = False
    except FileNotFoundError:
        logger.error("Scenario file not found: %s", path)
    return info
