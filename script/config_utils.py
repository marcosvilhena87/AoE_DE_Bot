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

ROOT = Path(__file__).resolve().parent.parent


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


with open(ROOT / "config.json", encoding="utf-8") as cfg_file:
    CFG = json.load(cfg_file)

validate_config(CFG)


@dataclass
class ScenarioInfo:
    starting_villagers: int = 0
    population_limit: int = 0
    objective_villagers: int = 0


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
                elif lower.startswith("objectives"):
                    in_objectives = True
                    continue
                elif in_objectives:
                    m = re.search(r"(\d+)\s+villagers", line, re.IGNORECASE)
                    if m:
                        info.objective_villagers = int(m.group(1))
                        in_objectives = False
    except FileNotFoundError:
        logging.error("Scenario file not found: %s", path)
    return info
