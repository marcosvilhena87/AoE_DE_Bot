"""Automation for the *Hunting* scenario of the Ascent of Egypt campaign.

This module orchestrates the very first tutorial mission of Age of Empires
Definitive Edition.  The objective of the scenario is to grow the tribe to
seven villagers.  The script performs the scenario specific setup such as
reading the scenario information from the accompanying text file and
initialising the population counters.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import script.common as common
import script.hud as hud
from script.config_utils import parse_scenario_info

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the automation routine for the *Hunting* mission.

    The function performs the following high level steps:

    1.  Wait for the game HUD to be detected on screen (with a fallback
        attempt if the first detection fails).
    2.  Parse the scenario information from ``1.Hunting.txt`` which lives
        alongside this file.
    3.  Initialise the internal population counters so that the rest of the
        automation knows how many villagers are currently available and what
        the target population is.
    """

    logger.info("Enter the campaign mission (Hunting). The script starts when the HUD is detected…")

    try:
        anchor, asset = hud.wait_hud(timeout=90)
        logger.info("HUD detected at %s using '%s'.", anchor, asset)
    except RuntimeError as exc:  # pragma: no cover - retry branch is defensive
        logger.error(str(exc))
        logger.info("Giving another 25s for you to adjust the camera/HUD (fallback)…")
        time.sleep(25)
        anchor, asset = hud.wait_hud(timeout=90)
        logger.info("HUD detected at %s using '%s'.", anchor, asset)

    scenario_txt = Path(__file__).with_suffix(".txt")
    info = parse_scenario_info(scenario_txt)

    # Configuração inicial de população
    common.CURRENT_POP = info.starting_villagers
    common.POP_CAP = 4  # População suportada pelo Town Center inicial
    common.TARGET_POP = info.objective_villagers

    logger.info("Setup complete.")


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    main()

