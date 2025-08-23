"""Automation for the *Hunting* scenario of the Ascent of Egypt campaign.

This module orchestrates the very first tutorial mission of Age of Empires
Definitive Edition.  The objective of the scenario is to grow the tribe to
seven villagers.  The script uses the generic economic routines provided by
``script.villager`` and ``script.town_center`` to train new villagers, place
basic buildings and assign villagers to gather resources.

The heavy lifting (resource gathering, villager training and house building)
is handled by :func:`script.villager.econ_loop`.  This file only performs the
scenario specific setup such as reading the scenario information from the
accompanying text file and initialising the population counters.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import script.common as common
import script.hud as hud
from script.villager import econ_loop
from script.config_utils import parse_scenario_info


def main() -> None:
    """Run the automation routine for the *Hunting* mission.

    The function performs the following high level steps:

    1.  Wait for the game HUD to be detected on screen (with a fallback
        attempt if the first detection fails).
    2.  Parse the scenario information from ``1.Hunting.txt`` which lives
        alongside this file.
    3.  Initialise the internal population counters so that the economic loop
        knows how many villagers are currently available and what the target
        population is.
    4.  Start the economic loop for the number of minutes configured in
        ``config.json``.
    """

    logging.info("Entre na missão da campanha (Hunting). O script inicia quando detectar a HUD…")

    try:
        anchor, asset = hud.wait_hud(timeout=90)
        logging.info("HUD detectada em %s usando '%s'. Rodando rotina econômica…", anchor, asset)
    except RuntimeError as exc:  # pragma: no cover - retry branch is defensive
        logging.error(str(exc))
        logging.info("Dando mais 25s para você ajustar a câmera/HUD (fallback)…")
        time.sleep(25)
        anchor, asset = hud.wait_hud(timeout=90)
        logging.info("HUD detectada em %s usando '%s'. Rodando rotina econômica…", anchor, asset)

    scenario_txt = Path(__file__).with_suffix(".txt")
    info = parse_scenario_info(scenario_txt)

    # Configuração inicial de população
    common.CURRENT_POP = info.starting_villagers
    common.POP_CAP = 4  # População suportada pelo Town Center inicial
    common.TARGET_POP = info.objective_villagers

    econ_loop(minutes=common.CFG["loop_minutes"])
    logging.info("Rotina concluída.")


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    main()

