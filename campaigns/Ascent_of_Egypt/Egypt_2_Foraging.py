"""Automation for the *Foraging* scenario of the Ascent of Egypt campaign.

This module orchestrates the second tutorial mission of Age of Empires
Definitive Edition. The objective of the scenario is to collect food and
construct a Granary, Storage Pit and Dock. The script performs the
scenario specific setup such as reading the scenario information from the
accompanying text file and initialising the population and resource
counters.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import script.common as common
import script.hud as hud
import script.resources.reader as resources
from script.config_utils import parse_scenario_info

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the automation routine for the *Foraging* mission.

    The function performs the following high level steps:

    1.  Wait for the game HUD to be detected on screen (with a fallback
        attempt if the first detection fails).
    2.  Parse the scenario information from ``Egypt_2_Foraging.txt`` which
        lives alongside this file.
    3.  Initialise the internal resource and population counters so that
        the rest of the automation knows the correct starting state.
    """

    logger.info(
        "Enter the campaign mission (Foraging). The script starts when the HUD is detected…"
    )

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

    # Leia o HUD para obter os valores atuais de recursos e população
    res_vals, (cur_pop, pop_cap) = resources.gather_hud_stats()

    # Validação dos recursos iniciais
    try:
        resources.validate_starting_resources(
            res_vals, info.starting_resources, raise_on_error=True
        )
    except resources.ResourceValidationError as exc:
        logger.error("Erro na validação dos recursos iniciais: %s", exc)
        raise

    # Atualize os caches com os valores confirmados
    now = time.time()
    resources.RESOURCE_CACHE.last_resource_values.update(res_vals)
    for name in res_vals:
        resources.RESOURCE_CACHE.last_resource_ts[name] = now
    resources.RESOURCE_CACHE.last_resource_values["idle_villager"] = (
        info.starting_idle_villagers
    )
    resources.RESOURCE_CACHE.last_resource_ts["idle_villager"] = now

    # Atualize população e limites
    common.CURRENT_POP = cur_pop if cur_pop is not None else info.starting_villagers
    common.POP_CAP = pop_cap if pop_cap is not None else 4
    common.TARGET_POP = info.objective_villagers

    logger.info("Setup complete.")


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    main()
