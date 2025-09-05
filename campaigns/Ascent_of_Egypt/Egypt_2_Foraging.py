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
from script.common import BotState, STATE
import script.hud as hud
import script.resources.reader as resources
from script.config_utils import parse_scenario_info
import script.input_utils as input_utils
from script.units import villager
from script.buildings.town_center import train_villagers

logger = logging.getLogger(__name__)


def main(config_path: str | Path | None = None, state: BotState = STATE) -> None:
    """Run the automation routine for the *Foraging* mission.

    The function performs the following high level steps:

    1.  Wait for the game HUD to be detected on screen (with a fallback
        attempt if the first detection fails).
    2.  Parse the scenario information from ``Egypt_2_Foraging.txt`` which
        lives alongside this file.
    3.  Initialise the internal resource and population counters so that
        the rest of the automation knows the correct starting state.
    """

    common.init_common(config_path, state)
    logger.info(
        "Enter the campaign mission (Foraging). The script starts when the HUD is detected…"
    )
    input_utils.configure_pyautogui()

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
    if not info.starting_buildings or info.starting_buildings.get("Town Center", 0) < 1:
        logger.error("Required starting building 'Town Center' not found; aborting scenario.")
        return

    # Leia o HUD para obter os valores atuais de recursos e população
    res_vals, (cur_pop, pop_cap) = resources.gather_hud_stats()

    if cur_pop != info.starting_villagers or pop_cap != info.population_limit:
        logger.error(
            "HUD population (%s/%s) does not match expected %s/%s; aborting scenario.",
            cur_pop,
            pop_cap,
            info.starting_villagers,
            info.population_limit,
        )
        return

    # Validate initial resources
    try:
        resources.validate_starting_resources(
            res_vals, info.starting_resources, raise_on_error=True
        )
    except resources.ResourceValidationError as exc:
        logger.error("Error validating starting resources: %s", exc)
        raise

    hud_idle = res_vals.get("idle_villager")
    if hud_idle != info.starting_idle_villagers:
        logger.error(
            "HUD idle villager count (%s) does not match expected %s; aborting scenario.",
            hud_idle,
            info.starting_idle_villagers,
        )
        return

    # Atualize os caches com os valores confirmados
    now = time.time()
    resources.RESOURCE_CACHE.last_resource_values.update(res_vals)
    for name in res_vals:
        resources.RESOURCE_CACHE.last_resource_ts[name] = now
    resources.RESOURCE_CACHE.last_resource_values["idle_villager"] = hud_idle
    resources.RESOURCE_CACHE.last_resource_ts["idle_villager"] = now

    # Atualize população e limites
    state.current_pop = cur_pop
    state.pop_cap = pop_cap
    state.target_pop = info.objective_villagers

    logger.info("Setup complete.")

    run_mission(info, state=state)


def run_mission(info, state: BotState = STATE) -> None:
    """Execute the mission objectives for the *Foraging* scenario."""

    logger.info("Starting mission objectives")

    food_spot = state.config.get("areas", {}).get("food_spot")
    for idx in range(info.starting_idle_villagers):
        if villager.select_idle_villager(state=state):
            if food_spot:
                input_utils._click_norm(*food_spot, button="right")
            logger.info("Villager %d assigned to gather food", idx + 1)
        else:
            logger.info(
                "No idle villager available when assigning villager %d", idx + 1
            )

    villager.build_granary(state=state)
    villager.build_storage_pit(state=state)
    villager.build_dock(state=state)

    loops = 0
    max_loops = state.config.get("max_mission_loops", 20)
    while state.current_pop < state.target_pop and loops < max_loops:
        loops += 1
        train_villagers(state.current_pop + 1, state=state)
        time.sleep(0.1)

    logger.info(
        "Mission loop ended after %s iterations. Population: %s/%s",
        loops,
        state.current_pop,
        state.target_pop,
    )


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    main()
