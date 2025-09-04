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
import os
import time
from pathlib import Path

import script.common as common
from script.common import BotState, STATE
import script.hud as hud
import script.resources.reader as resources
from script.config_utils import parse_scenario_info
import script.input_utils as input_utils
from script.units import villager

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
    if "PYTEST_CURRENT_TEST" in os.environ:
        logger.info(
            "PYTEST_CURRENT_TEST variable detected; run_mission will be skipped."
        )
    if "PYTEST_CURRENT_TEST" not in os.environ:
        run_mission(info, state=state)


def run_mission(info, state: BotState = STATE) -> None:
    """Execute the mission objectives for Foraging.

    Villagers are assigned to construct a Granary, Storage Pit and Dock. After
    each build command the resource cache and population counters are refreshed
    so that subsequent automation works with up-to-date information.
    """

    logger.info("Starting mission objectives")

    build_steps = [
        ("Granary", villager.build_granary, 120),
        ("Storage Pit", villager.build_storage_pit, 120),
        ("Dock", getattr(villager, "build_dock", None), 100),
    ]

    for name, func, _cost in build_steps:
        if func is None:
            logger.warning("No function available to build %s; skipping", name)
            continue

        if not villager.select_idle_villager(state=state):
            logger.info("No idle villager available for %s", name)
            continue

        if not func(state=state):
            logger.info("Failed to build %s", name)
            continue

        # Allow the HUD to update and refresh caches
        time.sleep(0.1)
        try:
            res_vals, (cur_pop, pop_cap) = resources.read_resources_from_hud(
                ["wood_stockpile", "population_limit"]
            )
        except common.ResourceReadError as exc:  # pragma: no cover - defensive
            logger.error("Failed to read resources after building %s: %s", name, exc)
            continue

        wood = res_vals.get("wood_stockpile")
        if isinstance(wood, int):
            resources.RESOURCE_CACHE.last_resource_values["wood_stockpile"] = wood
            resources.RESOURCE_CACHE.last_resource_ts["wood_stockpile"] = time.time()

        if cur_pop is not None:
            state.current_pop = cur_pop
        if pop_cap is not None:
            state.pop_cap = pop_cap

        logger.info("%s construction issued; wood remaining: %s", name, wood)

    logger.info("Mission sequence complete")


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    main()
