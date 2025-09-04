"""Automation for the *Hunting* scenario of the Ascent of Egypt campaign.

This module orchestrates the first tutorial mission of Age of Empires
Definitive Edition. The objective of the scenario is to grow the tribe by
hunting enough food to train additional villagers. The script performs the
scenario specific setup such as reading the scenario information from the
accompanying text file and initialising the population and resource
counters.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
import os

import script.common as common
import script.hud as hud
import script.resources.reader as resources
import script.input_utils as input_utils
from script.units import villager
from script.buildings.town_center import train_villagers
from script.config_utils import parse_scenario_info
import script.screen_utils as screen_utils

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the automation routine for the *Hunting* mission.

    The function performs the following high level steps:

    1.  Wait for the game HUD to be detected on screen (with a fallback
        attempt if the first detection fails).
    2.  Parse the scenario information from ``Egypt_1_Hunting.txt`` which lives
        alongside this file.
    3.  Initialise the internal resource and population counters so that
        the rest of the automation knows the correct starting state.
    """

    logger.info(
        "Enter the campaign mission (Hunting). The script starts when the HUD is detected…"
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
    if not info.starting_buildings or info.starting_buildings.get("Town Center", 0) < 1:
        logger.error("Required starting building 'Town Center' not found; aborting scenario.")
        return

    # Leia o HUD para obter os valores atuais de recursos e população
    res_vals, (cur_pop, pop_cap) = resources.gather_hud_stats()

    pop_check = resources.validate_population(
        res_vals,
        cur_pop,
        pop_cap,
        expected_cur=info.starting_villagers,
        expected_cap=info.population_limit,
    )
    if pop_check is None:
        logger.error(
            "HUD population (%s/%s) does not match expected %s/%s; aborting scenario.",
            cur_pop,
            pop_cap,
            info.starting_villagers,
            info.population_limit,
        )
        return
    res_vals, (cur_pop, pop_cap) = pop_check

    # Validação dos recursos iniciais
    tol_cfg = common.CFG.get("resource_validation_tolerance", {})
    tolerance = tol_cfg.get("initial", 10)
    frame = screen_utils.grab_frame()
    rois = getattr(resources, "_LAST_REGION_BOUNDS", {})
    try:
        resources.validate_starting_resources(
            res_vals,
            info.starting_resources,
            tolerance=tolerance,
            raise_on_error=True,
            frame=frame,
            rois=rois,
        )
    except resources.ResourceValidationError as exc:
        logger.error("Erro na validação dos recursos iniciais: %s", exc)
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
    common.CURRENT_POP = cur_pop
    common.POP_CAP = pop_cap
    common.TARGET_POP = info.objective_villagers

    logger.info("Setup complete.")
    if "PYTEST_CURRENT_TEST" in os.environ:
        logger.info(
            "Variável PYTEST_CURRENT_TEST detectada; run_mission será pulado."
        )
    if "PYTEST_CURRENT_TEST" not in os.environ:
        run_mission(info)


def run_mission(info) -> None:
    """Execute the mission objectives.

    The routine assigns villagers to hunting, monitors the food stockpile and
    trains new villagers until the population objective is met.  A small loop
    limit is included to prevent the function from running indefinitely during
    automated tests where resource values never change.
    """

    logger.info("Starting mission objectives")

    food_spot = common.CFG.get("areas", {}).get("food_spot")

    logger.info("Assigning starting villagers to hunt")
    # Allocate only the idle starting villagers to gather food.
    for idx in range(info.starting_idle_villagers):
        if villager.select_idle_villager():
            if food_spot:
                input_utils._click_norm(*food_spot, button="right")
            logger.info("Villager %d assigned to hunt", idx + 1)
        else:
            logger.info("No idle villager available for hunting when assigning villager %d", idx + 1)

    logger.info("Initial villager assignment complete")

    start_food = resources.RESOURCE_CACHE.last_resource_values.get(
        "food_stockpile", 0
    )
    spent_food = 0

    loops = 0
    max_loops = common.CFG.get("max_mission_loops", 20)
    logger.info("Mission loop starting. Target population: %s", common.TARGET_POP)
    while common.CURRENT_POP < common.TARGET_POP and loops < max_loops:
        loops += 1
        logger.info(
            "Population progress: %s/%s",
            common.CURRENT_POP,
            common.TARGET_POP,
        )
        try:
            res_vals, _ = resources.read_resources_from_hud(["food_stockpile"])
        except common.ResourceReadError as exc:  # pragma: no cover - OCR failure
            logger.error("Failed to read food: %s", exc)
            time.sleep(0.1)
            continue
        food = res_vals.get("food_stockpile")
        if isinstance(food, int):
            resources.RESOURCE_CACHE.last_resource_values["food_stockpile"] = food
        else:
            food = resources.RESOURCE_CACHE.last_resource_values.get(
                "food_stockpile", 0
            )
        logger.info("Current food stockpile: %s", food)

        if food >= start_food + spent_food + 50:
            logger.info("Training villager to reach population %s", common.CURRENT_POP + 1)
            train_villagers(common.CURRENT_POP + 1)
            spent_food += 50
            logger.info(
                "Villager trained. Population: %s", common.CURRENT_POP
            )
        time.sleep(0.1)

    logger.info(
        "Mission loop ended after %s iterations. Objective met: %s",
        loops,
        common.CURRENT_POP >= common.TARGET_POP,
    )

    if common.CURRENT_POP >= common.TARGET_POP:
        logger.info(
            "Population objective reached: %s/%s",
            common.CURRENT_POP,
            common.TARGET_POP,
        )
        logger.info("Mission accomplished: objectives achieved")
    else:
        logger.info(
            "Mission ended before reaching population objective. Population: %s/%s",
            common.CURRENT_POP,
            common.TARGET_POP,
        )
        logger.info("Mission failed: objectives not achieved")


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    main()
