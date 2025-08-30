import logging
import time

import script.common as common
import script.resources.reader as resources
import script.input_utils as input_utils
from script.units.villager import build_house, select_idle_villager

logger = logging.getLogger(__name__)


def train_villagers(target_pop: int):
    """Fila aldeões na Town Center até atingir ``target_pop``."""
    input_utils._press_key_safe(common.CFG["keys"]["select_tc"], 0.10)

    while common.CURRENT_POP < target_pop:
        # Always reselect the Town Center to ensure subsequent commands
        # affect the correct building, especially after actions that may
        # change the selection (e.g. building houses).
        input_utils._press_key_safe(common.CFG["keys"]["select_tc"], 0.10)

        res_vals = None
        food = None
        for attempt in range(1, 2):
            logger.debug(
                "Attempt %s to read food from HUD while training villagers", attempt
            )
            try:
                res_vals, _ = resources.read_resources_from_hud(["food_stockpile"])
            except common.ResourceReadError as exc:
                logger.error(
                    "Resource read error while training villagers (attempt %s): %s",
                    attempt,
                    exc,
                )
            else:
                food = res_vals.get("food_stockpile")
                if isinstance(food, int):
                    break
                logger.warning(
                    "food_stockpile not detected (attempt %s); HUD may not be updated",
                    attempt,
                )
            time.sleep(0.2)
        if not isinstance(food, int):
            logger.error(
                "Failed to obtain food stockpile after 1 attempt; stopping villager training"
            )
            break
        if food < 50:
            logger.info(
                "Insufficient food (%s) to train villagers.",
                food,
            )
            break
        input_utils._press_key_safe(common.CFG["keys"]["train_vill"], 0.0)
        common.CURRENT_POP += 1
        if common.CURRENT_POP == common.POP_CAP:
            if select_idle_villager():
                if build_house():
                    logger.info("House built to increase population")
                else:
                    logger.warning("Failed to build house to increase population")
            else:
                logger.warning("No idle villager to build house")
            # Reselect the Town Center after attempting to build a house so
            # that further villager training continues from the correct
            # building.
            input_utils._press_key_safe(common.CFG["keys"]["select_tc"], 0.10)
        time.sleep(0.10)
