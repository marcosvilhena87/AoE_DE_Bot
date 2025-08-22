import logging
import time

import script.common as common
from script.villager import build_house, select_idle_villager


def train_villagers(target_pop: int):
    """Fila aldeões na Town Center até atingir ``target_pop``."""
    common._press_key_safe(common.CFG["keys"]["select_tc"], 0.10)

    while common.CURRENT_POP < target_pop:
        resources = None
        for attempt in range(1, 4):
            logging.debug(
                "Attempt %s to read resources while training villagers", attempt
            )
            try:
                resources = common.read_resources_from_hud()
                break
            except common.ResourceReadError as exc:
                logging.error(
                    "Resource read error while training villagers (attempt %s/3): %s",
                    attempt,
                    exc,
                )
                time.sleep(0.1)
        if resources is None:
            logging.error("Failed to read resources after 3 attempts; stopping villager training")
            break
        food = resources.get("food_stockpile")
        if food is None:
            logging.error("Failed to read food; stopping villager training")
            break
        if food < 50:
            logging.info(
                "Comida insuficiente (%s) para treinar aldeões.",
                food,
            )
            break
        common._press_key_safe(common.CFG["keys"]["train_vill"], 0.0)
        common.CURRENT_POP += 1
        if common.CURRENT_POP == common.POP_CAP:
            select_idle_villager()
            if build_house():
                logging.info("Casa construída para expandir população")
            else:
                logging.warning("Falha ao construir casa para expandir população")
        time.sleep(0.10)
