import logging
import time

import script.common as common
from script.villager import build_house, select_idle_villager


def train_villagers(target_pop: int):
    """Fila aldeões na Town Center até atingir ``target_pop``."""
    common._press_key_safe(common.CFG["keys"]["select_tc"], 0.10)

    while common.CURRENT_POP < target_pop:
        try:
            resources = common.read_resources_from_hud()
        except common.ResourceReadError as exc:
            logging.error(
                "Resource bar not located; stopping villager training: %s", exc
            )
            break
        food = resources.get("food")
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
