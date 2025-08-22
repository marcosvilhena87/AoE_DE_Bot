import logging
import time

import script.common as common
from script.villager import build_house, select_idle_villager


def train_villagers(target_pop: int):
    """Fila aldeões na Town Center até atingir ``target_pop``."""
    common._press_key_safe(common.CFG["keys"]["select_tc"], 0.10)

    while common.CURRENT_POP < target_pop:
        resources = common.read_resources_from_hud()
        if resources.get("food", 0) < 50:
            logging.info(
                "Comida insuficiente (%s) para treinar aldeões.",
                resources.get("food", 0),
            )
            break
        common._press_key_safe(common.CFG["keys"]["train_vill"], 0.0)
        common.CURRENT_POP += 1
        if common.CURRENT_POP == common.POP_CAP:
            select_idle_villager()
            build_house()
            common.POP_CAP += 4
        time.sleep(0.10)
