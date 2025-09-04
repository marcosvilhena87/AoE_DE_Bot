import logging
import time

import script.common as common
from script.common import BotState, STATE
import script.input_utils as input_utils
from script.units.villager import build_house, select_idle_villager

logger = logging.getLogger(__name__)


def train_villagers(target_pop: int, state: BotState = STATE):
    """Fila aldeões na Town Center até atingir ``target_pop``."""
    input_utils._press_key_safe(state.config["keys"]["select_tc"], 0.10)

    while state.current_pop < target_pop:
        # Always reselect the Town Center to ensure subsequent commands
        # affect the correct building, especially after actions that may
        # change the selection (e.g. building houses).
        input_utils._press_key_safe(state.config["keys"]["select_tc"], 0.10)

        input_utils._press_key_safe(state.config["keys"]["train_vill"], 0.0)
        state.current_pop += 1
        if state.current_pop == state.pop_cap:
            if select_idle_villager(state=state):
                if build_house(state=state):
                    logger.info("House built to increase population")
                else:
                    logger.warning("Failed to build house to increase population")
            else:
                logger.warning("No idle villager to build house")
            # Reselect the Town Center after attempting to build a house so
            # that further villager training continues from the correct
            # building.
            input_utils._press_key_safe(state.config["keys"]["select_tc"], 0.10)
        time.sleep(0.10)
