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

        if state.current_pop >= state.pop_cap:
            # Need to increase population cap before training more villagers
            deadline = time.time() + 30.0
            built_house = False
            while time.time() < deadline:
                if select_idle_villager(state=state):
                    if build_house(state=state):
                        built_house = True
                        break
                    # build_house logs success or failure internally
                else:
                    logger.warning("No idle villager to build house")
                time.sleep(1.0)
            if not built_house:
                logger.error(
                    "Unable to build house; stopping villager training"
                )
                break
            # Reselect the Town Center after building a house so that
            # further villager training continues from the correct building.
            input_utils._press_key_safe(
                state.config["keys"]["select_tc"], 0.10
            )
            # After a successful house build, loop again to verify the new
            # population cap before training.
            continue

        input_utils._press_key_safe(state.config["keys"]["train_vill"], 0.0)
        state.current_pop += 1
        time.sleep(0.10)
