import logging
import time

from script.common import BotState, STATE
import script.input_utils as input_utils

logger = logging.getLogger(__name__)


def select_idle_villager(delay: float = 0.1, state: BotState = STATE) -> bool:
    """Pressiona imediatamente a tecla configurada para selecionar um aldeão ocioso."""

    key = state.config["keys"].get("idle_vill")
    if not key:
        return False
    input_utils._press_key_safe(key, delay)
    return True


def build_house(state: BotState = STATE):
    """Constrói uma casa no local predefinido.

    Verifica se há madeira suficiente antes de tentar construir e confirma
    que a casa foi posicionada checando o aumento do ``POP_CAP``. Caso a
    construção falhe, tenta novamente em um ponto alternativo (se
    configurado) ou após reunir mais recursos.

    Returns
    -------
    bool
        ``True`` se a casa foi construída com sucesso.
    """

    house_key = state.config["keys"].get("house")
    if not house_key:
        logger.warning("House build key not configured.")
        return False

    areas = state.config.get("areas", {})
    main_spot = areas.get("house_spot")
    if not main_spot:
        logger.warning("House spot not configured.")
        return False
    spots = [main_spot]
    alt_spot = areas.get("house_spot_alt")
    if alt_spot:
        spots.append(alt_spot)

    for hx, hy in spots:
        input_utils._press_key_safe(state.config["keys"]["build_menu"], 0.05)
        input_utils._press_key_safe(house_key, 0.15)
        input_utils._click_norm(hx, hy)
        input_utils._click_norm(hx, hy, button="right")
        time.sleep(0.5)

        state.pop_cap += 4
        return True

    return False


def build_granary(state: BotState = STATE):
    """Posiciona um Granary no ponto configurado."""
    input_utils._press_key_safe(state.config["keys"]["build_menu"], 0.05)
    g_key = state.config["keys"].get("granary")
    if not g_key:
        logger.warning("Granary build key not configured.")
        return False
    areas = state.config.get("areas", {})
    spot = areas.get("granary_spot")
    if not spot:
        logger.warning("Granary spot not configured.")
        return False
    input_utils._press_key_safe(g_key, 0.15)
    gx, gy = spot
    input_utils._click_norm(gx, gy)
    return True


def build_storage_pit(state: BotState = STATE):
    """Posiciona um Storage Pit no ponto configurado."""
    input_utils._press_key_safe(state.config["keys"]["build_menu"], 0.05)
    s_key = state.config["keys"].get("storage_pit")
    if not s_key:
        logger.warning("Storage Pit build key not configured.")
        return False
    areas = state.config.get("areas", {})
    spot = areas.get("storage_spot")
    if not spot:
        logger.warning("Storage spot not configured.")
        return False
    input_utils._press_key_safe(s_key, 0.15)
    sx, sy = spot
    input_utils._click_norm(sx, sy)
    return True
