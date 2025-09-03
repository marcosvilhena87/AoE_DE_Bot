import logging
import time

import script.common as common
import script.hud as hud
import script.resources.reader as resources
import script.input_utils as input_utils

logger = logging.getLogger(__name__)

# Último valor válido de aldeões ociosos lido do HUD
_last_idle_villager_count = 0


def select_idle_villager(delay: float = 0.1) -> bool:
    """Tenta selecionar um aldeão ocioso usando a tecla configurada.

    Lê o valor de ``idle_villager`` e o compara com a população atual. Se o
    valor de ``idle_villager`` exceder a população detectada, considera a
    leitura inválida e tenta novamente ou recorre ao último valor conhecido.
    """

    global _last_idle_villager_count

    for attempt in range(2):
        try:
            res_vals, _ = resources.read_resources_from_hud(["idle_villager"])
        except common.ResourceReadError as exc:  # pragma: no cover - falha de OCR
            logger.error("Failed to read idle_villager: %s", exc)
            return False

        idle_vill = res_vals.get("idle_villager")
        if not isinstance(idle_vill, int):
            return False

        cur_pop = common.CURRENT_POP
        try:
            cur_pop, _, low_conf = hud.read_population_from_hud()
            if low_conf:
                _last_idle_villager_count = 0
                continue
            common.CURRENT_POP = cur_pop
        except common.PopulationReadError as exc:
            logger.debug("Population read failed: %s", exc)
            _last_idle_villager_count = 0
            cur_pop = common.CURRENT_POP
            continue
        except common.ResourceReadError as exc:
            logger.debug("Population read failed: %s", exc)
            cur_pop = common.CURRENT_POP

        if (
            (cur_pop and idle_vill > cur_pop)
            or (common.CURRENT_POP and idle_vill > common.CURRENT_POP)
        ):
            logger.warning(
                "Idle villager count %s exceeds population %s; retrying",
                idle_vill,
                cur_pop,
            )
            time.sleep(0.05)
            continue

        _last_idle_villager_count = idle_vill
        if idle_vill > 0:
            input_utils._press_key_safe(common.CFG["keys"]["idle_vill"], delay)
            return True
        return False

    # Após tentativas sem leitura válida, usa o último valor conhecido
    if _last_idle_villager_count > 0:
        input_utils._press_key_safe(common.CFG["keys"]["idle_vill"], delay)
        return True
    return False


def build_house():
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

    wood_needed = 30
    res_vals = None
    wood = None
    for attempt in range(1, 2):
        logger.debug(
            "Attempt %s to read wood from HUD while building house", attempt
        )
        try:
            res_vals, _ = resources.read_resources_from_hud(["wood_stockpile"])
        except common.ResourceReadError as exc:
            logger.error(
                "Resource read error while building house (attempt %s): %s",
                attempt,
                exc,
            )
        else:
            wood = res_vals.get("wood_stockpile")
            if isinstance(wood, int):
                break
            logger.warning(
                "wood_stockpile not detected (attempt %s); HUD may not be updated",
                attempt,
            )
        time.sleep(0.2)
    if not isinstance(wood, int):
        logger.debug("Refreshing HUD anchor before final resource read")
        try:
            hud.wait_hud()
            res_vals, _ = resources.read_resources_from_hud(["wood_stockpile"])
        except common.ResourceReadError as exc:
            logger.error(
                "Failed to refresh HUD or read resources while building house: %s",
                exc,
            )
            return False
        wood = res_vals.get("wood_stockpile")
        if not isinstance(wood, int):
            logger.error(
                "Failed to obtain wood stockpile after HUD refresh; cannot build house"
            )
            return False
    if wood < wood_needed:
        logger.warning(
            "Insufficient wood (%s) to build house.",
            wood,
        )
        return False

    house_key = common.CFG["keys"].get("house")
    if not house_key:
        logger.warning("House build key not configured.")
        return False

    areas = common.CFG.get("areas", {})
    main_spot = areas.get("house_spot")
    if not main_spot:
        logger.warning("House spot not configured.")
        return False
    spots = [main_spot]
    alt_spot = areas.get("house_spot_alt")
    if alt_spot:
        spots.append(alt_spot)

    for idx, (hx, hy) in enumerate(spots, start=1):
        input_utils._press_key_safe(common.CFG["keys"]["build_menu"], 0.05)
        input_utils._press_key_safe(house_key, 0.15)
        input_utils._click_norm(hx, hy)
        input_utils._click_norm(hx, hy, button="right")
        time.sleep(0.5)

        try:
            cur, limit, _ = hud.read_population_from_hud()
        except (common.ResourceReadError, common.PopulationReadError) as exc:  # pragma: no cover - falha de OCR
            logger.warning("Failed to read population: %s", exc)
            limit = common.POP_CAP

        if limit > common.POP_CAP:
            common.POP_CAP = limit
            return True

        logger.warning("Attempt %s to build house failed.", idx)
        try:
            res_vals, _ = resources.read_resources_from_hud(["wood_stockpile"])
        except common.ResourceReadError as exc:
            logger.error(
                "Resource read error while retrying house construction: %s", exc
            )
            return False
        wood = res_vals.get("wood_stockpile")
        if wood is None:
            logger.error("Failed to read wood; aborting house construction")
            return False
        if wood < wood_needed:
            logger.warning(
                "Insufficient wood after attempt (%s).", wood
            )
            break

    return False


def build_granary():
    """Posiciona um Granary no ponto configurado."""
    input_utils._press_key_safe(common.CFG["keys"]["build_menu"], 0.05)
    g_key = common.CFG["keys"].get("granary")
    if not g_key:
        logger.warning("Granary build key not configured.")
        return False
    areas = common.CFG.get("areas", {})
    spot = areas.get("granary_spot")
    if not spot:
        logger.warning("Granary spot not configured.")
        return False
    input_utils._press_key_safe(g_key, 0.15)
    gx, gy = spot
    input_utils._click_norm(gx, gy)
    return True


def build_storage_pit():
    """Posiciona um Storage Pit no ponto configurado."""
    input_utils._press_key_safe(common.CFG["keys"]["build_menu"], 0.05)
    s_key = common.CFG["keys"].get("storage_pit")
    if not s_key:
        logger.warning("Storage Pit build key not configured.")
        return False
    areas = common.CFG.get("areas", {})
    spot = areas.get("storage_spot")
    if not spot:
        logger.warning("Storage spot not configured.")
        return False
    input_utils._press_key_safe(s_key, 0.15)
    sx, sy = spot
    input_utils._click_norm(sx, sy)
    return True
