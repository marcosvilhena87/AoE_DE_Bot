import logging
import time

import script.common as common
import script.hud as hud
import script.resources as resources
import script.input_utils as input_utils


def select_idle_villager():
    """Selecione um aldeão ocioso usando a tecla configurada."""
    input_utils._press_key_safe(common.CFG["keys"]["idle_vill"], 0.10)


def count_idle_villagers_via_hotkey(delay=0.1, return_selections=False):
    """Conta quantos aldeões ociosos existem usando o atalho de seleção.

    Lê ``idle_villager`` no HUD e pressiona o hotkey de seleção de
    aldeão ocioso repetidamente até que a contagem não diminua mais ou
    chegue a zero. Pequenas pausas são aplicadas entre as leituras para
    permitir que a interface seja atualizada.

    Parameters
    ----------
    delay : float, optional
        Intervalo, em segundos, aguardado entre as leituras do HUD.
        O padrão é ``0.1``.
    return_selections : bool, optional
        Se ``True``, também retorna o número de vezes que o hotkey foi
        acionado.

    Returns
    -------
    int or tuple
        Contagem inicial de aldeões ociosos. Quando ``return_selections``
        é ``True`` retorna ``(contagem_inicial, selecoes)``.
    """

    try:
        res = resources.read_resources_from_hud(["idle_villager"])
    except common.ResourceReadError as exc:  # pragma: no cover - falha de OCR
        logging.error("Falha ao ler idle_villager: %s", exc)
        initial = 0
    else:
        initial = res.get("idle_villager")
        if not isinstance(initial, int):
            initial = 0

    current = initial
    selections = 0
    while isinstance(current, int) and current > 0:
        select_idle_villager()
        selections += 1
        time.sleep(delay)
        try:
            res = resources.read_resources_from_hud(["idle_villager"])
        except common.ResourceReadError as exc:  # pragma: no cover - falha de OCR
            logging.error("Falha ao ler idle_villager: %s", exc)
            break
        new_val = res.get("idle_villager")
        if not isinstance(new_val, int) or new_val >= current:
            break
        current = new_val

    if return_selections:
        return initial, selections
    return initial


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
    for attempt in range(1, 4):
        logging.debug(
            "Attempt %s to read wood from HUD while building house", attempt
        )
        try:
            res_vals = resources.read_resources_from_hud(["wood_stockpile"])
        except common.ResourceReadError as exc:
            logging.error(
                "Resource read error while building house (attempt %s/3): %s",
                attempt,
                exc,
            )
        else:
            wood = res_vals.get("wood_stockpile")
            if isinstance(wood, int):
                break
            logging.warning(
                "wood_stockpile not detected (attempt %s/3); HUD may not be updated",
                attempt,
            )
        if attempt < 3:
            time.sleep(0.2)
    if not isinstance(wood, int):
        logging.debug("Refreshing HUD anchor before final resource read")
        try:
            hud.wait_hud()
            res_vals = resources.read_resources_from_hud(["wood_stockpile"])
        except Exception as exc:
            logging.error(
                "Failed to refresh HUD or read resources while building house: %s",
                exc,
            )
            return False
        wood = res_vals.get("wood_stockpile")
        if not isinstance(wood, int):
            logging.error(
                "Failed to obtain wood stockpile after HUD refresh; cannot build house"
            )
            return False
    if wood < wood_needed:
        logging.warning(
            "Madeira insuficiente (%s) para construir casa.",
            wood,
        )
        return False

    house_key = common.CFG["keys"].get("house")
    if not house_key:
        logging.warning("Tecla de construção de casa não configurada.")
        return False

    areas = common.CFG.get("areas", {})
    main_spot = areas.get("house_spot")
    if not main_spot:
        logging.warning("House spot not configured.")
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
            cur, limit = hud.read_population_from_hud()
        except Exception as exc:  # pragma: no cover - falha de OCR
            logging.warning("Falha ao ler população: %s", exc)
            limit = common.POP_CAP

        if limit > common.POP_CAP:
            common.POP_CAP = limit
            return True

        logging.warning("Tentativa %s de construir casa falhou.", idx)
        try:
            res_vals = resources.read_resources_from_hud(["wood_stockpile"])
        except common.ResourceReadError as exc:
            logging.error(
                "Resource read error while retrying house construction: %s", exc
            )
            return False
        wood = res_vals.get("wood_stockpile")
        if wood is None:
            logging.error("Failed to read wood; aborting house construction")
            return False
        if wood < wood_needed:
            logging.warning(
                "Madeira insuficiente após tentativa (%s).", wood
            )
            break

    return False


def build_granary():
    """Posiciona um Granary no ponto configurado."""
    input_utils._press_key_safe(common.CFG["keys"]["build_menu"], 0.05)
    g_key = common.CFG["keys"].get("granary")
    if not g_key:
        logging.warning("Tecla de construção de Granary não configurada.")
        return False
    areas = common.CFG.get("areas", {})
    spot = areas.get("granary_spot")
    if not spot:
        logging.warning("Granary spot not configured.")
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
        logging.warning("Tecla de construção de Storage Pit não configurada.")
        return False
    areas = common.CFG.get("areas", {})
    spot = areas.get("storage_spot")
    if not spot:
        logging.warning("Storage spot not configured.")
        return False
    input_utils._press_key_safe(s_key, 0.15)
    sx, sy = spot
    input_utils._click_norm(sx, sy)
    return True


def econ_loop(minutes=5):
    """Rotina econômica básica para a missão Hunting."""
    import script.town_center as town_center

    logging.info("Iniciando rotina econômica por %s minutos", minutes)
    town_center.train_villagers(common.TARGET_POP)

    select_idle_villager()
    if build_granary():
        logging.info("Granary posicionado")
    else:
        logging.warning("Falha ao posicionar Granary")
    time.sleep(0.5)
    select_idle_villager()
    if build_storage_pit():
        logging.info("Storage Pit posicionado")
    else:
        logging.warning("Falha ao posicionar Storage Pit")
    time.sleep(0.5)

    areas = common.CFG.get("areas", {})
    food_spot = areas.get("food_spot")
    if not food_spot:
        logging.warning("Food spot not configured.")
        return False
    wood_spot = areas.get("wood_spot")
    if not wood_spot:
        logging.warning("Wood spot not configured.")
        return False
    food_x, food_y = food_spot
    wood_x, wood_y = wood_spot

    t0 = time.time()
    while time.time() - t0 < minutes * 60:
        select_idle_villager()
        input_utils._click_norm(food_x, food_y)
        time.sleep(common.CFG["timers"]["idle_gap"])

        select_idle_villager()
        input_utils._click_norm(wood_x, wood_y)
        time.sleep(common.CFG["timers"]["idle_gap"])

        if common.CURRENT_POP >= common.POP_CAP - 2:
            res_vals = None
            for attempt in range(1, 4):
                logging.debug(
                    "Attempt %s to read resources during economic loop", attempt
                )
                try:
                    res_vals = resources.read_resources_from_hud(["idle_villager"])
                    break
                except common.ResourceReadError as exc:
                    logging.error(
                        "Resource read error during economic loop (attempt %s/3): %s",
                        attempt,
                        exc,
                    )
                    time.sleep(0.1)
            if res_vals is None:
                logging.error(
                    "Failed to read resources after 3 attempts; ending economic loop"
                )
                break
            idle_before = res_vals.get("idle_villager")
            if idle_before is None:
                logging.error("Failed to read idle villager count; ending economic loop")
                break
            if "idle_villager" in resources._LAST_READ_FROM_CACHE:
                logging.warning(
                    "Using cached idle villager count; confidence degraded"
                )
            took_from_wood = False

            if idle_before > 0:
                select_idle_villager()
                time.sleep(0.1)
                idle_after_res = None
                for attempt in range(1, 4):
                    logging.debug(
                        "Attempt %s to read idle villager count during economic loop",
                        attempt,
                    )
                    try:
                        idle_after_res = resources.read_resources_from_hud(["idle_villager"])
                        break
                    except common.ResourceReadError as exc:
                        logging.error(
                            "Resource read error when checking idle villagers (attempt %s/3): %s",
                            attempt,
                            exc,
                        )
                        time.sleep(0.1)
                if idle_after_res is None:
                    idle_after_res = {}
                idle_after = idle_after_res.get("idle_villager") if idle_after_res else None
                if idle_after is None:
                    logging.error("Failed to read idle villager count after selection")
                    idle_after = 0
                elif "idle_villager" in resources._LAST_READ_FROM_CACHE:
                    logging.warning(
                        "Using cached idle villager count; confidence degraded"
                    )
                if idle_after >= idle_before:
                    input_utils._click_norm(wood_x, wood_y)
                    took_from_wood = True
            else:
                input_utils._click_norm(wood_x, wood_y)
                took_from_wood = True

            if build_house():
                logging.info("Casa construída para expandir população")
            else:
                logging.warning("Falha ao construir casa para expandir população")

            if took_from_wood:
                input_utils._click_norm(wood_x, wood_y)

            time.sleep(0.5)

        time.sleep(common.CFG["timers"]["loop_sleep"])
    logging.info("Rotina econômica finalizada")
