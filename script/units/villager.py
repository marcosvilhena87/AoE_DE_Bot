import logging
import time

import script.common as common
import script.hud as hud
import script.resources as resources
import script.input_utils as input_utils

logger = logging.getLogger(__name__)


def select_idle_villager(delay: float = 0.1) -> bool:
    """Tenta selecionar um aldeão ocioso usando a tecla configurada.

    Lê o valor de ``idle_villager`` antes e depois de pressionar o hotkey.
    O parâmetro ``delay`` é repassado para ``force_delay`` na segunda leitura
    para dar tempo da interface atualizar. Retorna ``True`` se a contagem
    diminuir (um aldeão foi selecionado) ou ``False`` caso contrário.
    """

    before = None
    try:
        res_before, _ = resources.read_resources_from_hud(["idle_villager"])
    except common.ResourceReadError as exc:  # pragma: no cover - falha de OCR
        logger.error("Falha ao ler idle_villager: %s", exc)
    else:
        before = res_before.get("idle_villager")

    input_utils._press_key_safe(common.CFG["keys"]["idle_vill"], 0.10)

    after = None
    try:
        res_after, _ = resources.read_resources_from_hud(
            ["idle_villager"], force_delay=delay
        )
    except common.ResourceReadError as exc:  # pragma: no cover - falha de OCR
        logger.error("Falha ao ler idle_villager: %s", exc)
    else:
        after = res_after.get("idle_villager")

    if isinstance(before, int) and isinstance(after, int) and after < before:
        return True
    return False


def count_idle_villagers_via_hotkey(
    delay=0.1, stop_threshold: int | None = None, return_selections=False
):
    """Conta quantos aldeões ociosos existem usando o atalho de seleção.

    Lê ``idle_villager`` no HUD e pressiona o hotkey de seleção de
    aldeão ocioso repetidamente até que a contagem não diminua mais,
    chegue a zero ou atinja ``stop_threshold``. Pequenas pausas são
    aplicadas entre as leituras para permitir que a interface seja
    atualizada.

    Parameters
    ----------
    delay : float, optional
        Intervalo, em segundos, aguardado entre as leituras do HUD.
        O padrão é ``0.1``.
    stop_threshold : int or None, optional
        Interrompe a contagem quando o valor lido for menor ou igual a
        este limite. O padrão é ``None`` (não interrompe).
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
        res, _ = resources.read_resources_from_hud(["idle_villager"])
    except common.ResourceReadError as exc:  # pragma: no cover - falha de OCR
        logger.error("Falha ao ler idle_villager: %s", exc)
        initial = 0
    else:
        initial = res.get("idle_villager")
        if not isinstance(initial, int):
            initial = 0

    current = initial
    selections = 0
    threshold = stop_threshold if isinstance(stop_threshold, int) else 0
    while isinstance(current, int) and current > threshold:
        select_idle_villager()
        selections += 1
        try:
            res, _ = resources.read_resources_from_hud(
                ["idle_villager"], force_delay=delay
            )
        except common.ResourceReadError as exc:  # pragma: no cover - falha de OCR
            logger.error("Falha ao ler idle_villager: %s", exc)
            break
        new_val = res.get("idle_villager")
        if (
            not isinstance(new_val, int)
            or new_val >= current
            or new_val <= threshold
        ):
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
        logger.debug(
            "Attempt %s to read wood from HUD while building house", attempt
        )
        try:
            res_vals, _ = resources.read_resources_from_hud(["wood_stockpile"])
        except common.ResourceReadError as exc:
            logger.error(
                "Resource read error while building house (attempt %s/3): %s",
                attempt,
                exc,
            )
        else:
            wood = res_vals.get("wood_stockpile")
            if isinstance(wood, int):
                break
            logger.warning(
                "wood_stockpile not detected (attempt %s/3); HUD may not be updated",
                attempt,
            )
        if attempt < 3:
            time.sleep(0.2)
    if not isinstance(wood, int):
        logger.debug("Refreshing HUD anchor before final resource read")
        try:
            hud.wait_hud()
            res_vals, _ = resources.read_resources_from_hud(["wood_stockpile"])
        except Exception as exc:
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
            "Madeira insuficiente (%s) para construir casa.",
            wood,
        )
        return False

    house_key = common.CFG["keys"].get("house")
    if not house_key:
        logger.warning("Tecla de construção de casa não configurada.")
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
            cur, limit = hud.read_population_from_hud()
        except Exception as exc:  # pragma: no cover - falha de OCR
            logger.warning("Falha ao ler população: %s", exc)
            limit = common.POP_CAP

        if limit > common.POP_CAP:
            common.POP_CAP = limit
            return True

        logger.warning("Tentativa %s de construir casa falhou.", idx)
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
                "Madeira insuficiente após tentativa (%s).", wood
            )
            break

    return False


def build_granary():
    """Posiciona um Granary no ponto configurado."""
    input_utils._press_key_safe(common.CFG["keys"]["build_menu"], 0.05)
    g_key = common.CFG["keys"].get("granary")
    if not g_key:
        logger.warning("Tecla de construção de Granary não configurada.")
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
        logger.warning("Tecla de construção de Storage Pit não configurada.")
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
