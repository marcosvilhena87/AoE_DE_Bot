import logging
import time
from pathlib import Path

import cv2

from script.common import BotState, STATE, ResourceReadError
import script.input_utils as input_utils
from script import resources, hud, template_utils, screen_utils

logger = logging.getLogger(__name__)


def select_idle_villager(delay: float = 0.1, state: BotState = STATE) -> bool:
    """Pressiona imediatamente a tecla configurada para selecionar um aldeão ocioso."""

    key = state.config["keys"].get("idle_vill")
    if not key:
        return False
    input_utils._press_key_safe(key, delay)
    return True


def build_house(state: BotState = STATE):
    """Constrói uma casa no local predefinido com confirmação.

    Após emitir o comando de construção, a função tenta confirmar o sucesso
    monitorando os recursos, população e presença do template de uma casa na
    região clicada. A confirmação exige que pelo menos dois dos seguintes
    critérios sejam verdadeiros: aumento da população máxima, consumo de
    ~30 de madeira ou detecção do template da casa.

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

    threshold = state.config.get("threshold", 0.8)
    tmpl_path = Path("assets/house.png")
    tmpl = cv2.imread(str(tmpl_path), cv2.IMREAD_GRAYSCALE) if tmpl_path.exists() else None

    for hx, hy in spots:
        wood_before = pop_before = None
        try:
            res_vals, (_cur, pop_cap) = resources.reader.read_resources_from_hud(
                required_icons=["wood_stockpile", "population_limit"],
                icons_to_read=["wood_stockpile", "population_limit"],
            )
            wood_before = res_vals.get("wood_stockpile")
            pop_before = pop_cap
        except ResourceReadError:
            logger.debug("Resource bar missing; proceeding without initial wood/pop readings")

        if pop_before is None:
            _cur, pop_before, _low = hud.read_population_from_hud()

        x, y = input_utils._to_px(hx, hy)
        roi_w = roi_h = 80
        W, H = screen_utils.get_screen_size()
        rx = max(0, min(x - roi_w // 2, W - roi_w))
        ry = max(0, min(y - roi_h // 2, H - roi_h))
        roi_bbox = (rx, ry, roi_w, roi_h)

        ts = time.time()
        logger.info(
            "house(cmd): wood_before=%s pop_before=%s roi_bbox=%s ts=%d",
            wood_before,
            pop_before,
            roi_bbox,
            int(ts * 1000),
        )

        select_idle_villager(state=state)
        input_utils._press_key_safe(house_key, 0.15)
        input_utils._click_norm(hx, hy, button="right")

        state_name = "ISSUED"
        best_score = -1.0
        wood_after = pop_after = None
        queue_val = None
        deadline = ts + 25.0

        while time.time() < deadline:
            try:
                res_vals, (_cur, pop_cap) = resources.reader.read_resources_from_hud(
                    required_icons=["wood_stockpile", "population_limit"],
                    icons_to_read=["wood_stockpile", "population_limit"],
                )
                wood_after = res_vals.get("wood_stockpile", wood_after)
                pop_after = pop_cap if pop_cap is not None else pop_after
            except ResourceReadError:
                pass

            _cur, pop_cap, _low = hud.read_population_from_hud()
            if pop_cap is not None:
                pop_after = pop_cap

            frame = screen_utils.screen_capture.grab_frame()
            rx, ry, rw, rh = roi_bbox
            roi = frame[ry : ry + rh, rx : rx + rw]
            if tmpl is not None:
                _box, score, _heat = template_utils.find_template(
                    roi, tmpl, threshold=threshold, scales=state.config.get("scales")
                )
                if score is not None and score > best_score:
                    best_score = score
            else:
                score = None

            queue_reader = getattr(hud, "read_villager_build_queue_from_hud", None)
            if queue_reader:
                try:
                    queue_val = queue_reader()
                except Exception:
                    queue_val = None

            cond_pop = (
                pop_before is not None and pop_after is not None and pop_after > pop_before
            )
            cond_wood = (
                wood_before is not None
                and wood_after is not None
                and abs((wood_before - wood_after) - 30) <= 5
            )
            cond_tmpl = best_score >= threshold if best_score >= 0 else False

            if state_name == "ISSUED" and (cond_pop or cond_wood or cond_tmpl):
                state_name = "BUILDING_SEEN"

            if state_name == "BUILDING_SEEN" and sum((cond_pop, cond_wood, cond_tmpl)) >= 2:
                elapsed_ms = int((time.time() - ts) * 1000)
                logger.info(
                    "house(confirm): pop %s->%s wood %s->%s score=%.3f roi=%s elapsed_ms=%d queue=%s",
                    pop_before,
                    pop_after,
                    wood_before,
                    wood_after,
                    best_score,
                    roi_bbox,
                    elapsed_ms,
                    queue_val,
                )
                if pop_after is not None:
                    state.pop_cap = pop_after
                else:
                    state.pop_cap += 4
                return True

            time.sleep(1.0)

        elapsed_ms = int((time.time() - ts) * 1000)
        logger.warning(
            "house(failed-timeout): pop %s->%s wood %s->%s score=%.3f roi=%s elapsed_ms=%d",
            pop_before,
            pop_after,
            wood_before,
            wood_after,
            best_score,
            roi_bbox,
            elapsed_ms,
        )

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


def build_dock(state: BotState = STATE):
    """Posiciona um Dock no ponto configurado."""
    input_utils._press_key_safe(state.config["keys"]["build_menu"], 0.05)
    d_key = state.config["keys"].get("dock")
    if not d_key:
        logger.warning("Dock build key not configured.")
        return False
    areas = state.config.get("areas", {})
    spot = areas.get("dock_spot")
    if not spot:
        logger.warning("Dock spot not configured.")
        return False
    input_utils._press_key_safe(d_key, 0.15)
    dx, dy = spot
    input_utils._click_norm(dx, dy)
    return True
