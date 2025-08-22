import logging
import time

import script.common as common


def select_idle_villager():
    """Selecione um aldeão ocioso usando a tecla configurada."""
    common._press_key_safe(common.CFG["keys"]["idle_vill"], 0.10)


def build_house():
    """Constrói uma casa no local predefinido."""
    common._press_key_safe(common.CFG["keys"]["build_menu"], 0.05)
    house_key = common.CFG["keys"].get("house")
    if house_key:
        common._press_key_safe(house_key, 0.15)
        hx, hy = common.CFG["areas"]["house_spot"]
        common._click_norm(hx, hy)


def build_granary():
    """Posiciona um Granary no ponto configurado."""
    common._press_key_safe(common.CFG["keys"]["build_menu"], 0.05)
    g_key = common.CFG["keys"].get("granary")
    if g_key:
        common._press_key_safe(g_key, 0.15)
        gx, gy = common.CFG["areas"]["granary_spot"]
        common._click_norm(gx, gy)


def build_storage_pit():
    """Posiciona um Storage Pit no ponto configurado."""
    common._press_key_safe(common.CFG["keys"]["build_menu"], 0.05)
    s_key = common.CFG["keys"].get("storage_pit")
    if s_key:
        common._press_key_safe(s_key, 0.15)
        sx, sy = common.CFG["areas"]["storage_spot"]
        common._click_norm(sx, sy)


def econ_loop(minutes=5):
    """Rotina econômica básica para a missão Hunting."""
    import script.town_center as town_center

    logging.info("Iniciando rotina econômica por %s minutos", minutes)
    town_center.train_villagers(common.TARGET_POP)

    select_idle_villager()
    build_granary()
    logging.info("Granary posicionado")
    time.sleep(0.5)
    select_idle_villager()
    build_storage_pit()
    logging.info("Storage Pit posicionado")
    time.sleep(0.5)

    hunt_x, hunt_y = common.CFG["areas"]["hunt_food"]
    wood_x, wood_y = common.CFG["areas"]["wood"]

    t0 = time.time()
    while time.time() - t0 < minutes * 60:
        select_idle_villager()
        common._click_norm(hunt_x, hunt_y)
        time.sleep(common.CFG["timers"]["idle_gap"])

        select_idle_villager()
        common._click_norm(wood_x, wood_y)
        time.sleep(common.CFG["timers"]["idle_gap"])

        if common.CURRENT_POP >= common.POP_CAP - 2:
            select_idle_villager()
            build_house()
            logging.info("Casa construída para expandir população")
            time.sleep(0.5)
            common.POP_CAP += 4

        time.sleep(common.CFG["timers"]["loop_sleep"])
    logging.info("Rotina econômica finalizada")
