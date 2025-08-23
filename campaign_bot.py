import logging
import time

import script.common as common
from script.villager import econ_loop


def main():
    logging.info(
        "Entre na missão da campanha (Hunting). O script inicia quando detectar a HUD…"
    )
    try:
        hud, asset = common.wait_hud(timeout=90)
        logging.info(
            "HUD detectada em %s usando '%s'. Rodando rotina econômica…",
            hud,
            asset,
        )
    except RuntimeError as e:
        logging.error(str(e))
        logging.info("Dando mais 25s para você ajustar a câmera/HUD (fallback)…")
        time.sleep(25)
        try:
            hud, asset = common.wait_hud(timeout=90)
            logging.info(
                "HUD detectada em %s usando '%s'. Rodando rotina econômica…",
                hud,
                asset,
            )
        except RuntimeError as e2:
            logging.error(str(e2))
            logging.warning(
                "HUD não detectada após tentativa extra; rotina continuará sem HUD ancorada."
            )
            raise SystemExit(
                "HUD não detectada após duas tentativas; encerrando script."
            )

    info = common.parse_scenario_info("campaigns/Ascent_of_Egypt/1.Hunting.txt")
    common.CURRENT_POP = info.starting_villagers
    common.POP_CAP = 4  # 1 Town Center
    common.TARGET_POP = info.objective_villagers

    econ_loop(minutes=common.CFG["loop_minutes"])
    logging.info("Rotina concluída.")


if __name__ == "__main__":
    main()
