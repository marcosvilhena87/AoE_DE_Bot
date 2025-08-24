
import argparse
import logging
import time

import script.common as common
import script.hud as hud
import script.screen_utils as screen_utils
from script.config_utils import parse_scenario_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        default=common.CFG.get(
            "scenario_path", "campaigns/Ascent_of_Egypt/1.Hunting.txt"
        ),
        help="Path to scenario text file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if common.CFG.get("verbose_logging") else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("campaign_bot")

    screen_utils.init_sct()
    try:
        logger.info(
            "Entre na missão da campanha (Hunting). O script inicia quando detectar a HUD…",
        )
        try:
            anchor, asset = hud.wait_hud(timeout=90)
            logger.info("HUD detectada em %s usando '%s'.", anchor, asset)
        except RuntimeError as e:
            logger.error(str(e))
            logger.info("Dando mais 25s para você ajustar a câmera/HUD (fallback)…")
            time.sleep(25)
            try:
                anchor, asset = hud.wait_hud(timeout=90)
                logger.info("HUD detectada em %s usando '%s'.", anchor, asset)
            except RuntimeError as e2:
                logger.error(str(e2))
                logger.warning(
                    "HUD não detectada após tentativa extra; rotina continuará sem HUD ancorada."
                )
                raise SystemExit(
                    "HUD não detectada após duas tentativas; encerrando script."
                )

        info = parse_scenario_info(args.scenario)
        common.CURRENT_POP = info.starting_villagers
        common.POP_CAP = 4  # 1 Town Center
        common.TARGET_POP = info.objective_villagers

        logger.info("Setup concluído.")
    finally:
        screen_utils.teardown_sct()


if __name__ == "__main__":
    main()
