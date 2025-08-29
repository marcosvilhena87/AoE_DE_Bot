
import argparse
import logging
import time

import script.common as common
import script.hud as hud
import script.screen_utils as screen_utils
import script.resources as resources
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
            "Enter the campaign mission (Hunting). The script starts when the HUD is detected…",
        )
        try:
            anchor, asset = hud.wait_hud(timeout=90)
            logger.info("HUD detected at %s using '%s'.", anchor, asset)
        except RuntimeError as e:
            logger.error(str(e))
            logger.info("Giving another 25s for you to adjust the camera/HUD (fallback)…")
            time.sleep(25)
            try:
                anchor, asset = hud.wait_hud(timeout=90)
                logger.info("HUD detected at %s using '%s'.", anchor, asset)
            except RuntimeError as e2:
                logger.error(str(e2))
                logger.warning(
                    "HUD not detected after extra attempt; routine will continue without anchored HUD."
                )
                raise SystemExit(
                    "HUD not detected after two attempts; exiting script."
                )

        info = parse_scenario_info(args.scenario)
        common.CURRENT_POP = info.starting_villagers
        common.POP_CAP = 4  # 1 Town Center
        common.TARGET_POP = info.objective_villagers
        try:
            icon_cfg = common.CFG.get("hud_icons", {})
            res, (cur_pop, pop_cap) = resources.gather_hud_stats(
                force_delay=0.1,
                required_icons=icon_cfg.get("required"),
                optional_icons=icon_cfg.get("optional"),
            )
            resources.validate_starting_resources(
                res,
                info.starting_resources,
                tolerance=10,
                raise_on_error=True,
            )
            logger.info(
                "Detected resources: wood=%s, food=%s, gold=%s, stone=%s",
                res.get("wood_stockpile"),
                res.get("food_stockpile"),
                res.get("gold_stockpile"),
                res.get("stone_stockpile"),
            )
            logger.info("Detected population: %s/%s", cur_pop, pop_cap)
            logger.info(
                "Detected idle villagers: %s", res.get("idle_villager")
            )
        except Exception as e:
            logger.error("Failed to detect resources or population: %s", e)
            raise SystemExit("Failed to detect resources or population")

        logger.info("Setup complete.")
    finally:
        screen_utils.teardown_sct()


if __name__ == "__main__":
    main()
