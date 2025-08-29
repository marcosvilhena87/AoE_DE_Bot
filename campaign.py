
import argparse
import logging
import time
from pathlib import Path

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
    scenario_name = Path(args.scenario).stem

    logging.basicConfig(
        level=logging.DEBUG if common.CFG.get("verbose_logging") else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("campaign_bot")

    screen_utils.init_sct()
    try:
        logger.info(
            "Enter the campaign mission (%s). The script starts when the HUD is detected…",
            scenario_name,
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
            non_zero = {k: v for k, v in info.starting_resources.items() if v}
            zero_icons = {k for k, v in info.starting_resources.items() if v == 0}
            required = [i for i in icon_cfg.get("required", []) if i not in zero_icons]
            optional = [i for i in icon_cfg.get("optional", []) if i not in zero_icons]
            res, (cur_pop, pop_cap) = resources.gather_hud_stats(
                force_delay=0.1,
                required_icons=required,
                optional_icons=optional,
            )
            try:
                resources.validate_starting_resources(
                    res,
                    non_zero,
                    tolerance=10,
                    raise_on_error=True,
                )
            except ValueError as e:
                logger.warning("Starting resource validation failed: %s", e)
                # Retry OCR once more and attempt to save ROI diagnostics
                res, (cur_pop, pop_cap) = resources.gather_hud_stats(
                    force_delay=0.2,
                    required_icons=required,
                    optional_icons=optional,
                )
                frame = screen_utils._grab_frame()
                rois = getattr(resources, "_LAST_REGION_BOUNDS", {})
                try:
                    resources.validate_starting_resources(
                        res,
                        non_zero,
                        tolerance=10,
                        raise_on_error=True,
                        frame=frame,
                        rois=rois,
                    )
                except ValueError as e2:
                    logger.error(
                        "Second resource validation failed: %s", e2
                    )
                    max_dev = 0
                    for k, v in non_zero.items():
                        actual = res.get(k)
                        if actual is None:
                            max_dev = float("inf")
                            break
                        max_dev = max(max_dev, abs(actual - v))
                    if max_dev <= 20:
                        logger.warning(
                            "Resource readings close to expected; continuing."
                        )
                        resources.validate_starting_resources(
                            res,
                            non_zero,
                            tolerance=10,
                            raise_on_error=False,
                            frame=frame,
                            rois=rois,
                        )
                    else:
                        raise
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
