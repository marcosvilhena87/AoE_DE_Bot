
import argparse
import importlib
import logging
import re
import time
from pathlib import Path
import inspect

import script.common as common
import script.hud as hud
import script.resources.reader as resources
import script.screen_utils as screen_utils
from script.config_utils import parse_scenario_info


def _scenario_to_module(path: str) -> str:
    p = Path(path).with_suffix("")
    parts = [re.sub(r"\W", "_", part) for part in p.parts]
    return ".".join(parts)


def wait_for_hud_with_retry(timeout: int = 90, retry_delay: int = 25):
    logger = logging.getLogger("campaign_bot")
    try:
        anchor, asset = hud.wait_hud(timeout=timeout)
        logger.info("HUD detected at %s using '%s'.", anchor, asset)
        return anchor, asset
    except RuntimeError as e:
        logger.error(str(e))
        logger.info("Giving another %ds for you to adjust the camera/HUD (fallback)…", retry_delay)
        time.sleep(retry_delay)
        try:
            anchor, asset = hud.wait_hud(timeout=timeout)
            logger.info("HUD detected at %s using '%s'.", anchor, asset)
            return anchor, asset
        except RuntimeError as e2:
            logger.error(str(e2))
            logger.warning(
                "HUD not detected after extra attempt; routine will continue without anchored HUD."
            )
            raise SystemExit(
                "HUD not detected after two attempts; exiting script."
            ) from e2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        default=common.CFG.get(
            "scenario_path", "campaigns/Ascent_of_Egypt/Egypt_1_Hunting.txt"
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
        anchor, asset = wait_for_hud_with_retry(timeout=90, retry_delay=25)

        info = parse_scenario_info(args.scenario)
        common.CURRENT_POP = info.starting_villagers
        common.POP_CAP = 4  # 1 Town Center
        common.TARGET_POP = info.objective_villagers
        idle_start = getattr(info, "starting_idle_villagers", info.starting_villagers)

        # Reset resource cache to prevent stale OCR values across scenarios
        resources.RESOURCE_CACHE.last_resource_values.clear()
        resources.RESOURCE_CACHE.last_resource_ts.clear()
        resources.RESOURCE_CACHE.resource_failure_counts.clear()

        resources.RESOURCE_CACHE.last_resource_values["idle_villager"] = idle_start
        resources.RESOURCE_CACHE.last_resource_ts["idle_villager"] = time.time()
        try:
            icon_cfg = common.CFG.get("hud_icons", {})
            if info.starting_resources is None:
                logger.warning(
                    "No starting resources specified; skipping validation."
                )
                starting_res = {}
            else:
                starting_res = info.starting_resources
            non_zero = {k: v for k, v in starting_res.items() if v}
            zero_icons = {k for k, v in starting_res.items() if v == 0}
            required = [i for i in icon_cfg.get("required", []) if i not in zero_icons]
            optional = [i for i in icon_cfg.get("optional", []) if i not in zero_icons]
            res, (cur_pop, pop_cap) = resources.gather_hud_stats(
                force_delay=0.1,
                required_icons=required,
                optional_icons=optional,
            )
            skip_validation = common.CFG.get(
                "skip_starting_resource_validation", False
            )
            res_tolerances = common.CFG.get("resource_validation_tolerances", {})
            if non_zero and not skip_validation:
                retry_limit = common.CFG.get("resource_validation_retries", 3)
                tol_cfg = common.CFG.get("resource_validation_tolerance", {})
                tolerance = tol_cfg.get("initial", 10)
                increment = tol_cfg.get("increment", 5)
                max_tolerance = tolerance + increment
                relaxed_threshold = tolerance + 2 * increment
                attempt = 1
                while True:
                    frame = screen_utils._grab_frame()
                    rois = getattr(resources, "_LAST_REGION_BOUNDS", {})
                    logger.info(
                        "Starting resource validation attempt %d/%d (±%d)",
                        attempt,
                        retry_limit,
                        tolerance,
                    )
                    try:
                        resources.validate_starting_resources(
                            res,
                            non_zero,
                            tolerance=tolerance,
                            tolerances=res_tolerances,
                            raise_on_error=True,
                            frame=frame,
                            rois=rois,
                        )
                        break
                    except resources.ResourceValidationError as e:
                        logger.warning(
                            "Starting resource validation attempt %d failed: %s",
                            attempt,
                            e,
                        )
                        if attempt >= retry_limit:
                            max_dev = 0
                            for k, v in non_zero.items():
                                actual = res.get(k)
                                if actual is None:
                                    max_dev = float("inf")
                                    break
                                max_dev = max(max_dev, abs(actual - v))
                            if max_dev <= relaxed_threshold:
                                logger.warning(
                                    "Resource readings within ±%d; continuing.",
                                    relaxed_threshold,
                                )
                                resources.validate_starting_resources(
                                    res,
                                    non_zero,
                                    tolerance=tolerance,
                                    tolerances=res_tolerances,
                                    raise_on_error=False,
                                    frame=frame,
                                    rois=rois,
                                )
                                break
                            else:
                                raise
                        attempt += 1
                        tolerance = min(max_tolerance, tolerance + increment)
                        low_conf = resources.RESOURCE_CACHE.last_low_confidence
                        no_digits = resources.RESOURCE_CACHE.last_no_digits
                        for k in e.failing_keys:
                            if k in low_conf or k in no_digits:
                                resources._NARROW_ROI_DEFICITS[k] = (
                                    resources._NARROW_ROI_DEFICITS.get(k, 0) + 2
                                )
                        res, (cur_pop, pop_cap) = resources.gather_hud_stats(
                            force_delay=0.1 * attempt,
                            required_icons=required,
                            optional_icons=optional,
                        )
            elif non_zero and skip_validation:
                logger.info(
                    "Skipping starting resource validation; initial readings will be logged."
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
        except (common.ResourceReadError, common.PopulationReadError) as e:
            logger.error("Failed to detect resources or population: %s", e)
            raise SystemExit("Failed to detect resources or population") from e

        logger.info("Setup complete.")
        module_name = _scenario_to_module(args.scenario)
        try:
            mission = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            logger.error("Failed to import mission module '%s': %s", module_name, e)
            raise SystemExit(f"Mission module '{module_name}' not found") from e
        func = getattr(mission, "run_mission", None) or getattr(mission, "main", None)
        if func is None:
            logger.error(
                "Mission module '%s' lacks 'run_mission' or 'main' entry point", module_name
            )
            raise SystemExit(
                f"Mission module '{module_name}' lacks run_mission or main"
            )
        sig = inspect.signature(func)
        accepts_info = "info" in sig.parameters
        logger.info("Starting mission '%s'.", module_name)
        logger.info("Tentando cumprir os objetivos do cenário %s...", module_name)
        if accepts_info:
            func(info)
        else:
            func()
        logger.info("Mission '%s' completed.", module_name)
    finally:
        screen_utils.teardown_sct()


if __name__ == "__main__":
    main()
