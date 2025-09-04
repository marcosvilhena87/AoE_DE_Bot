
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
from script.input_utils import configure_pyautogui


def _calculate_next_tolerance(tolerance: int, increment: int, max_tolerance: int) -> int:
    """Increment the tolerance while capping it at ``max_tolerance``.

    Args:
        tolerance (int): Current tolerance value.
        increment (int): Amount to increase the tolerance per attempt.
        max_tolerance (int): Upper bound for the tolerance.

    Returns:
        int: The next tolerance value respecting the maximum limit.
    """

    return min(max_tolerance, tolerance + increment)


def _retry_ocr(resources_module, attempt: int, required, optional):
    """Retry OCR gathering with an incremental delay.

    Args:
        resources_module: Module providing ``gather_hud_stats``.
        attempt (int): Attempt number for calculating the delay.
        required (list[str]): Required HUD icons to read.
        optional (list[str]): Optional HUD icons to read.

    Returns:
        tuple[dict, tuple[int, int]]: Resource readings and population tuple.
    """

    return resources_module.gather_hud_stats(
        force_delay=0.1 * attempt,
        required_icons=required,
        optional_icons=optional,
    )


def _update_rois(resources_module, failing_keys):
    """Update ROI deficits for keys failing OCR with low confidence.

    Args:
        resources_module: Module maintaining ROI state.
        failing_keys (Iterable[str]): Resource keys that failed validation.
    """

    low_conf = resources_module.RESOURCE_CACHE.last_low_confidence
    no_digits = resources_module.RESOURCE_CACHE.last_no_digits
    for key in failing_keys:
        if key in low_conf or key in no_digits:
            resources_module._NARROW_ROI_DEFICITS[key] = (
                resources_module._NARROW_ROI_DEFICITS.get(key, 0) + 2
            )


def _within_relaxed_threshold(res, expected, threshold):
    """Check if ``res`` deviates from ``expected`` within ``threshold``.

    Args:
        res (dict): Actual resource values.
        expected (dict): Expected resource values.
        threshold (int): Maximum allowed absolute deviation.

    Returns:
        bool: ``True`` if all deviations are within ``threshold``.
    """

    max_dev = 0
    for key, val in expected.items():
        actual = res.get(key)
        if actual is None:
            return False
        max_dev = max(max_dev, abs(actual - val))
    return max_dev <= threshold


def validate_resources_with_retry(
    res,
    expected,
    config,
    required,
    optional,
    screen_utils_module=screen_utils,
    resources_module=resources,
    logger=None,
    cur_pop=None,
    pop_cap=None,
):
    """Validate starting resources with retries and adaptive tolerance.

    Args:
        res (dict): Initial resource readings.
        expected (dict): Expected resource values (non-zero only).
        config (dict): Configuration dictionary containing tolerance options.
        required (list[str]): Required HUD icons to read.
        optional (list[str]): Optional HUD icons to read.
        screen_utils_module: Module providing ``grab_frame``.
        resources_module: Module providing resource helpers.
        logger (logging.Logger, optional): Logger for status messages.

    Returns:
        tuple[dict, tuple[int, int]]: Final resource readings and population.
    """

    if logger is None:
        logger = logging.getLogger("campaign_bot")

    skip_validation = config.get("skip_starting_resource_validation", False)
    res_tolerances = config.get("resource_validation_tolerances", {})

    if expected and not skip_validation:
        retry_limit = config.get("resource_validation_retries", 3)
        tol_cfg = config.get("resource_validation_tolerance", {})
        tolerance = tol_cfg.get("initial", 10)
        increment = tol_cfg.get("increment", 5)
        max_tolerance = tolerance + increment
        relaxed_threshold = tolerance + 2 * increment
        for attempt in range(1, retry_limit + 1):
            frame = screen_utils_module.grab_frame()
            rois = getattr(resources_module, "_LAST_REGION_BOUNDS", {})
            logger.info(
                "Starting resource validation attempt %d/%d (±%d)",
                attempt,
                retry_limit,
                tolerance,
            )
            try:
                resources_module.validate_starting_resources(
                    res,
                    expected,
                    tolerance=tolerance,
                    tolerances=res_tolerances,
                    raise_on_error=True,
                    frame=frame,
                    rois=rois,
                )
                break
            except resources_module.ResourceValidationError as e:
                logger.warning(
                    "Starting resource validation attempt %d failed: %s",
                    attempt,
                    e,
                )
                if attempt == retry_limit:
                    if _within_relaxed_threshold(res, expected, relaxed_threshold):
                        logger.warning(
                            "Resource readings within ±%d; continuing.",
                            relaxed_threshold,
                        )
                        resources_module.validate_starting_resources(
                            res,
                            expected,
                            tolerance=tolerance,
                            tolerances=res_tolerances,
                            raise_on_error=False,
                            frame=frame,
                            rois=rois,
                        )
                        break
                    raise
                tolerance = _calculate_next_tolerance(
                    tolerance, increment, max_tolerance
                )
                _update_rois(resources_module, e.failing_keys)
                res, (cur_pop, pop_cap) = _retry_ocr(
                    resources_module, attempt + 1, required, optional
                )
    elif expected and skip_validation:
        logger.info(
            "Skipping starting resource validation; initial readings will be logged."
        )
    return res, (cur_pop, pop_cap)


def _scenario_to_module(path: str) -> str:
    """Convert a scenario file path to an importable module path.

    The file extension is stripped and non-alphanumeric characters in each
    path component are replaced with underscores.

    Args:
        path (str): Filesystem path to the scenario text file.

    Returns:
        str: Dot-separated module path corresponding to the scenario.
    """

    p = Path(path).with_suffix("")
    parts = [re.sub(r"\W", "_", part) for part in p.parts]
    return ".".join(parts)


def wait_for_hud_with_retry(
    timeout: int = 90, retry_delay: int = 25, max_retries: int = 2
):
    logger = logging.getLogger("campaign_bot")

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("HUD detection attempt %d/%d", attempt, max_retries)
            anchor, asset = hud.wait_hud(timeout=timeout)
            logger.info("HUD detected at %s using '%s'.", anchor, asset)
            return anchor, asset
        except RuntimeError as e:
            logger.error("%s (attempt %d/%d)", e, attempt, max_retries)
            if attempt < max_retries:
                logger.info(
                    "Giving another %ds for you to adjust the camera/HUD (next: %d/%d)…",
                    retry_delay,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(retry_delay)
            else:
                logger.warning(
                    "HUD not detected after %d attempts; routine will continue without anchored HUD.",
                    max_retries,
                )
                raise SystemExit(
                    f"HUD not detected after {max_retries} attempts; exiting script."
                ) from e

def main(config_path: str | Path | None = None) -> None:
    """Run a campaign mission based on command-line arguments.

    The function parses CLI options, configures logging and screen capture,
    waits for the in-game HUD, validates starting resources, and then imports
    and executes the mission module for the chosen scenario.

    Args:
        config_path: Optional configuration file path.
        --scenario (str): Path to the scenario text file. Defaults to the
            configuration value or
            ``campaigns/Ascent_of_Egypt/Egypt_1_Hunting.txt``.

    Returns:
        None
    """

    state = common.init_common(config_path)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        default=state.config.get(
            "scenario_path", "campaigns/Ascent_of_Egypt/Egypt_1_Hunting.txt"
        ),
        help="Path to scenario text file",
    )
    args = parser.parse_args()
    scenario_name = Path(args.scenario).stem

    logging.basicConfig(
        level=logging.DEBUG if state.config.get("verbose_logging") else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("campaign_bot")
    configure_pyautogui()

    screen_utils.screen_capture.init_sct()
    try:
        logger.info(
            "Enter the campaign mission (%s). The script starts when the HUD is detected…",
            scenario_name,
        )
        anchor, asset = wait_for_hud_with_retry(timeout=90, retry_delay=25)

        info = parse_scenario_info(args.scenario)
        state.current_pop = info.starting_villagers
        state.pop_cap = 4  # 1 Town Center
        state.target_pop = info.objective_villagers
        idle_start = getattr(info, "starting_idle_villagers", info.starting_villagers)

        # Reset resource cache to prevent stale OCR values across scenarios
        resources.cache.reset()

        resources.RESOURCE_CACHE.last_resource_values["idle_villager"] = idle_start
        resources.RESOURCE_CACHE.last_resource_ts["idle_villager"] = time.time()
        try:
            icon_cfg = state.config.get("hud_icons", {})
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
            res, (cur_pop, pop_cap) = validate_resources_with_retry(
                res,
                non_zero,
                state.config,
                required,
                optional,
                screen_utils_module=screen_utils,
                resources_module=resources,
                logger=logger,
                cur_pop=cur_pop,
                pop_cap=pop_cap,
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
        logger.info("Attempting to complete scenario objectives for %s...", module_name)
        if accepts_info:
            func(info, state=state)
        else:
            func(state=state)
        logger.info("Mission '%s' completed.", module_name)
    finally:
        screen_utils.screen_capture.teardown_sct()


if __name__ == "__main__":
    main()
