"""Resource panel utilities."""
import logging
from dataclasses import dataclass
from typing import Iterable

from config import Config
from .. import CFG, common, RESOURCE_ICON_ORDER, cache
from ... import screen_utils

logger = logging.getLogger(__name__)


@dataclass
class ResourcePanelCfg:
    match_threshold: float
    scales: Iterable[float]
    pad_left: list
    pad_right: list
    icon_trims: list
    max_widths: list
    min_widths: list
    min_requireds: list
    top_pct: float
    height_pct: float
    idle_roi_extra_width: int
    min_pop_width: int
    pop_roi_extra_width: int


def _get_resource_panel_cfg(cfg: Config | None = None):
    """Return processed configuration values for the resource panel."""

    cfg = cfg or CFG
    res_cfg = cfg.get("resource_panel", {})
    profile = cfg.get("profile")
    profile_cfg = cfg.get("profiles", {}).get(profile, {})
    profile_res = profile_cfg.get("resource_panel", {})

    match_threshold = profile_res.get(
        "match_threshold", res_cfg.get("match_threshold", 0.8)
    )
    scales = res_cfg.get("scales", cfg.get("scales", [1.0]))

    pad_left = res_cfg.get("roi_padding_left", 2)
    pad_right = res_cfg.get("roi_padding_right", 2)
    num_icons = len(RESOURCE_ICON_ORDER)
    pad_left = pad_left if isinstance(pad_left, (list, tuple)) else [pad_left] * num_icons
    pad_right = pad_right if isinstance(pad_right, (list, tuple)) else [pad_right] * num_icons

    icon_trims = profile_res.get(
        "icon_trim_pct", res_cfg.get("icon_trim_pct", [0] * num_icons)
    )
    icon_trims = (
        icon_trims if isinstance(icon_trims, (list, tuple)) else [icon_trims] * num_icons
    )

    max_width_cfg = res_cfg.get("max_width", 160)
    max_widths = (
        max_width_cfg
        if isinstance(max_width_cfg, (list, tuple))
        else [max_width_cfg] * num_icons
    )

    min_width_cfg = res_cfg.get("min_width", 90)
    min_widths = (
        min_width_cfg
        if isinstance(min_width_cfg, (list, tuple))
        else [min_width_cfg] * num_icons
    )

    min_req_cfg = res_cfg.get("min_required_width", 0)
    min_requireds = (
        min_req_cfg
        if isinstance(min_req_cfg, (list, tuple))
        else [min_req_cfg] * num_icons
    )

    top_pct = profile_res.get("top_pct", res_cfg.get("top_pct", 0.08))
    height_pct = profile_res.get("height_pct", res_cfg.get("height_pct", 0.84))

    idle_extra = res_cfg.get("idle_roi_extra_width", 0)
    min_pop_width = res_cfg.get("min_pop_width", 0)
    pop_extra = res_cfg.get("pop_roi_extra_width", 0)

    return ResourcePanelCfg(
        match_threshold,
        scales,
        pad_left,
        pad_right,
        icon_trims,
        max_widths,
        min_widths,
        min_requireds,
        top_pct,
        height_pct,
        idle_extra,
        min_pop_width,
        pop_extra,
    )


from .roi import compute_resource_rois, _fallback_rois_from_slice
from .detection import detect_hud, locate_resource_panel
from .calibration import (
    _auto_calibrate_from_icons,
    _recalibrate_low_variance,
    _apply_custom_rois,
    _remove_overlaps,
)


def detect_resource_regions(
    frame, required_icons, cache_obj: cache.ResourceCache = cache.RESOURCE_CACHE
):
    """Detect resource value regions on the HUD."""

    regions = locate_resource_panel(frame, cache_obj)
    regions = _apply_custom_rois(frame, regions)
    missing = [name for name in required_icons if name not in regions]

    if missing:
        calibrated = _auto_calibrate_from_icons(frame, cache_obj)
        if calibrated:
            regions = calibrated
            if not CFG.get("disable_roi_overrides_on_calibration"):
                regions = _apply_custom_rois(frame, regions, include_idle=False)
            missing = [name for name in required_icons if name not in regions]

    if missing and common.HUD_ANCHOR:
        if common.HUD_ANCHOR.get("asset") == "assets/resources.png":
            x = common.HUD_ANCHOR["left"]
            y = common.HUD_ANCHOR["top"]
            w = common.HUD_ANCHOR["width"]
            h = common.HUD_ANCHOR["height"]

            res_cfg = CFG.get("resource_panel", {})
            profile = CFG.get("profile")
            profile_res = CFG.get("profiles", {}).get(profile, {}).get(
                "resource_panel", {},
            )
            top_pct = profile_res.get("top_pct", res_cfg.get("top_pct", 0.08))
            height_pct = profile_res.get("height_pct", res_cfg.get("height_pct", 0.84))
            icon_trims_cfg = profile_res.get(
                "icon_trim_pct",
                res_cfg.get(
                    "icon_trim_pct",
                    [0.25, 0.20, 0.20, 0.20, 0.20, 0.20],
                ),
            )
            if not isinstance(icon_trims_cfg, (list, tuple)):
                icon_trims_cfg = [icon_trims_cfg] * 6
            right_trim = profile_res.get(
                "right_trim_pct", res_cfg.get("right_trim_pct", 0.02)
            )

            top = y + int(top_pct * h)
            height = int(height_pct * h)

            regions = _fallback_rois_from_slice(
                x,
                w,
                top,
                height,
                icon_trims_cfg,
                right_trim,
                required_icons,
            )
        else:
            W, H = screen_utils.get_screen_size()
            margin = int(0.01 * W)
            panel_w = int(568 / 1920 * W)
            panel_h = int(59 / 1080 * H)
            x = common.HUD_ANCHOR["left"] + common.HUD_ANCHOR["width"] + margin
            y = common.HUD_ANCHOR["top"]

            res_cfg = CFG.get("resource_panel", {})
            profile = CFG.get("profile")
            profile_res = CFG.get("profiles", {}).get(profile, {}).get(
                "resource_panel", {},
            )
            top_pct = profile_res.get(
                "anchor_top_pct", res_cfg.get("anchor_top_pct", 0.15)
            )
            height_pct = profile_res.get(
                "anchor_height_pct", res_cfg.get("anchor_height_pct", 0.70)
            )
            icon_trims = profile_res.get(
                "anchor_icon_trim_pct",
                res_cfg.get(
                    "anchor_icon_trim_pct",
                    [0.42, 0.42, 0.35, 0.35, 0.35, 0.35],
                ),
            )
            if not isinstance(icon_trims, (list, tuple)):
                icon_trims = [icon_trims] * 6
            right_trim = profile_res.get(
                "anchor_right_trim_pct", res_cfg.get("anchor_right_trim_pct", 0.02)
            )

            top = y + int(top_pct * panel_h)
            height = int(height_pct * panel_h)
            regions = _fallback_rois_from_slice(
                x,
                panel_w,
                top,
                height,
                icon_trims,
                right_trim,
                required_icons,
            )

        missing = [name for name in required_icons if name not in regions]

    if not missing:
        regions = _recalibrate_low_variance(frame, regions, required_icons, cache_obj)
        missing = [name for name in required_icons if name not in regions]

    if missing:
        raise common.ResourceReadError(
            "Resource icon(s) not located on HUD: " + ", ".join(missing)
        )
    regions = _remove_overlaps(regions, required_icons)

    return regions


__all__ = [
    "detect_hud",
    "compute_resource_rois",
    "_get_resource_panel_cfg",
    "locate_resource_panel",
    "detect_resource_regions",
    "_auto_calibrate_from_icons",
    "_fallback_rois_from_slice",
    "_apply_custom_rois",
    "_recalibrate_low_variance",
    "_remove_overlaps",
    "ResourcePanelCfg",
]
