"""HUD panel detection utilities."""
import logging
import time
from dataclasses import dataclass

from config import Config
from .. import CFG, ROOT, common, RESOURCE_ICON_ORDER, cache, cv2, np
from ... import screen_utils
from .roi import compute_resource_rois
from . import _get_resource_panel_cfg

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IconBounds:
    """Simple bounding box representation for matched icons."""

    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return bounds as a plain tuple for caching."""

        return self.x, self.y, self.w, self.h

    # Allow tuple-like unpacking and indexing for compatibility
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.w
        yield self.h

    def __getitem__(self, idx):
        return (self.x, self.y, self.w, self.h)[idx]


def match_icons(panel_gray, cfg, cache_obj: cache.ResourceCache = cache.RESOURCE_CACHE):
    """Match resource icons within ``panel_gray`` and update cache.

    Parameters
    ----------
    panel_gray:
        Grayscale image of the resource panel.
    cfg:
        Configuration section for the resource panel.
    cache_obj:
        Cache instance for persisting icon bounds.

    Returns
    -------
    dict[str, IconBounds]
        Mapping of icon name to detected bounds.
    """

    screen_utils.load_icon_templates()
    detected: dict[str, IconBounds] = {}

    for name in RESOURCE_ICON_ORDER:
        icon = screen_utils.ICON_TEMPLATES.get(name)
        if icon is None:
            continue
        best = (0.0, None, None)
        for scale in cfg.scales:
            icon_scaled = cv2.resize(icon, None, fx=scale, fy=scale)
            result = cv2.matchTemplate(panel_gray, icon_scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best[0]:
                best = (max_val, max_loc, icon_scaled.shape[::-1])
        score, loc, size = best
        if score >= cfg.match_threshold and loc is not None:
            bw, bh = size
            bounds = IconBounds(loc[0], loc[1], bw, bh)
            detected[name] = bounds
            cache_obj.last_icon_bounds[name] = bounds.as_tuple()
        elif name in cache_obj.last_icon_bounds:
            logger.info(
                "Using previous position for icon '%s'; score=%.3f", name, score
            )
            cached = cache_obj.last_icon_bounds[name]
            detected[name] = IconBounds(*cached)
        else:
            logger.warning("Icon '%s' not matched; score=%.3f", name, score)

    return detected


def adjust_food_roi(
    regions,
    spans,
    narrow,
    panel_left: int,
    panel_right: int,
) -> None:
    """Expand the food ROI when space allows to meet minimum width requirements."""

    deficit = narrow.get("food_stockpile")
    if "food_stockpile" not in spans or not deficit:
        return

    prev_span = spans.get("wood_stockpile")
    next_span = spans.get("gold_stockpile")
    prev_right = prev_span[1] if prev_span else panel_left
    next_left = next_span[0] if next_span else panel_right

    left, right = spans["food_stockpile"]
    space_left = left - prev_right
    space_right = next_left - right

    expand_left = min(deficit // 2, space_left)
    expand_right = min(deficit - expand_left, space_right)

    if not (expand_left or expand_right):
        return

    new_left = left - expand_left
    new_right = right + expand_right
    spans["food_stockpile"] = (new_left, new_right)
    fx, fy, fw, fh = regions["food_stockpile"]
    regions["food_stockpile"] = (new_left, fy, new_right - new_left, fh)

    actual = expand_left + expand_right
    if actual >= deficit:
        narrow.pop("food_stockpile", None)
    else:
        narrow["food_stockpile"] = deficit - actual


def detect_hud(frame, cfg: Config | None = None):
    """Locate the resource panel and return its bounding box and score."""

    from .. import find_template

    cfg = cfg or CFG
    tmpl = screen_utils.HUD_TEMPLATE
    if tmpl is None:
        return None, 0.0

    def _save_debug(img, heatmap):
        debug_dir = ROOT / "debug"
        debug_dir.mkdir(exist_ok=True)
        ts = int(time.time() * 1000)
        cv2.imwrite(str(debug_dir / f"resource_panel_fail_{ts}.png"), img)
        if heatmap is not None:
            hm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(
                str(debug_dir / f"resource_panel_heat_{ts}.png"), hm.astype("uint8")
            )

    box, score, heat = find_template(
        frame, tmpl, threshold=cfg["threshold"], scales=cfg["scales"]
    )
    if not box:
        logger.warning(
            "Resource panel template not matched; score=%.3f", score
        )
        _save_debug(frame, heat)
        fallback = cfg.get("threshold_fallback")
        if fallback is not None:
            box, score, heat = find_template(
                frame, tmpl, threshold=fallback, scales=cfg["scales"]
            )
            if not box:
                logger.warning(
                    "Resource panel template not matched with fallback; score=%.3f",
                    score,
                )
                _save_debug(frame, heat)
                return None, score
        else:
            return None, score

    return box, score


def locate_resource_panel(
    frame,
    cache_obj: cache.ResourceCache = cache.RESOURCE_CACHE,
    cfg: Config | None = None,
):
    """Locate the resource panel and return bounding boxes for each value."""

    cfg = cfg or CFG
    box, _score = detect_hud(frame, cfg)
    if not box:
        return {}

    x, y, w, h = box
    panel_gray = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)

    panel_cfg = _get_resource_panel_cfg(cfg)
    detected = match_icons(panel_gray, panel_cfg, cache_obj)

    if "population_limit" not in detected and "idle_villager" in detected:
        idle_bounds = detected["idle_villager"]
        prev = cache_obj.last_icon_bounds.get("population_limit")
        ph = prev[3] if prev else idle_bounds.h

        base_w = max(2 * idle_bounds.w, panel_cfg.min_pop_width)
        px = max(0, idle_bounds.x - base_w)
        pw = base_w + panel_cfg.pop_roi_extra_width

        bounds = IconBounds(px, idle_bounds.y, pw, ph)
        detected["population_limit"] = bounds
        cache_obj.last_icon_bounds["population_limit"] = bounds.as_tuple()

    if detected:
        min_y = min(v.y for v in detected.values())
        max_y = max(v.y + v.h for v in detected.values())
        top = y + min_y
        height = max_y - min_y
    else:
        top = y + int(panel_cfg.top_pct * h)
        height = int(panel_cfg.height_pct * h)

    regions, spans, narrow = compute_resource_rois(
        x,
        x + w,
        top,
        height,
        panel_cfg.pad_left,
        panel_cfg.pad_right,
        panel_cfg.icon_trims,
        panel_cfg.max_widths,
        panel_cfg.min_widths,
        panel_cfg.min_pop_width,
        panel_cfg.idle_roi_extra_width,
        panel_cfg.min_requireds,
        detected,
    )

    adjust_food_roi(regions, spans, narrow, x, x + w)

    cache._NARROW_ROIS = set(narrow.keys())
    cache._NARROW_ROI_DEFICITS = narrow.copy()
    cache._LAST_REGION_SPANS = spans.copy()

    if cache._LAST_REGION_BOUNDS != regions:
        cache._LAST_REGION_BOUNDS = regions.copy()
        cache_obj.last_resource_values.clear()
        cache_obj.last_resource_ts.clear()

    return regions
