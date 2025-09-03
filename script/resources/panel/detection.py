"""HUD panel detection utilities."""
import logging
import time

from .. import CFG, ROOT, screen_utils, common, RESOURCE_ICON_ORDER, cache, cv2, np
from .roi import compute_resource_rois
from . import _get_resource_panel_cfg

logger = logging.getLogger(__name__)


def detect_hud(frame):
    """Locate the resource panel and return its bounding box and score."""

    from .. import find_template

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
        frame, tmpl, threshold=CFG["threshold"], scales=CFG["scales"]
    )
    if not box:
        logger.warning(
            "Resource panel template not matched; score=%.3f", score
        )
        _save_debug(frame, heat)
        fallback = CFG.get("threshold_fallback")
        if fallback is not None:
            box, score, heat = find_template(
                frame, tmpl, threshold=fallback, scales=CFG["scales"]
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


def locate_resource_panel(frame, cache_obj: cache.ResourceCache = cache.RESOURCE_CACHE):
    """Locate the resource panel and return bounding boxes for each value."""

    box, _score = detect_hud(frame)
    if not box:
        return {}

    x, y, w, h = box
    panel_gray = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)

    cfg = _get_resource_panel_cfg()
    screen_utils._load_icon_templates()

    detected = {}
    for name in RESOURCE_ICON_ORDER:
        icon = screen_utils.ICON_TEMPLATES.get(name)
        if icon is None:
            continue
        best = (0, None, None)
        for scale in cfg.scales:
            icon_scaled = cv2.resize(icon, None, fx=scale, fy=scale)
            result = cv2.matchTemplate(panel_gray, icon_scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best[0]:
                best = (max_val, max_loc, icon_scaled.shape[::-1])
        if best[0] >= cfg.match_threshold and best[1] is not None:
            bw, bh = best[2]
            detected[name] = (best[1][0], best[1][1], bw, bh)
            cache_obj.last_icon_bounds[name] = (best[1][0], best[1][1], bw, bh)
        elif name in cache_obj.last_icon_bounds:
            logger.info(
                "Using previous position for icon '%s'; score=%.3f", name, best[0]
            )
            detected[name] = cache_obj.last_icon_bounds[name]
        else:
            logger.warning("Icon '%s' not matched; score=%.3f", name, best[0])

    if "population_limit" not in detected and "idle_villager" in detected:
        xi, yi, wi, hi = detected["idle_villager"]
        prev = cache_obj.last_icon_bounds.get("population_limit")
        ph = prev[3] if prev else hi

        base_w = max(2 * wi, cfg.min_pop_width)
        px = max(0, xi - base_w)
        pw = base_w + cfg.pop_roi_extra_width

        detected["population_limit"] = (px, yi, pw, ph)
        cache_obj.last_icon_bounds["population_limit"] = (px, yi, pw, ph)

    if detected:
        min_y = min(v[1] for v in detected.values())
        max_y = max(v[1] + v[3] for v in detected.values())
        top = y + min_y
        height = max_y - min_y
    else:
        top = y + int(cfg.top_pct * h)
        height = int(cfg.height_pct * h)

    regions, spans, narrow = compute_resource_rois(
        x,
        x + w,
        top,
        height,
        cfg.pad_left,
        cfg.pad_right,
        cfg.icon_trims,
        cfg.max_widths,
        cfg.min_widths,
        cfg.min_pop_width,
        cfg.idle_roi_extra_width,
        cfg.min_requireds,
        detected,
    )

    deficit = narrow.get("food_stockpile")
    if "food_stockpile" in spans and deficit:
        prev_span = spans.get("wood_stockpile")
        next_span = spans.get("gold_stockpile")
        prev_right = prev_span[1] if prev_span else x
        next_left = next_span[0] if next_span else x + w
        left, right = spans["food_stockpile"]
        space_left = left - prev_right
        space_right = next_left - right
        expand_left = min(deficit // 2, space_left)
        expand_right = min(deficit - expand_left, space_right)
        if expand_left or expand_right:
            new_left = left - expand_left
            new_right = right + expand_right
            spans["food_stockpile"] = (new_left, new_right)
            fx, fy, fw, fh = regions["food_stockpile"]
            regions["food_stockpile"] = (
                new_left,
                fy,
                new_right - new_left,
                fh,
            )
            actual = expand_left + expand_right
            if actual >= deficit:
                narrow.pop("food_stockpile", None)
            else:
                narrow["food_stockpile"] = deficit - actual

    cache._NARROW_ROIS = set(narrow.keys())
    cache._NARROW_ROI_DEFICITS = narrow.copy()
    cache._LAST_REGION_SPANS = spans.copy()

    if cache._LAST_REGION_BOUNDS != regions:
        cache._LAST_REGION_BOUNDS = regions.copy()
        cache_obj.last_resource_values.clear()
        cache_obj.last_resource_ts.clear()

    return regions
