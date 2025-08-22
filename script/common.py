import logging
import json
import time
from pathlib import Path
import os
import re
from dataclasses import dataclass

import numpy as np
import cv2
from mss import mss
import pyautogui as pg
import pytesseract
from .template_utils import find_template

# =========================
# CONFIGURAÇÃO
# =========================
pg.PAUSE = 0.05
pg.FAILSAFE = True  # mouse no canto sup-esq aborta instantaneamente

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


def validate_config(cfg: dict) -> None:
    """Ensure mandatory config sections and coordinates are present.

    Raises
    ------
    RuntimeError
        If required sections or fields are missing, suggesting how to fix it.
    """

    if "areas" not in cfg:
        raise RuntimeError(
            "Missing mandatory 'areas' section in config.json. "
            "Copy values from config.sample.json or run the calibration tools."
        )

    required_coords = [
        "house_spot",
        "granary_spot",
        "storage_spot",
        "wood_spot",
        "food_spot",
        "pop_box",
    ]
    areas = cfg.get("areas", {})
    missing = [name for name in required_coords if name not in areas]
    if missing:
        raise RuntimeError(
            "Missing required coordinate(s) in 'areas': "
            + ", ".join(missing)
            + ". Copy them from config.sample.json or run the calibration tools."
        )

with open(ROOT / "config.json", encoding="utf-8") as cfg_file:
    CFG = json.load(cfg_file)
    
validate_config(CFG)

tesseract_cmd = os.environ.get("TESSERACT_CMD") or CFG.get("tesseract_path")
if tesseract_cmd:
    tesseract_path = Path(tesseract_cmd)
    if not (tesseract_path.is_file() and os.access(tesseract_path, os.X_OK)):
        raise RuntimeError(
            "Invalid Tesseract OCR path. Install Tesseract or update 'tesseract_path' in config.json."
        )
    pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)

logging.basicConfig(
    level=logging.DEBUG if CFG.get("verbose_logging") else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Contadores internos de população
CURRENT_POP = 0
POP_CAP = 0
TARGET_POP = 0

# Instância única do mss para reutilização
SCT = mss()
MONITOR = SCT.monitors[1]  # tela principal
# Posição detectada do HUD usada apenas como referência
HUD_ANCHOR = None


class PopulationReadError(RuntimeError):
    """Raised when population values cannot be extracted from the HUD."""


class ResourceReadError(RuntimeError):
    """Raised when resource values cannot be extracted from the HUD."""


@dataclass
class ScenarioInfo:
    starting_villagers: int = 0
    population_limit: int = 0
    objective_villagers: int = 0


def parse_scenario_info(path: str) -> ScenarioInfo:
    """Parse basic scenario information from a text file."""
    info = ScenarioInfo()
    in_objectives = False
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    in_objectives = False
                    continue
                lower = line.lower()
                if lower.startswith("population limit"):
                    m = re.search(r"(\d+)", line)
                    if m:
                        info.population_limit = int(m.group(1))
                elif lower.startswith("starting units"):
                    m = re.search(r"(\d+)\s+villagers", line, re.IGNORECASE)
                    if m:
                        info.starting_villagers = int(m.group(1))
                elif lower.startswith("objectives"):
                    in_objectives = True
                    continue
                elif in_objectives:
                    m = re.search(r"(\d+)\s+villagers", line, re.IGNORECASE)
                    if m:
                        info.objective_villagers = int(m.group(1))
                        in_objectives = False
    except FileNotFoundError:
        logging.error("Scenario file not found: %s", path)
    return info

# =========================
# CAPTURA & TEMPLATE MATCH
# =========================

def _grab_frame(bbox=None):
    """Captura um frame da tela."""
    region = bbox or MONITOR
    img = np.array(SCT.grab(region))[:, :, :3]  # BGRA -> BGR
    return img

def _load_gray(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(f"Asset não encontrado: {path}")
    return im

HUD_TEMPLATES = {name: _load_gray(ROOT / name) for name in CFG.get("look_for", [])}

def wait_hud(timeout=60):
    logging.info("Aguardando HUD por até %ss...", timeout)
    t0 = time.time()
    last_best = (-1, None)
    while time.time() - t0 < timeout:
        frame = _grab_frame()
        for name, tmpl in HUD_TEMPLATES.items():
            box, score, heat = find_template(
                frame, tmpl, threshold=CFG["threshold"], scales=CFG["scales"]
            )
            if score > last_best[0]:
                last_best = (score, name)
            if box:
                if CFG["debug"]:
                    cv2.imwrite(f"debug_hud_{name}.png", frame)
                x, y, w, h = box
                logging.info("HUD detectada com template '%s'", name)
                global HUD_ANCHOR
                HUD_ANCHOR = {
                    "left": x,
                    "top": y,
                    "width": w,
                    "height": h,
                    "asset": name,
                }
                return HUD_ANCHOR, name
        time.sleep(0.25)
    logging.error(
        "HUD não encontrada. Melhor score=%.3f no template '%s'. Re-capture o asset e verifique ESCALA 100%%.",
        last_best[0],
        last_best[1],
    )
    raise RuntimeError(
        f"HUD não encontrada. Melhor score={last_best[0]:.3f} no template '{last_best[1]}'. "
        "Re-capture o asset (recorte mais justo) e verifique ESCALA 100%."
    )

# =========================
# LEITURA DE POPULAÇÃO NA HUD
# =========================

def read_population_from_hud(retries=3, conf_threshold=None, save_failed_roi=False):
    if conf_threshold is None:
        conf_threshold = CFG.get("ocr_conf_threshold", 60)

    frame_full = _grab_frame()
    regions = locate_resource_panel(frame_full)
    roi_bbox = None
    if "population" in regions:
        x, y, w, h = regions["population"]
        roi_bbox = {"left": x, "top": y, "width": w, "height": h}
    else:
        x, y, w, h = CFG["areas"]["pop_box"]
        screen_width, screen_height = _screen_size()
        abs_left = int(x * screen_width)
        abs_top = int(y * screen_height)
        pw = int(w * screen_width)
        ph = int(h * screen_height)
        if pw <= 0 or ph <= 0:
            raise PopulationReadError(
                f"Population ROI has non-positive dimensions after scaling: width={pw}, height={ph}. "
                "Recalibrate areas.pop_box in config.json."
            )
        abs_right = abs_left + pw
        abs_bottom = abs_top + ph
        if (
            abs_left < 0
            or abs_top < 0
            or abs_right > screen_width
            or abs_bottom > screen_height
        ):
            raise PopulationReadError(
                "Population ROI out of screen bounds: "
                f"left={abs_left}, top={abs_top}, width={pw}, height={ph}, "
                f"screen={screen_width}x{screen_height}. "
                "Recalibrate areas.pop_box em config.json ou use a âncora por template."
            )
        roi_bbox = {"left": abs_left, "top": abs_top, "width": pw, "height": ph}

    last_roi = None
    last_thresh = None
    last_text = ""
    last_confidences = []

    for attempt in range(retries):
        roi = _grab_frame(roi_bbox)
        if roi.size == 0:
            logging.warning("Population ROI has zero size")
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        data = pytesseract.image_to_data(
            thresh,
            config="--psm 7 -c tessedit_char_whitelist=0123456789/",
            output_type=pytesseract.Output.DICT,
        )
        text = "".join(data.get("text", [])).replace(" ", "")
        confidences = [int(c) for c in data.get("conf", []) if c != "-1"]
        last_roi = roi
        last_thresh = thresh
        last_text = text
        last_confidences = confidences
        parts = [p for p in text.split("/") if p]
        if len(parts) >= 2 and (not confidences or min(confidences) >= conf_threshold):
            cur = int("".join(filter(str.isdigit, parts[0])) or 0)
            limit = int("".join(filter(str.isdigit, parts[1])) or 0)
            return cur, limit
        logging.debug("OCR attempt %s failed: text='%s', conf=%s", attempt + 1, text, confidences)
        time.sleep(0.1)

    logging.warning(
        "Falha ao ler população da HUD após %s tentativas; último texto='%s', conf=%s",
        retries,
        last_text,
        last_confidences,
    )
    if (CFG.get("debug") or save_failed_roi) and last_roi is not None:
        ts = int(time.time() * 1000)
        cv2.imwrite(str(ROOT / f"debug_pop_roi_{ts}.png"), last_roi)
        cv2.imwrite(str(ROOT / f"debug_pop_thresh_{ts}.png"), last_thresh)
        logging.info("ROI salva; texto extraído: '%s'; conf=%s", last_text, last_confidences)

    raise PopulationReadError(
        f"Falha ao ler população da HUD após {retries} tentativas. Texto='{last_text}', confs={last_confidences}"
    )

# =========================
# RECURSOS NA HUD
# =========================

def locate_resource_panel(frame):
    """Locate the resource panel and return bounding boxes for each value."""

    tmpl = HUD_TEMPLATES.get("assets/resources.png")
    if tmpl is None:
        return {}

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
        logging.warning(
            "Resource panel template not matched; score=%.3f", score
        )
        _save_debug(frame, heat)
        fallback = CFG.get("threshold_fallback")
        if fallback is not None:
            box, score, heat = find_template(
                frame, tmpl, threshold=fallback, scales=CFG["scales"]
            )
            if not box:
                logging.warning(
                    "Resource panel template not matched with fallback; score=%.3f",
                    score,
                )
                _save_debug(frame, heat)
                return {}
        else:
            return {}

    x, y, w, h = box
    panel_gray = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)

    res_cfg = CFG.get("resource_panel", {})
    match_threshold = res_cfg.get("match_threshold", 0.8)
    scales = res_cfg.get("scales", CFG.get("scales", [1.0]))
    pad_left = res_cfg.get("roi_padding_left", 0)
    pad_right = res_cfg.get("roi_padding_right", 0)
    min_width = res_cfg.get("min_width", 18)
    top_pct = res_cfg.get("top_pct", 0.08)
    height_pct = res_cfg.get("height_pct", 0.84)

    icons_dir = ASSETS / "icons"
    names = ["wood", "food", "gold", "stone", "population", "idle_villager"]
    detections = []
    for name in names:
        icon = cv2.imread(str(icons_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if icon is None:
            logging.warning("Icon asset missing: %s", name)
            continue
        best = (-1, None, None)  # score, loc, (w,h)
        for scale in scales:
            icon_scaled = cv2.resize(icon, None, fx=scale, fy=scale)
            if (
                icon_scaled.shape[0] > panel_gray.shape[0]
                or icon_scaled.shape[1] > panel_gray.shape[1]
            ):
                continue
            result = cv2.matchTemplate(panel_gray, icon_scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best[0]:
                best = (max_val, max_loc, icon_scaled.shape[::-1])
        if best[0] >= match_threshold and best[1] is not None:
            (bw, bh) = best[2]
            detections.append((name, best[1][0], best[1][1], bw, bh))
        else:
            logging.warning("Icon '%s' not matched; score=%.3f", name, best[0])

    detections.sort(key=lambda d: d[1])  # sort by x position
    top = y + int(top_pct * h)
    height = int(height_pct * h)
    regions = {}
    for idx, (name, xi, yi, wi, hi) in enumerate(detections):
        if name == "idle_villager":
            left = x + xi + pad_left
            right = x + xi + wi - pad_right
            top_i = y + yi
            height_i = hi
            width = max(1, right - left)
        else:
            left = x + xi + wi + pad_left
            if idx + 1 < len(detections):
                right = x + detections[idx + 1][1] - pad_right
            else:
                right = x + w - pad_right
            width = max(min_width, right - left)
            top_i = top
            height_i = height
        regions[name] = (left, top_i, width, height_i)

    return regions


def _ocr_digits_better(gray):
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks = [thresh, cv2.bitwise_not(thresh)]
    results = []
    for mask in masks:
        data = pytesseract.image_to_data(
            mask,
            config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789",
            output_type=pytesseract.Output.DICT,
        )
        text = "".join(data.get("text", [])).strip()
        digits = "".join(filter(str.isdigit, text))
        results.append((digits, data, mask))
    results.sort(key=lambda r: len(r[0]), reverse=True)
    return results[0]

def read_resources_from_hud():
    frame = _grab_frame()
    regions = locate_resource_panel(frame)

    if not regions and HUD_ANCHOR:
        if HUD_ANCHOR.get("asset") == "assets/resources.png":
            x = HUD_ANCHOR["left"]
            y = HUD_ANCHOR["top"]
            w = HUD_ANCHOR["width"]
            h = HUD_ANCHOR["height"]

            slice_w = w / 6
            res_cfg = CFG.get("resource_panel", {})
            top_pct = res_cfg.get("top_pct", 0.08)
            height_pct = res_cfg.get("height_pct", 0.84)
            icon_trims = res_cfg.get("icon_trim_pct", 0.18)
            if not isinstance(icon_trims, (list, tuple)):
                icon_trims = [icon_trims] * 6
            right_trim = res_cfg.get("right_trim_pct", 0.02)

            top = y + int(top_pct * h)
            height = int(height_pct * h)
            names = [
                "wood",
                "food",
                "gold",
                "stone",
                "population",
                "idle_villager",
            ]
            regions = {}
            for idx, name in enumerate(names):
                icon_trim = icon_trims[idx] if idx < len(icon_trims) else icon_trims[-1]
                left = x + int(idx * slice_w + icon_trim * slice_w)
                right_limit = x + int((idx + 1) * slice_w - right_trim * slice_w)
                width = max(18, right_limit - left)
                regions[name] = (left, top, width, height)
        else:
            # Fallback: estimate the resource bar position from the previously
            # detected HUD anchor.  The anchor is assumed to sit at the top of the
            # screen and the resource panel spans to its right.  The original
            # template ``resources.png`` measured 568x59 px on a
            # 1920x1080 display.  Offsets are configurable via
            # ``resource_panel`` entries in ``config.json``.
            W, H = _screen_size()
            margin = int(0.01 * W)  # ~1% horizontal gap between anchor and panel
            panel_w = int(568 / 1920 * W)
            panel_h = int(59 / 1080 * H)
            x = HUD_ANCHOR["left"] + HUD_ANCHOR["width"] + margin
            y = HUD_ANCHOR["top"]

            res_cfg = CFG.get("resource_panel", {})
            top_pct = res_cfg.get("anchor_top_pct", 0.15)
            height_pct = res_cfg.get("anchor_height_pct", 0.70)
            icon_trims = res_cfg.get(
                "anchor_icon_trim_pct", [0.42, 0.42, 0.35, 0.35, 0.35, 0.35]
            )
            if not isinstance(icon_trims, (list, tuple)):
                icon_trims = [icon_trims] * 6
            right_trim = res_cfg.get("anchor_right_trim_pct", 0.02)

            slice_w = panel_w / 6
            top = y + int(top_pct * panel_h)
            height = int(height_pct * panel_h)
            names = [
                "wood",
                "food",
                "gold",
                "stone",
                "population",
                "idle_villager",
            ]
            regions = {}
            for idx, name in enumerate(names):
                icon_trim = icon_trims[idx] if idx < len(icon_trims) else icon_trims[-1]
                left = x + int(idx * slice_w + icon_trim * slice_w)
                right_limit = x + int((idx + 1) * slice_w - right_trim * slice_w)
                width = max(10, right_limit - left)
                regions[name] = (left, top, width, height)

    if not regions:
        raise ResourceReadError("Resource bar not located on HUD")

    results = {}
    for name, (x, y, w, h) in regions.items():
        roi = _grab_frame({"left": x, "top": y, "width": w, "height": h})
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)

        digits, data, mask = _ocr_digits_better(gray)
        if not digits:
            logging.debug("OCR failed for %s; raw boxes=%s", name, data.get("text"))
            debug_cfg = CFG.get("resource_panel", {}).get("debug_failed_ocr")
            if CFG.get("debug") or debug_cfg:
                debug_dir = ROOT / "debug"
                debug_dir.mkdir(exist_ok=True)
                ts = int(time.time() * 1000)
                cv2.imwrite(str(debug_dir / f"resource_{name}_roi_{ts}.png"), roi)
                cv2.imwrite(str(debug_dir / f"resource_{name}_thresh_{ts}.png"), mask)
            results[name] = None
        else:
            results[name] = int(digits)

    if all(v is None for v in results.values()):
        debug_cfg = CFG.get("resource_panel", {}).get("debug_failed_ocr")
        if CFG.get("debug") or debug_cfg:
            debug_dir = ROOT / "debug"
            debug_dir.mkdir(exist_ok=True)
            ts = int(time.time() * 1000)
            cv2.imwrite(str(debug_dir / f"resource_panel_fail_{ts}.png"), frame)
        tess_path = pytesseract.pytesseract.tesseract_cmd
        raise ResourceReadError(
            "OCR failed to read resource values "
            f"(regions={regions}, tesseract_cmd={tess_path})"
        )

    return results

# =========================
# AÇÕES DE JOGO BÁSICAS
# =========================

def _screen_size():
    return MONITOR["width"], MONITOR["height"]

def _to_px(nx, ny):
    W, H = _screen_size()
    return int(nx * W), int(ny * H)

def _click_norm(nx, ny, button="left"):
    x, y = _to_px(nx, ny)
    try:
        pg.click(x, y, button=button)
    except pg.FailSafeException:
        logging.warning(
            "Fail-safe triggered during click at (%s, %s). Moving cursor to center.",
            x,
            y,
        )
        _move_cursor_safe()
        pg.click(x, y, button=button)

def _move_cursor_safe():
    W, H = _screen_size()
    failsafe_state = pg.FAILSAFE
    pg.FAILSAFE = False
    pg.moveTo(W // 2, H // 2)
    pg.FAILSAFE = failsafe_state

def _press_key_safe(key, pause):
    try:
        pg.press(key)
        time.sleep(pause)
    except pg.FailSafeException:
        logging.warning(
            "Fail-safe triggered while pressing '%s'. Moving cursor to center.",
            key,
        )
        _move_cursor_safe()
        pg.press(key)
        time.sleep(pause)
