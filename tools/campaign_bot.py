import logging
import json
import time
from pathlib import Path
import os

import numpy as np
import cv2
from mss import mss
import pyautogui as pg
import pytesseract
from script.template_utils import find_template

# =========================
# CONFIGURAÇÃO
# =========================
pg.PAUSE = 0.05
pg.FAILSAFE = True  # mouse no canto sup-esq aborta instantaneamente

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

with open(ROOT / "config.json", encoding="utf-8") as cfg_file:
    CFG = json.load(cfg_file)

tesseract_cmd = CFG.get("tesseract_path") or os.environ.get("TESSERACT_CMD")
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

# Configuração de logging
logging.basicConfig(
    level=logging.DEBUG if CFG.get("verbose_logging") else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Contador interno da população atual
CURRENT_POP = 3

# Instância única do mss para reutilização
SCT = mss()
MONITOR = SCT.monitors[1]  # tela principal
# Posição detectada do HUD usada apenas como referência
HUD_ANCHOR = None


class PopulationReadError(RuntimeError):
    """Raised when population values cannot be extracted from the HUD."""

# =========================
# CAPTURA & TEMPLATE MATCH
# =========================
def _grab_frame(bbox=None):
    """Captura um frame da tela.

    Se ``bbox`` for fornecido, captura apenas a região especificada.
    Caso contrário, captura a tela inteira. A detecção do HUD
    (``HUD_ANCHOR``) serve apenas como confirmação visual e não
    restringe a área capturada.
    """
    region = bbox or MONITOR
    img = np.array(SCT.grab(region))[:, :, :3]  # BGRA -> BGR
    return img

def _load_gray(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(f"Asset não encontrado: {path}")
    return im

# Preload grayscale templates referenced in the configuration to avoid
# repeatedly reading them from disk during HUD detection.
HUD_TEMPLATES = {name: _load_gray(ROOT / name) for name in CFG.get("look_for", [])}

def wait_hud(timeout=60):
    logging.info("Aguardando HUD por até %ss...", timeout)
    t0 = time.time()
    last_best = (-1, None)  # (score, name)
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

    # 1) Tente ancorar pelo template da barra de recursos
    frame_full = _grab_frame()
    regions = locate_resource_panel(frame_full)
    roi_bbox = None
    if "population" in regions:
        x, y, w, h = regions["population"]
        roi_bbox = {"left": x, "top": y, "width": w, "height": h}
    else:
        # 2) Fallback: use areas.pop_box (normalizado à tela inteira)
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
        if abs_left < 0 or abs_top < 0 or abs_right > screen_width or abs_bottom > screen_height:
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
        # OTSU + binarização costuma funcionar bem na HUD
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        data = pytesseract.image_to_data(
            thresh,
            # PERMITA A BARRA "/" para ler "atual/limite"
            config="--psm 7 -c tessedit_char_whitelist=0123456789/",
            output_type=pytesseract.Output.DICT,
        )

        text = "".join(data.get("text", [])).replace(" ", "")
        confidences = [int(c) for c in data.get("conf", []) if c != "-1"]

        last_roi = roi
        last_thresh = thresh
        last_text = text
        last_confidences = confidences

        # Aceite "12/20", "3/5", etc.
        parts = [p for p in text.split("/") if p]
        if len(parts) >= 2 and (not confidences or min(confidences) >= conf_threshold):
            cur = int("".join(filter(str.isdigit, parts[0])) or 0)
            limit = int("".join(filter(str.isdigit, parts[1])) or 0)
            return cur, limit

        logging.debug("OCR attempt %s failed: text='%s', conf=%s", attempt + 1, text, confidences)
        time.sleep(0.1)

    logging.warning(
        "Falha ao ler população da HUD após %s tentativas; último texto='%s', conf=%s",
        retries, last_text, last_confidences
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
    """Localiza o painel de recursos e retorna sub-regiões absolutas.

    O dicionário resultante possui chaves para cada contador com valores
    ``(x, y, w, h)`` absolutos.
    """
    tmpl = HUD_TEMPLATES.get("assets/resources_population.png")
    if tmpl is None:
        return {}
    box, score, _ = find_template(
        frame, tmpl, threshold=CFG["threshold"], scales=CFG["scales"]
    )
    if not box:
        return {}
    x, y, w, h = box
    offsets = {
        "food": (0.05, 0.2, 0.1, 0.6),
        "wood": (0.21, 0.2, 0.1, 0.6),
        "gold": (0.37, 0.2, 0.1, 0.6),
        "stone": (0.53, 0.2, 0.1, 0.6),
        "population": (0.69, 0.2, 0.1, 0.6),
        "idle_villager": (0.85, 0.2, 0.1, 0.6),
    }
    regions = {}
    for name, (ox, oy, fw, fh) in offsets.items():
        regions[name] = (
            x + int(ox * w),
            y + int(oy * h),
            int(fw * w),
            int(fh * h),
        )
    return regions


def _ocr_digits_better(gray):
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    normal = thresh
    inverted = cv2.bitwise_not(thresh)
    results = []
    for mask in (normal, inverted):
        data = pytesseract.image_to_data(
            mask,
            config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789",
            output_type=pytesseract.Output.DICT,
        )
        text = "".join(data.get("text", [])).strip()
        digits = "".join(filter(str.isdigit, text))
        results.append((digits, data))
    if len(results[0][0]) >= len(results[1][0]):
        return results[0]
    return results[1]


def read_resources_from_hud():
    """Lê os valores de recursos diretamente da HUD."""
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
            names = ["wood", "food", "gold", "stone", "population", "idle_villager"]
            regions = {}
            for idx, name in enumerate(names):
                icon_trim = icon_trims[idx] if idx < len(icon_trims) else icon_trims[-1]
                left = x + int(idx * slice_w + icon_trim * slice_w)
                right_limit = x + int((idx + 1) * slice_w - right_trim * slice_w)
                width = max(18, right_limit - left)
                regions[name] = (left, top, width, height)
        else:
            W, H = _screen_size()
            margin = int(0.01 * W)
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
            names = ["wood", "food", "gold", "stone", "population", "idle_villager"]
            regions = {}
            for idx, name in enumerate(names):
                icon_trim = icon_trims[idx] if idx < len(icon_trims) else icon_trims[-1]
                left = x + int(idx * slice_w + icon_trim * slice_w)
                right_limit = x + int((idx + 1) * slice_w - right_trim * slice_w)
                width = max(10, right_limit - left)
                regions[name] = (left, top, width, height)

    results = {}
    for name, (x, y, w, h) in regions.items():
        roi = _grab_frame({"left": x, "top": y, "width": w, "height": h})
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)

        digits, data = _ocr_digits_better(gray)
        if not digits:
            logging.debug("OCR failed for %s; raw boxes=%s", name, data.get("text"))
            results[name] = None
        else:
            results[name] = int(digits)
    return results

# =========================
# AÇÕES DE JOGO
# =========================
def _screen_size():
    """Retorna as dimensões da tela principal.

    Usa as informações do monitor capturadas pelo ``mss`` em vez de
    ``pyautogui.size`` para garantir consistência entre as leituras de
    pixels e as capturas de tela.
    """
    return MONITOR["width"], MONITOR["height"]

def _to_px(nx, ny):
    W, H = _screen_size()
    return int(nx * W), int(ny * H)

def _click_norm(nx, ny):
    x, y = _to_px(nx, ny)
    try:
        pg.click(x, y)
    except pg.FailSafeException:
        logging.warning(
            "Fail-safe triggered during click at (%s, %s). Moving cursor to center.",
            x,
            y,
        )
        _move_cursor_safe()
        pg.click(x, y)

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

def select_idle_villager():
    try:
        pg.press(CFG["keys"]["idle_vill"])
        time.sleep(0.10)
    except pg.FailSafeException:
        logging.warning(
            "Fail-safe triggered while selecting idle villager. Moving cursor to center."
        )
        _move_cursor_safe()
        pg.press(CFG["keys"]["idle_vill"])
        time.sleep(0.10)

def build_house():
    # Abra menu de construção e selecione a tecla configurada para "Casa"
    _press_key_safe(CFG["keys"]["build_menu"], 0.05)
    house_key = CFG["keys"]["house"]
    if house_key:
        _press_key_safe(house_key, 0.15)
        hx, hy = CFG["areas"]["house_spot"]
        _click_norm(hx, hy)

def build_granary():
    # Abre menu de construção e posiciona o "Granary" no local definido
    _press_key_safe(CFG["keys"]["build_menu"], 0.05)
    g_key = CFG["keys"].get("granary")
    if g_key:
        _press_key_safe(g_key, 0.15)
        gx, gy = CFG["areas"]["granary_spot"]
        _click_norm(gx, gy)

def build_storage_pit():
    # Abre menu de construção e posiciona o "Storage Pit" no local definido
    _press_key_safe(CFG["keys"]["build_menu"], 0.05)
    s_key = CFG["keys"].get("storage_pit")
    if s_key:
        _press_key_safe(s_key, 0.15)
        sx, sy = CFG["areas"]["storage_spot"]
        _click_norm(sx, sy)

def train_villagers(target_pop: int):
    """Fila aldeões na Town Center até atingir `target_pop`.

    Lê a população atual diretamente da HUD após cada aldeão ser enfileirado.
    """
    global CURRENT_POP
    try:
        pg.press(CFG["keys"]["select_tc"])  # seleciona a TC
        time.sleep(0.10)
    except pg.FailSafeException:
        logging.warning(
            "Fail-safe triggered while selecting Town Center. Moving cursor to center."
        )
        _move_cursor_safe()
        pg.press(CFG["keys"]["select_tc"])
        time.sleep(0.10)
    try:
        CURRENT_POP, _ = read_population_from_hud(conf_threshold=CFG.get("ocr_conf_threshold", 60))
    except PopulationReadError as e:
        logging.error(
            "Não foi possível ler população inicial: %s. Abortando treino de aldeões.",
            e,
        )
        return


    while CURRENT_POP < target_pop:
        resources = read_resources_from_hud()
        food = resources.get("food")
        if food is None:
            logging.error("Failed to read food; stopping villager training")
            break
        if food < 50:
            logging.info(
                "Comida insuficiente (%s) para treinar aldeões.",
                food,
            )
            break
        try:
            pg.press(CFG["keys"]["train_vill"])
        except pg.FailSafeException:
            logging.warning(
                "Fail-safe triggered while training villager. Moving cursor to center.",
            )
            _move_cursor_safe()
            pg.press(CFG["keys"]["train_vill"])
        time.sleep(0.10)
        try:
            CURRENT_POP, _ = read_population_from_hud(conf_threshold=CFG.get("ocr_conf_threshold", 60))
        except PopulationReadError as e:
            logging.error(
                "Falha ao atualizar população durante treinamento: %s. Encerrando treino.",
                e,
            )
            break


def econ_loop(minutes=5):
    """Baseline para 'Hunting': prioriza comida (caça/frutos) + madeira p/ casas."""
    logging.info("Iniciando rotina econômica por %s minutos", minutes)
    train_villagers(12)

    # Construções iniciais: Granary e Storage Pit
    select_idle_villager()
    build_granary()
    logging.info("Granary posicionado")
    time.sleep(0.5)
    select_idle_villager()
    build_storage_pit()
    logging.info("Storage Pit posicionado")
    time.sleep(0.5)

    hunt_x, hunt_y = CFG["areas"]["food_stockpile"]
    wood_x, wood_y = CFG["areas"]["wood_stockpile"]
    try:
        _, limit = read_population_from_hud(conf_threshold=CFG.get("ocr_conf_threshold", 60))
    except PopulationReadError as e:
        logging.error(
            "Falha ao ler população inicial: %s. Abortando rotina econômica.",
            e,
        )
        return
    next_house = limit - 2

    t0 = time.time()
    while time.time() - t0 < minutes * 60:
        # 1) Um ocioso para COMIDA (caça/arbusto)
        select_idle_villager()
        _click_norm(hunt_x, hunt_y)
        time.sleep(CFG["timers"]["idle_gap"])

        # 2) Um ocioso para MADEIRA (garantir casas)
        select_idle_villager()
        _click_norm(wood_x, wood_y)
        time.sleep(CFG["timers"]["idle_gap"])

        # 3) Construir casa quando próximo do limite de população
        try:
            current, limit = read_population_from_hud(conf_threshold=CFG.get("ocr_conf_threshold", 60))
        except PopulationReadError as e:
            logging.error(
                "Falha ao ler população durante loop econômico: %s. Encerrando rotina.",
                e,
            )
            break
        global CURRENT_POP
        CURRENT_POP = current
        if current >= next_house:
            select_idle_villager()
            build_house()
            logging.info("Casa construída para expandir população")
            time.sleep(0.5)
            try:
                _, limit = read_population_from_hud(conf_threshold=CFG.get("ocr_conf_threshold", 60))
            except PopulationReadError as e:
                logging.error(
                    "Falha ao atualizar limite de população após construir casa: %s. Encerrando rotina.",
                    e,
                )
                break
            next_house = limit - 2

        time.sleep(CFG["timers"]["loop_sleep"])
    logging.info("Rotina econômica finalizada")

# =========================
# MAIN
# =========================
def main():
    logging.info(
        "Entre na missão da campanha (Hunting). O script inicia quando detectar a HUD…"
    )
    try:
        hud, asset = wait_hud(timeout=90)
        logging.info("HUD detectada em %s usando '%s'. Rodando rotina econômica…", hud, asset)
    except RuntimeError as e:
        logging.error(str(e))
        logging.info("Dando mais 25s para você ajustar a câmera/HUD (fallback)…")
        time.sleep(25)

    econ_loop(minutes=CFG["loop_minutes"])
    logging.info("Rotina concluída.")

if __name__ == "__main__":
    main()
