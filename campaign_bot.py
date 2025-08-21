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
# Posição do minimapa utilizada apenas como referência
HUD_ANCHOR = None


class PopulationReadError(RuntimeError):
    """Raised when population values cannot be extracted from the HUD."""

# =========================
# CAPTURA & TEMPLATE MATCH
# =========================
def _grab_frame(bbox=None):
    """Captura um frame da tela.

    Se ``bbox`` for fornecido, captura apenas a região especificada.
    Caso contrário, captura a tela inteira. A posição do minimapa
    (``HUD_ANCHOR``) não limita a área capturada; ela é utilizada apenas
    como referência para cálculos de offset.
    """
    region = bbox or MONITOR
    img = np.array(SCT.grab(region))[:, :, :3]  # BGRA -> BGR
    return img

def _load_gray(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(f"Asset não encontrado: {path}")
    return im

def _find_template(frame_bgr, tmpl_gray, threshold=0.82, scales=None):
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h0, w0 = tmpl_gray.shape[:2]
    best = (None, -1, None)  # (box, score, heatmap)

    for s in (scales or [1.0]):
        th, tw = int(h0 * s), int(w0 * s)
        if th < 10 or tw < 10:
            continue
        if th > frame_gray.shape[0] or tw > frame_gray.shape[1]:
            logging.debug(
                "Template %sx%s exceeds frame %sx%s at scale %.2f, skipping",
                tw,
                th,
                frame_gray.shape[1],
                frame_gray.shape[0],
                s,
            )
            continue
        tmpl = cv2.resize(tmpl_gray, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(frame_gray, tmpl, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > best[1]:
            x, y = max_loc
            best = ((x, y, tw, th), max_val, res)

    box, score, heat = best
    if score >= threshold:
        return box, score, heat
    return None, score, heat

def wait_hud(timeout=60):
    logging.info("Aguardando HUD por até %ss...", timeout)
    t0 = time.time()
    last_best = (-1, None)  # (score, name)
    while time.time() - t0 < timeout:
        frame = _grab_frame()
        for name in CFG["look_for"]:
            tmpl = _load_gray(ROOT / name)
            box, score, heat = _find_template(
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
                HUD_ANCHOR = {"left": x, "top": y, "width": w, "height": h}
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
def read_population_from_hud(retries=3, conf_threshold=None):
    """Captura a população atual e o limite máximo a partir da HUD.

    Tenta realizar OCR algumas vezes para aumentar a robustez. Retorna
    ``(pop_atual, pop_limite)``. Se todas as tentativas falharem, lança
    :class:`PopulationReadError` com detalhes para auxiliar na calibração.
    """
    if conf_threshold is None:
        conf_threshold = CFG.get("ocr_conf_threshold", 60)
    x, y, w, h = CFG["areas"]["pop_box"]
    screen_width, screen_height = _screen_size()

    if HUD_ANCHOR:
        ax, ay, aw, ah = (
            HUD_ANCHOR["left"],
            HUD_ANCHOR["top"],
            HUD_ANCHOR["width"],
            HUD_ANCHOR["height"],
        )
        abs_left = int(ax + x * aw)
        abs_top = int(ay + y * ah)
        pw = int(w * aw)
        ph = int(h * ah)
    else:
        abs_left = int(x * screen_width)
        abs_top = int(y * screen_height)
        pw = int(w * screen_width)
        ph = int(h * screen_height)

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
            f"screen={screen_width}x{screen_height}, HUD_ANCHOR={HUD_ANCHOR}. "
            "Recalibrate areas.pop_box in config.json."
        )

    x1, y1, x2, y2 = abs_left, abs_top, abs_right, abs_bottom

    last_roi = None
    last_thresh = None
    last_text = ""

    for attempt in range(retries):
        frame = _grab_frame()

        roi = frame[y1:y2, x1:x2]
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
        parts = [p for p in text.split("/") if p]
        confidences = [int(c) for c in data.get("conf", []) if c != "-1"]

        last_roi = roi
        last_thresh = thresh
        last_text = text

        if len(parts) >= 2 and (not confidences or min(confidences) >= conf_threshold):
            cur = int("".join(filter(str.isdigit, parts[0])) or 0)
            limit = int("".join(filter(str.isdigit, parts[1])) or 0)
            return cur, limit

        logging.debug(
            "OCR attempt %s failed: text='%s', conf=%s", attempt + 1, text, confidences
        )
        time.sleep(0.1)

    logging.warning(
        "Falha ao ler população da HUD após %s tentativas", retries
    )
    if CFG.get("debug") and last_roi is not None:
        ts = int(time.time() * 1000)
        roi_path = ROOT / f"debug_pop_roi_{ts}.png"
        cv2.imwrite(str(roi_path), last_roi)
        thresh_path = ROOT / f"debug_pop_thresh_{ts}.png"
        cv2.imwrite(str(thresh_path), last_thresh)
        logging.info("ROI salva em %s; texto extraído: '%s'", roi_path, last_text)
        logging.debug("ROI binarizada salva em %s", thresh_path)
    raise PopulationReadError(
        f"Falha ao ler população da HUD após {retries} tentativas"
    )

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
    try:
        pg.press(CFG["keys"]["build_menu"])
        time.sleep(0.05)
    except pg.FailSafeException:
        logging.warning(
            "Fail-safe triggered while opening build menu. Moving cursor to center."
        )
        _move_cursor_safe()
        pg.press(CFG["keys"]["build_menu"])
        time.sleep(0.05)
    house_key = CFG["keys"]["house"]
    if house_key:
        try:
            pg.press(house_key)
            time.sleep(0.15)
        except pg.FailSafeException:
            logging.warning(
                "Fail-safe triggered while selecting house. Moving cursor to center."
            )
            _move_cursor_safe()
            pg.press(house_key)
            time.sleep(0.15)
        hx, hy = CFG["areas"]["house_spot"]
        _click_norm(hx, hy)

def build_granary():
    # Abre menu de construção e posiciona o "Granary" no local definido
    try:
        pg.press(CFG["keys"]["build_menu"])
        time.sleep(0.05)
    except pg.FailSafeException:
        logging.warning(
            "Fail-safe triggered while opening build menu. Moving cursor to center."
        )
        _move_cursor_safe()
        pg.press(CFG["keys"]["build_menu"])
        time.sleep(0.05)
    g_key = CFG["keys"].get("granary")
    if g_key:
        try:
            pg.press(g_key)
            time.sleep(0.15)
        except pg.FailSafeException:
            logging.warning(
                "Fail-safe triggered while selecting granary. Moving cursor to center."
            )
            _move_cursor_safe()
            pg.press(g_key)
            time.sleep(0.15)
        gx, gy = CFG["areas"]["granary_spot"]
        _click_norm(gx, gy)

def build_storage_pit():
    # Abre menu de construção e posiciona o "Storage Pit" no local definido
    try:
        pg.press(CFG["keys"]["build_menu"])
        time.sleep(0.05)
    except pg.FailSafeException:
        logging.warning(
            "Fail-safe triggered while opening build menu. Moving cursor to center."
        )
        _move_cursor_safe()
        pg.press(CFG["keys"]["build_menu"])
        time.sleep(0.05)
    s_key = CFG["keys"].get("storage_pit")
    if s_key:
        try:
            pg.press(s_key)
            time.sleep(0.15)
        except pg.FailSafeException:
            logging.warning(
                "Fail-safe triggered while selecting storage pit. Moving cursor to center."
            )
            _move_cursor_safe()
            pg.press(s_key)
            time.sleep(0.15)
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
        try:
            pg.press(CFG["keys"]["train_vill"])
        except pg.FailSafeException:
            logging.warning(
                "Fail-safe triggered while training villager. Moving cursor to center."
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

    hunt_x, hunt_y = CFG["areas"]["hunt_food"]
    wood_x, wood_y = CFG["areas"]["wood"]
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
