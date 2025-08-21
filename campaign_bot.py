import time
from pathlib import Path

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

ASSETS = Path("assets")

CFG = {
    "profile": "aoe1de",
    # âncoras fixas da HUD (use 1 ou mais; recortes bem justos)
    "look_for": ["ui_minimap.png"],  # adicione "ui_resources.png" se quiser
    "threshold": 0.82,               # ajuste 0.75–0.90 conforme a qualidade do asset
    "scales": [0.88, 0.92, 0.96, 1.0, 1.04, 1.08, 1.12],
    "loop_minutes": 6,
    "debug": False,                  # True: salva imagem de debug
    # Hotkeys (ajuste aos seus bindings)
    "keys": {
        "idle_vill": ".",            # selecionar aldeão ocioso
        "build_menu": "b",           # abrir menu de construção
        "house": "e",                # tecla da "Casa" no AoE1 DE (ajuste se usa grid diferente!)
        "select_tc": "h",            # selecionar Town Center
        "train_vill": "q",           # treinar aldeão na TC
        # opcional (se quiser construir depois):
        "granary": None,             # ex.: "g"
        "storage_pit": None          # ex.: "p"
    },
    # Áreas normalizadas (0..1) — ajuste após a 1ª rodada olhando o mouse
    # Na Hunting, priorize COMIDA (caça/frutos) e um pouco de madeira para casas.
    "areas": {
        "hunt_food": (0.46, 0.70),   # região onde geralmente ficam gazelas/arbustos
        "wood":      (0.62, 0.55),   # bloco de árvores frequente
        "house_spot":(0.47, 0.72),   # onde posicionar Casa (solo livre, próximo TC)
        "granary_spot": (0.44, 0.66),    # local para Granary próximo a frutos
        "storage_spot": (0.58, 0.52),    # local para Storage Pit próximo a madeira
        "pop_box":   (0.93, 0.02, 0.05, 0.04),  # x,y,w,h normalizados da população na HUD
    },
    # Heurísticas simples
    "timers": {
        "house_interval": 45.0,      # construir casa a cada ~45s (heurístico)
        "idle_gap": 0.35,            # intervalo entre comandos de aldeões
        "loop_sleep": 0.7            # descanso curto por iteração
    }
}

# Contador interno simples da população atual
CURRENT_POP = 3


def get_current_pop() -> int:
    """Retorna população corrente.

    Implementa um contador interno que é atualizado ao treinar novos aldeões.
    """
    return CURRENT_POP

# =========================
# CAPTURA & TEMPLATE MATCH
# =========================
def _grab_frame():
    with mss() as sct:
        mon = sct.monitors[1]  # tela principal
        img = np.array(sct.grab(mon))[:, :, :3]  # BGRA -> BGR
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
    t0 = time.time()
    last_best = (-1, None)  # (score, name)
    while time.time() - t0 < timeout:
        frame = _grab_frame()
        for name in CFG["look_for"]:
            tmpl = _load_gray(ASSETS / name)
            box, score, heat = _find_template(
                frame, tmpl, threshold=CFG["threshold"], scales=CFG["scales"]
            )
            if score > last_best[0]:
                last_best = (score, name)
            if box:
                if CFG["debug"]:
                    cv2.imwrite(f"debug_hud_{name}.png", frame)
                x, y, w, h = box
                return (x, y, w, h)
        time.sleep(0.25)
    raise RuntimeError(
        f"HUD não encontrada. Melhor score={last_best[0]:.3f} no template '{last_best[1]}'. "
        "Re-capture o asset (recorte mais justo) e verifique ESCALA 100%."
    )


# =========================
# LEITURA DE POPULAÇÃO NA HUD
# =========================
def read_population_from_hud():
    """Captura a população atual e o limite máximo a partir da HUD.

    Retorna `(pop_atual, pop_limite)`. Em caso de falha, devolve `(0, 0)`.
    """
    frame = _grab_frame()
    x, y, w, h = CFG["areas"]["pop_box"]
    H, W = frame.shape[:2]
    px, py = int(x * W), int(y * H)
    pw, ph = int(w * W), int(h * H)
    roi = frame[py : py + ph, px : px + pw]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, config="--psm 7").strip()

    parts = [p for p in text.replace(" ", "").split("/") if p]
    if len(parts) >= 2:
        cur = int("".join(filter(str.isdigit, parts[0])) or 0)
        limit = int("".join(filter(str.isdigit, parts[1])) or 0)
        return cur, limit
    return 0, 0

# =========================
# AÇÕES DE JOGO
# =========================
def _screen_size():
    W, H = pg.size()
    return W, H

def _to_px(nx, ny):
    W, H = _screen_size()
    return int(nx * W), int(ny * H)

def _click_norm(nx, ny):
    x, y = _to_px(nx, ny)
    pg.click(x, y)

def select_idle_villager():
    pg.press(CFG["keys"]["idle_vill"])
    time.sleep(0.10)

def build_house():
    # Abra menu de construção e selecione a tecla configurada para "Casa"
    pg.press(CFG["keys"]["build_menu"])
    time.sleep(0.05)
    house_key = CFG["keys"]["house"]
    if house_key:
        pg.press(house_key)
        time.sleep(0.15)
        hx, hy = CFG["areas"]["house_spot"]
        _click_norm(hx, hy)

def build_granary():
    # Abre menu de construção e posiciona o "Granary" no local definido
    pg.press(CFG["keys"]["build_menu"])
    time.sleep(0.05)
    g_key = CFG["keys"].get("granary")
    if g_key:
        pg.press(g_key)
        time.sleep(0.15)
        gx, gy = CFG["areas"]["granary_spot"]
        _click_norm(gx, gy)

def build_storage_pit():
    # Abre menu de construção e posiciona o "Storage Pit" no local definido
    pg.press(CFG["keys"]["build_menu"])
    time.sleep(0.05)
    s_key = CFG["keys"].get("storage_pit")
    if s_key:
        pg.press(s_key)
        time.sleep(0.15)
        sx, sy = CFG["areas"]["storage_spot"]
        _click_norm(sx, sy)

def train_villagers(target_pop: int):
    """Fila aldeões na Town Center até atingir `target_pop`.

    Usa um contador interno simples para estimar a população atual.
    """
    global CURRENT_POP
    pg.press(CFG["keys"]["select_tc"])  # seleciona a TC
    time.sleep(0.10)
    while get_current_pop() < target_pop:
        pg.press(CFG["keys"]["train_vill"])
        CURRENT_POP += 1
        time.sleep(0.10)


def econ_loop(minutes=5):
    """Baseline para 'Hunting': prioriza comida (caça/frutos) + madeira p/ casas."""
    train_villagers(12)

    # Construções iniciais: Granary e Storage Pit
    select_idle_villager()
    build_granary()
    time.sleep(0.5)
    select_idle_villager()
    build_storage_pit()
    time.sleep(0.5)

    hunt_x, hunt_y = CFG["areas"]["hunt_food"]
    wood_x, wood_y = CFG["areas"]["wood"]
    _, limit = read_population_from_hud()
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
        current, limit = read_population_from_hud()
        global CURRENT_POP
        CURRENT_POP = current
        if current >= next_house:
            select_idle_villager()
            build_house()
            time.sleep(0.5)
            _, limit = read_population_from_hud()
            next_house = limit - 2

        time.sleep(CFG["timers"]["loop_sleep"])

# =========================
# MAIN
# =========================
def main():
    print("Entre na missão da campanha (Hunting). O script inicia quando detectar a HUD…")
    try:
        hud = wait_hud(timeout=90)
        print(f"HUD detectada em {hud}. Rodando rotina econômica…")
    except RuntimeError as e:
        print(str(e))
        print("Dando mais 25s para você ajustar a câmera/HUD (fallback)…")
        time.sleep(25)

    econ_loop(minutes=CFG["loop_minutes"])
    print("Rotina concluída.")

if __name__ == "__main__":
    main()
