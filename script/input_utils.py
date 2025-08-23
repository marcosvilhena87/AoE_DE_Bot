import logging
import time

import pyautogui as pg

from . import screen_utils

pg.PAUSE = 0.05
pg.FAILSAFE = True  # mouse no canto sup-esq aborta instantaneamente

def _screen_size():
    return screen_utils.MONITOR["width"], screen_utils.MONITOR["height"]

def _to_px(nx, ny):
    W, H = _screen_size()
    return int(nx * W), int(ny * H)

def _move_cursor_safe():
    W, H = _screen_size()
    failsafe_state = pg.FAILSAFE
    pg.FAILSAFE = False
    pg.moveTo(W // 2, H // 2)
    pg.FAILSAFE = failsafe_state

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
