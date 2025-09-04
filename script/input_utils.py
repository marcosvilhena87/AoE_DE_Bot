import logging
import time
from contextlib import contextmanager

import pyautogui as pg

from . import screen_utils

logger = logging.getLogger(__name__)


def configure_pyautogui(pause: float = 0.05, failsafe: bool = True) -> None:
    """Configure global PyAutoGUI settings used by the bot.

    Args:
        pause (float): Delay applied after each PyAutoGUI call. Defaults to
            ``0.05`` seconds.
        failsafe (bool): Whether moving the cursor to the top-left corner
            aborts automation. Defaults to ``True``.

    Returns:
        None
    """

    pg.PAUSE = pause
    pg.FAILSAFE = failsafe  # mouse no canto sup-esq aborta instantaneamente


def _to_px(nx: float, ny: float) -> tuple[int, int]:
    """Convert normalized coordinates to pixel values.

    Normalized coordinates are fractions of the screen size where ``0`` is the
    minimum and ``1`` is the maximum. This helper is primarily used by
    functions that work with screen regions.

    Args:
        nx (float): Normalized x coordinate.
        ny (float): Normalized y coordinate.

    Returns:
        tuple[int, int]: Corresponding pixel coordinates ``(x, y)``.

    Example:
        >>> _to_px(0.5, 0.5)
        (960, 540)
    """

    W, H = screen_utils.get_screen_size()
    return int(nx * W), int(ny * H)


@contextmanager
def temporary_failsafe_disabled():
    """Temporarily disable PyAutoGUI's fail-safe.

    This context manager stores the current ``pg.FAILSAFE`` state, disables
    it for the duration of the ``with`` block, and restores the original state
    afterwards, even if an exception occurs.
    """

    failsafe_state = pg.FAILSAFE
    pg.FAILSAFE = False
    try:
        yield
    finally:
        pg.FAILSAFE = failsafe_state


def _move_cursor_safe() -> None:
    """Move the cursor to the screen centre while temporarily disabling the
    fail-safe.

    This helper ensures that automated mouse operations are not interrupted
    by the user moving the cursor to the fail-safe corner.
    """

    W, H = screen_utils.get_screen_size()
    with temporary_failsafe_disabled():
        pg.moveTo(W // 2, H // 2)


def _click_norm(nx: float, ny: float, button: str = "left") -> None:
    """Click on normalized screen coordinates.

    Args:
        nx (float): Normalized x coordinate.
        ny (float): Normalized y coordinate.
        button (str, optional): Mouse button to use (``"left"`` or
            ``"right"``). Defaults to ``"left"``.

    Returns:
        None

    Example:
        >>> _click_norm(0.5, 0.5)  # Click the centre of the screen
    """

    x, y = _to_px(nx, ny)
    try:
        pg.click(x, y, button=button)
    except pg.FailSafeException:
        logger.warning(
            "Fail-safe triggered during click at (%s, %s). Moving cursor to center.",
            x,
            y,
        )
        _move_cursor_safe()
        pg.click(x, y, button=button)


def _press_key_safe(key: str, pause: float) -> None:
    """Press a key while recovering from fail-safe interruptions.

    Args:
        key (str): Key to press as recognised by PyAutoGUI.
        pause (float): Time in seconds to pause after the key press.

    Returns:
        None

    Example:
        >>> _press_key_safe("enter", 0.1)
    """

    try:
        pg.press(key)
        time.sleep(pause)
    except pg.FailSafeException:
        logger.warning(
            "Fail-safe triggered while pressing '%s'. Moving cursor to center.",
            key,
        )
        _move_cursor_safe()
        pg.press(key)
        time.sleep(pause)
