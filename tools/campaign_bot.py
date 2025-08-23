"""Testing proxy for :mod:`script.common` functions.

This module exposes a thin wrapper around :mod:`script.common` so tests can
patch helper functions without duplicating the full implementation.
"""
import script.common as common
import script.hud as hud
import script.screen_utils as screen_utils
import script.resources as resources
import numpy as np

# Public API expected by the tests
HUD_ANCHOR = common.HUD_ANCHOR
HUD_TEMPLATES = {"assets/ui_minimap.png": np.zeros((1, 1), dtype=np.uint8)}

# Default helpers (can be monkeypatched in tests)
_grab_frame = screen_utils._grab_frame
find_template = hud.find_template
locate_resource_panel = resources.locate_resource_panel

def wait_hud(timeout=60):
    """Delegate to :func:`script.hud.wait_hud` using patched helpers."""
    original_templates = screen_utils.HUD_TEMPLATES
    screen_utils.HUD_TEMPLATES = HUD_TEMPLATES
    original_grab = screen_utils._grab_frame
    original_find = hud.find_template
    screen_utils._grab_frame = _grab_frame
    hud.find_template = find_template
    try:
        anchor, asset = hud.wait_hud(timeout)
        global HUD_ANCHOR
        HUD_ANCHOR = common.HUD_ANCHOR
        return anchor, asset
    finally:
        screen_utils._grab_frame = original_grab
        hud.find_template = original_find
        screen_utils.HUD_TEMPLATES = original_templates

def _ocr_digits_better(gray):
    """Delegate to resource helper; patched in tests."""
    return resources._ocr_digits_better(gray)

def read_resources_from_hud(required_icons=None):
    """Delegate to :func:`script.resources.read_resources_from_hud` using patched helpers."""
    common.HUD_ANCHOR = HUD_ANCHOR
    original_locate = resources.locate_resource_panel
    original_grab = screen_utils._grab_frame
    original_ocr = resources._ocr_digits_better

    def wrapper(gray):
        res = _ocr_digits_better(gray)
        if isinstance(res, tuple) and len(res) == 2:
            digits, data = res
            return digits, data, None
        return res

    resources.locate_resource_panel = locate_resource_panel
    screen_utils._grab_frame = _grab_frame
    resources._ocr_digits_better = wrapper
    try:
        return resources.read_resources_from_hud(required_icons)
    finally:
        resources.locate_resource_panel = original_locate
        screen_utils._grab_frame = original_grab
        resources._ocr_digits_better = original_ocr
