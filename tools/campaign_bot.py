"""Testing proxy for :mod:`script.common` functions.

This module exposes a thin wrapper around :mod:`script.common` so tests can
patch helper functions without duplicating the full implementation.
"""
import script.common as common

# Public API expected by the tests
HUD_ANCHOR = common.HUD_ANCHOR
HUD_TEMPLATES = {"assets/ui_minimap.png": None}

# Default helpers (can be monkeypatched in tests)
_grab_frame = common._grab_frame
find_template = common.find_template
locate_resource_panel = common.locate_resource_panel

def wait_hud(timeout=60):
    """Delegate to :func:`script.common.wait_hud` using patched helpers."""
    original_templates = common.HUD_TEMPLATES
    common.HUD_TEMPLATES = HUD_TEMPLATES
    original_grab = common._grab_frame
    original_find = common.find_template
    common._grab_frame = _grab_frame
    common.find_template = find_template
    try:
        anchor, asset = common.wait_hud(timeout)
        global HUD_ANCHOR
        HUD_ANCHOR = common.HUD_ANCHOR
        return anchor, asset
    finally:
        common._grab_frame = original_grab
        common.find_template = original_find
        common.HUD_TEMPLATES = original_templates

def _ocr_digits_better(gray):
    """Delegate to common helper; patched in tests."""
    return common._ocr_digits_better(gray)

def read_resources_from_hud():
    """Delegate to :func:`script.common.read_resources_from_hud` using patched helpers."""
    common.HUD_ANCHOR = HUD_ANCHOR
    original_locate = common.locate_resource_panel
    original_grab = common._grab_frame
    original_ocr = common._ocr_digits_better

    def wrapper(gray):
        res = _ocr_digits_better(gray)
        if isinstance(res, tuple) and len(res) == 2:
            digits, data = res
            return digits, data, None
        return res

    common.locate_resource_panel = locate_resource_panel
    common._grab_frame = _grab_frame
    common._ocr_digits_better = wrapper
    try:
        return common.read_resources_from_hud()
    finally:
        common.locate_resource_panel = original_locate
        common._grab_frame = original_grab
        common._ocr_digits_better = original_ocr
