"""Compatibility wrapper for tests expecting :mod:`tools.campaign_bot`.

This module re-exports everything from :mod:`tools.campaign`, including
attributes prefixed with underscores, so that tests can patch internal
helpers as needed.
"""

from . import campaign as _campaign

# Copy all attributes from tools.campaign into this module's namespace.
globals().update({name: getattr(_campaign, name) for name in dir(_campaign)})

