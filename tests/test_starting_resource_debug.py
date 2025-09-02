import os
import sys
import types
import numpy as np
import re
from pathlib import Path
from unittest.mock import patch


dummy_pg = types.SimpleNamespace(
    PAUSE=0,
    FAILSAFE=False,
    size=lambda: (200, 200),
    click=lambda *a, **k: None,
    moveTo=lambda *a, **k: None,
    press=lambda *a, **k: None,
)


class DummyMSS:
    monitors = [{}, {"left": 0, "top": 0, "width": 200, "height": 200}]

    def grab(self, region):
        h, w = region["height"], region["width"]
        return np.zeros((h, w, 4), dtype=np.uint8)


sys.modules.setdefault("pyautogui", dummy_pg)
sys.modules.setdefault("mss", types.SimpleNamespace(mss=lambda: DummyMSS()))

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from script.resources.reader.core import validate_starting_resources, _LAST_LOW_CONFIDENCE


def test_debug_files_written_on_deviation(tmp_path):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    rois = {"wood_stockpile": (0, 0, 10, 10)}
    current = {"wood_stockpile": 0}
    expected = {"wood_stockpile": 50}

    _LAST_LOW_CONFIDENCE.clear()

    with patch("script.resources.reader.core.ROOT", Path(tmp_path)):
        validate_starting_resources(current, expected, frame=frame, rois=rois)

    debug_dir = Path(tmp_path) / "debug"
    files = {p.name for p in debug_dir.iterdir()}
    assert any(re.match(r"resource_roi_wood_stockpile_\d+\.png", n) for n in files)
    assert any(re.match(r"resource_gray_wood_stockpile_\d+\.png", n) for n in files)
    assert any(re.match(r"resource_thresh_wood_stockpile_\d+\.png", n) for n in files)
