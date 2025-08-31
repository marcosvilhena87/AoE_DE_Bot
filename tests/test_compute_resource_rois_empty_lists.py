import json
import subprocess
import sys
import pytest

SCRIPT = """
import json
import os
import sys
os.environ['TESSERACT_CMD'] = '/usr/bin/true'
import script.resources as resources
params = json.loads(sys.argv[1])
try:
    resources.compute_resource_rois(0, 100, 0, 10, *params)
except ValueError:
    sys.exit(0)
sys.exit(1)
"""

BASE_PARAMS = {
    "pad_left": [1],
    "pad_right": [1],
    "icon_trims": [0],
    "max_widths": [10],
    "min_widths": [1],
}
ORDER = ["pad_left", "pad_right", "icon_trims", "max_widths", "min_widths"]


@pytest.mark.parametrize("missing", ORDER)
def test_empty_config_lists_raise_value_error(missing):
    params = [BASE_PARAMS[key] for key in ORDER]
    idx = ORDER.index(missing)
    params[idx] = []
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT, json.dumps(params)]
    )
    assert proc.returncode == 0
