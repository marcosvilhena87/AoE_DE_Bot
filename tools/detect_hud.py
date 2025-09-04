import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import script.screen_utils as screen_utils
import script.resources as resources
import script.common as common


def main(config_path: str | Path | None = None):
    common.init_common(config_path)
    frame = screen_utils.screen_capture.grab_frame()
    box, score = resources.detect_hud(frame)
    print(box, score)


if __name__ == "__main__":
    main()
