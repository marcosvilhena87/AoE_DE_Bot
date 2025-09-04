import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import script.screen_utils as screen_utils
import script.resources as resources


def main():
    frame = screen_utils.grab_frame()
    box, score = resources.detect_hud(frame)
    print(box, score)


if __name__ == "__main__":
    main()
