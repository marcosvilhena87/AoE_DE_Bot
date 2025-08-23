"""Interactive tool to calibrate resource panel percentages.

This script captures the screen, lets the user select the resource panel and
then the digit area within the first resource slot. It outputs ``top_pct``,
``height_pct``, ``icon_trim_pct`` and ``right_trim_pct`` values compatible with
``config.json``. The selected panel and first slice are saved as PNG files for
reference.
"""

from mss import mss
import cv2
import numpy as np


def _select_roi(image, window_name):
    """Display ``image`` and return a bounding box selected by the user."""
    r = cv2.selectROI(window_name, image, showCrosshair=True)
    cv2.destroyWindow(window_name)
    return r  # (x, y, w, h)


def main():
    with mss() as sct:
        monitor = sct.monitors[1]
        screenshot = np.array(sct.grab(monitor))[:, :, :3]

    x, y, w, h = _select_roi(screenshot, "Select resource panel")
    if w == 0 or h == 0:
        print("No panel selected; aborting.")
        return

    panel = screenshot[y : y + h, x : x + w]
    slice_w = w // 6
    first_slice = panel[:, :slice_w]

    dx, dy, dw, dh = _select_roi(first_slice, "Select digits in first slot")
    if dw == 0 or dh == 0:
        print("No digit region selected; aborting.")
        return

    top_pct = dy / h
    height_pct = dh / h
    icon_trim_pct = dx / slice_w
    right_trim_pct = (slice_w - (dx + dw)) / slice_w

    print(f"top_pct: {top_pct:.4f}")
    print(f"height_pct: {height_pct:.4f}")
    print(f"icon_trim_pct: {icon_trim_pct:.4f}")
    print(f"right_trim_pct: {right_trim_pct:.4f}")

    cv2.imwrite("resource_panel_roi.png", panel)
    cv2.imwrite("resource_first_slice.png", first_slice)
    print("Saved resource_panel_roi.png and resource_first_slice.png")


if __name__ == "__main__":
    main()
