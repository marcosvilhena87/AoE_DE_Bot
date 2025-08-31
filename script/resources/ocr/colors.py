"""Color-based heuristics for OCR masking."""

import cv2
import numpy as np


def color_mask_sets(hsv, resource, kernel):
    """Return base and closed masks based on resource color heuristics.

    Parameters
    ----------
    hsv : ndarray
        HSV image of the ROI.
    resource : str | None
        Name of the resource being read.
    kernel : ndarray
        Structuring element for morphological operations.
    """

    if resource == "wood_stockpile":
        # Broaden the brown range to better filter the resource panel background
        # and explicitly capture light digits using white/gray masks.  Combining
        # these avoids breaking loops in numbers like "8" while still excluding
        # the brown backdrop.
        brown_mask = cv2.inRange(
            hsv, np.array([5, 50, 20]), np.array([30, 255, 220])
        )
        inv_brown = cv2.bitwise_not(brown_mask)
        white_mask = cv2.inRange(
            hsv, np.array([0, 0, 200]), np.array([180, 80, 255])
        )
        gray_mask = cv2.inRange(
            hsv, np.array([0, 0, 150]), np.array([180, 80, 220])
        )
        digit_mask = cv2.bitwise_or(white_mask, gray_mask)
        base_masks = []
        closed_masks = []
        for mask in [digit_mask, inv_brown]:
            base_masks.extend([mask, cv2.bitwise_not(mask)])
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            closed_masks.extend([closed, cv2.bitwise_not(closed)])
        return base_masks, closed_masks
    else:
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([20, 100, 180]), np.array([40, 255, 255]))
        gray_mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 50, 220]))
        digit_mask = cv2.bitwise_or(white_mask, yellow_mask)
        digit_mask = cv2.bitwise_or(digit_mask, gray_mask)

        base_masks = [digit_mask, cv2.bitwise_not(digit_mask)]
        closed = cv2.morphologyEx(digit_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        closed_masks = [closed, cv2.bitwise_not(closed)]
        return base_masks, closed_masks


__all__ = ["color_mask_sets"]
