"""Image preprocessing utilities for OCR."""

import cv2

from .. import CFG


def preprocess_roi(roi):
    """Convert ROI to a blurred grayscale image."""

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    kernel = CFG.get("ocr_blur_kernel", 1)
    if (
        kernel
        and kernel > 1
        and gray.shape[0] >= 30
        and hasattr(cv2, "medianBlur")
    ):
        gray = cv2.medianBlur(gray, kernel)
    return gray


__all__ = ["preprocess_roi"]
