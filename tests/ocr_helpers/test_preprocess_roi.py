import numpy as np

from script.resources import ocr


def test_grayscale_conversion():
    ocr.CFG["ocr_blur_kernel"] = 0
    roi = np.array(
        [[[10, 20, 30], [40, 50, 60]],
         [[70, 80, 90], [100, 110, 120]]],
        dtype=np.uint8,
    )
    gray = ocr.preprocess_roi(roi)
    expected = np.array([[20, 50], [80, 110]], dtype=np.uint8)
    assert gray.shape == (2, 2)
    assert np.array_equal(gray, expected)


def test_blur_applied_when_configured():
    ocr.CFG["ocr_blur_kernel"] = 3
    roi = np.ones((40, 40, 3), dtype=np.uint8) * 50
    gray = ocr.preprocess_roi(roi)
    assert np.all(gray == 1)
