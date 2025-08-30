import os
import sys
import types
import numpy as np

# Stub external modules so OCR helpers can be imported without heavy deps

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


def _cvt_color(src, code):
    # Simple BGR to grayscale conversion using mean
    if getattr(src, "ndim", 0) == 3:
        return src.mean(axis=2).astype(np.uint8)
    return src


def _median_blur(src, k):
    if k > 1:
        return np.full_like(src, 1)
    return src


dummy_cv2 = types.SimpleNamespace(
    cvtColor=_cvt_color,
    resize=lambda img, *a, **k: img,
    matchTemplate=lambda *a, **k: np.zeros((1, 1), dtype=np.float32),
    minMaxLoc=lambda *a, **k: (0, 0, (0, 0), (0, 0)),
    imread=lambda *a, **k: np.zeros((1, 1), dtype=np.uint8),
    imwrite=lambda *a, **k: True,
    medianBlur=_median_blur,
    bitwise_not=lambda src: src,
    threshold=lambda src, *a, **k: (None, src),
    rectangle=lambda img, pt1, pt2, color, thickness: img,
    IMREAD_GRAYSCALE=0,
    COLOR_BGR2GRAY=0,
    INTER_LINEAR=0,
    THRESH_BINARY=0,
    THRESH_OTSU=0,
    TM_CCOEFF_NORMED=0,
)


dummy_pytesseract = types.SimpleNamespace(
    image_to_data=lambda *a, **k: {"text": [""], "conf": ["0"]},
    image_to_string=lambda *a, **k: "",
    Output=types.SimpleNamespace(DICT="dict"),
    pytesseract=types.SimpleNamespace(tesseract_cmd=""),
)

sys.modules.setdefault("pyautogui", dummy_pg)
sys.modules.setdefault("mss", types.SimpleNamespace(mss=lambda: DummyMSS()))
sys.modules.setdefault("cv2", dummy_cv2)
sys.modules.setdefault("pytesseract", dummy_pytesseract)

os.environ.setdefault("TESSERACT_CMD", "/usr/bin/true")

# Ensure project root is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
