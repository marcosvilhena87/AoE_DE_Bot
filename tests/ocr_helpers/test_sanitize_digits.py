from script.resources.ocr.confidence import _sanitize_digits


def test_trim_long_value():
    assert _sanitize_digits("1400") == "140"


def test_keep_short_value():
    assert _sanitize_digits("800") == "800"


def test_trim_exact_thousand():
    assert _sanitize_digits("1000") == "100"
