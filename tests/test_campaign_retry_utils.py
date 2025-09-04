from campaign import (
    _calculate_next_tolerance,
    _retry_ocr,
    _update_rois,
    _within_relaxed_threshold,
)


def test_calculate_next_tolerance_caps_at_max():
    assert _calculate_next_tolerance(10, 5, 20) == 15
    assert _calculate_next_tolerance(18, 5, 20) == 20


def test_retry_ocr_calls_module():
    class DummyResources:
        def __init__(self):
            self.args = None

        def gather_hud_stats(self, force_delay, required_icons, optional_icons):
            self.args = (force_delay, required_icons, optional_icons)
            return {"food": 100}, (5, 10)

    dummy = DummyResources()
    res, pop = _retry_ocr(dummy, 2, ["food"], [])
    assert res == {"food": 100}
    assert pop == (5, 10)
    assert dummy.args == (0.2, ["food"], [])


def test_update_rois_increments_deficits():
    class DummyCache:
        last_low_confidence = {"food"}
        last_no_digits = {"wood"}

    class DummyResources:
        RESOURCE_CACHE = DummyCache()
        _NARROW_ROI_DEFICITS = {}

    dummy = DummyResources()
    _update_rois(dummy, ["food", "gold", "wood"])
    assert dummy._NARROW_ROI_DEFICITS == {"food": 2, "wood": 2}


def test_within_relaxed_threshold():
    res = {"food": 100, "wood": 50}
    expected = {"food": 105, "wood": 55}
    assert _within_relaxed_threshold(res, expected, 5)
    assert not _within_relaxed_threshold(res, expected, 4)
