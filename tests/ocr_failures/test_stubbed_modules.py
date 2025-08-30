import script.resources.panel as panel



def test_overlapping_rois_are_trimmed():
    regions = {
        "wood_stockpile": (0, 0, 50, 10),
        "food_stockpile": (40, 0, 50, 10),
    }
    regions = panel._remove_overlaps(regions, ["wood_stockpile", "food_stockpile"])
    assert regions["wood_stockpile"] == (0, 0, 40, 10)
    assert regions["food_stockpile"] == (40, 0, 50, 10)
    assert (
        regions["wood_stockpile"][0] + regions["wood_stockpile"][2]
        <= regions["food_stockpile"][0]
    )
