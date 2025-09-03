from unittest import TestCase
from unittest.mock import patch

import script.resources as resources


class TestConsecutiveIconSpacing(TestCase):
    def test_spans_and_narrow_flags(self):
        detected = {
            "wood_stockpile": (0, 0, 10, 10),
            "food_stockpile": (50, 0, 10, 10),
            "gold_stockpile": (90, 0, 10, 10),
            "stone_stockpile": (150, 0, 10, 10),
            "population_limit": (190, 0, 10, 10),
            "idle_villager": (220, 0, 20, 10),
        }

        pad = [0] * 6
        trims = [0] * 6
        max_w = [999] * 6
        min_w = [0] * 6
        panel_left = 0
        panel_right = 300

        with patch.dict(
            resources.CFG,
            {"idle_icon_inner_trim": 2, "population_idle_padding": 6},
            clear=False,
        ):
            regions, spans, narrow = resources.compute_resource_rois(
                panel_left,
                panel_right,
                0,
                10,
                pad,
                pad,
                trims,
                max_w,
                min_w,
                0,
                0,
                detected=detected,
            )

        icon_pairs = [
            ("wood_stockpile", "food_stockpile"),
            ("food_stockpile", "gold_stockpile"),
            ("gold_stockpile", "stone_stockpile"),
            ("stone_stockpile", "population_limit"),
            ("population_limit", "idle_villager"),
        ]

        # spans stay between consecutive icons
        for cur, nxt in icon_pairs:
            span_left, span_right = spans[cur]
            cur_x, _cy, cur_w, _ch = detected[cur]
            cur_right = panel_left + cur_x + cur_w
            next_left = panel_left + detected[nxt][0]
            self.assertGreaterEqual(span_left, cur_right)
            self.assertLessEqual(span_right, next_left)

        # idle villager span remains within its icon
        idle_left, idle_right = spans["idle_villager"]
        idle_x, _iy, idle_w, _ih = detected["idle_villager"]
        self.assertGreaterEqual(idle_left, panel_left + idle_x)
        self.assertLessEqual(idle_right, panel_left + idle_x + idle_w)

        # narrow entries only when width between icons is insufficient
        min_span = 30
        expected_narrow = set()
        for cur, nxt in icon_pairs:
            cur_x, _cy, cur_w, _ch = detected[cur]
            cur_right = panel_left + cur_x + cur_w
            next_left = panel_left + detected[nxt][0]
            available = next_left - cur_right
            if available < min_span:
                expected_narrow.add(cur)
        self.assertEqual(set(narrow.keys()), expected_narrow)

