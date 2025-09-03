import os
import sys
import tempfile
import json
from pathlib import Path
from unittest import TestCase

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.config_utils as config_utils


class TestLoadConfigErrors(TestCase):
    def _write_config(self, extra=None):
        base = {
            "areas": {
                "house_spot": [0, 0],
                "granary_spot": [0, 0],
                "storage_spot": [0, 0],
                "wood_spot": [0, 0],
                "food_spot": [0, 0],
                "pop_box": [0, 0, 0, 0],
            },
            "keys": {
                "idle_vill": "a",
                "build_menu": "b",
                "house": "c",
                "select_tc": "d",
                "train_vill": "e",
            },
        }
        if extra:
            base.update(extra)
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            json.dump(base, tmp)
            return tmp.name

    def test_missing_config_file(self):
        missing = Path('nonexistent_config.json')
        with self.assertRaises(RuntimeError) as ctx:
            config_utils.load_config(missing)
        self.assertIn('not found', str(ctx.exception).lower())

    def test_invalid_json_file(self):
        with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
            tmp.write('{ invalid json')
            tmp_path = tmp.name
        try:
            with self.assertRaises(RuntimeError) as ctx:
                config_utils.load_config(tmp_path)
            self.assertIn('invalid json', str(ctx.exception).lower())
        finally:
            os.remove(tmp_path)

    def test_allow_low_conf_digits_enabled(self):
        tmp_path = self._write_config({"allow_low_conf_digits": True})
        try:
            cfg = config_utils.load_config(tmp_path)
            self.assertTrue(cfg["allow_low_conf_digits"])
        finally:
            os.remove(tmp_path)

    def test_allow_low_conf_digits_disabled(self):
        tmp_path = self._write_config({"allow_low_conf_digits": False})
        try:
            cfg = config_utils.load_config(tmp_path)
            self.assertFalse(cfg["allow_low_conf_digits"])
        finally:
            os.remove(tmp_path)

    def test_allow_low_conf_digits_type_validation(self):
        tmp_path = self._write_config({"allow_low_conf_digits": "yes"})
        try:
            with self.assertRaises(RuntimeError):
                config_utils.load_config(tmp_path)
        finally:
            os.remove(tmp_path)

    def test_allow_low_conf_population_enabled(self):
        tmp_path = self._write_config({"allow_low_conf_population": True})
        try:
            with self.assertLogs(config_utils.logger, level="WARNING") as cm:
                cfg = config_utils.load_config(tmp_path)
            self.assertTrue(cfg["allow_low_conf_population"])
            self.assertTrue(
                any("allow_low_conf_population" in msg for msg in cm.output),
                "Expected warning log when allow_low_conf_population is true",
            )
        finally:
            os.remove(tmp_path)

    def test_allow_low_conf_population_disabled(self):
        tmp_path = self._write_config({"allow_low_conf_population": False})
        try:
            cfg = config_utils.load_config(tmp_path)
            self.assertFalse(cfg["allow_low_conf_population"])
        finally:
            os.remove(tmp_path)

    def test_allow_low_conf_population_type_validation(self):
        tmp_path = self._write_config({"allow_low_conf_population": "yes"})
        try:
            with self.assertRaises(RuntimeError):
                config_utils.load_config(tmp_path)
        finally:
            os.remove(tmp_path)

    def test_conflict_allow_low_conf_population_and_treat_as_failure_logs_warning(self):
        tmp_path = self._write_config(
            {"allow_low_conf_population": True, "treat_low_conf_as_failure": True}
        )
        try:
            with self.assertLogs(config_utils.logger, level="WARNING") as cm:
                cfg = config_utils.load_config(tmp_path)
            self.assertTrue(cfg["allow_low_conf_population"])
            self.assertTrue(cfg["treat_low_conf_as_failure"])
            self.assertTrue(
                any(
                    "treat_low_conf_as_failure" in msg and "allow_low_conf_population" in msg
                    for msg in cm.output
                ),
                "Expected conflict warning when both allow_low_conf_population and treat_low_conf_as_failure are true",
            )
        finally:
            os.remove(tmp_path)

    def test_profile_conflict_logs_warning(self):
        tmp_path = self._write_config(
            {
                "allow_low_conf_population": False,
                "treat_low_conf_as_failure": True,
                "profiles": {
                    "p1": {
                        "allow_low_conf_population": True,
                        "treat_low_conf_as_failure": True,
                    }
                },
            }
        )
        try:
            with self.assertLogs(config_utils.logger, level="WARNING") as cm:
                cfg = config_utils.load_config(tmp_path)
            profile_cfg = cfg["profiles"]["p1"]
            self.assertTrue(profile_cfg["allow_low_conf_population"])
            self.assertTrue(profile_cfg["treat_low_conf_as_failure"])
            self.assertTrue(
                any(
                    "profile 'p1'" in msg.lower()
                    and "treat_low_conf_as_failure" in msg
                    for msg in cm.output
                ),
                "Expected conflict warning for profile when both options are true",
            )
        finally:
            os.remove(tmp_path)
