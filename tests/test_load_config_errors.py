import os
import sys
import tempfile
from pathlib import Path
from unittest import TestCase

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import script.config_utils as config_utils


class TestLoadConfigErrors(TestCase):
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
