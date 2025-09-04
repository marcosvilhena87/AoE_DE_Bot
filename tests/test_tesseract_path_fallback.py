import os
import sys
import types
import importlib


def test_tesseract_path_fallback(monkeypatch, tmp_path):
    fake_dir = tmp_path / "bin"
    fake_dir.mkdir()
    fake_tesseract = fake_dir / "tesseract"
    fake_tesseract.write_text("#!/bin/sh\nexit 0\n")
    fake_tesseract.chmod(0o755)

    monkeypatch.setenv("PATH", f"{fake_dir}{os.pathsep}" + os.environ.get("PATH", ""))
    monkeypatch.setenv("TESSERACT_CMD", "/invalid/path/tesseract")

    fake_pytesseract = types.SimpleNamespace(pytesseract=types.SimpleNamespace(tesseract_cmd=""))
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)
    sys.modules.pop("script.common", None)
    common = importlib.import_module("script.common")
    common.init_common()
    assert common.pytesseract.pytesseract.tesseract_cmd == str(fake_tesseract)
    monkeypatch.delitem(sys.modules, "script.common", raising=False)

