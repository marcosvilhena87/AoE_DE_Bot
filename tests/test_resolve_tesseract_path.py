import os
from pathlib import Path
import pytest

from script.common import resolve_tesseract_path


def _make_fake_tesseract(dir_path: Path) -> Path:
    fake = dir_path / "tesseract"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    return fake


def test_resolve_from_env(monkeypatch, tmp_path):
    fake = _make_fake_tesseract(tmp_path)
    monkeypatch.setenv("TESSERACT_CMD", str(fake))
    monkeypatch.setenv("PATH", "")
    assert resolve_tesseract_path({}) == str(fake)


def test_resolve_from_config(monkeypatch, tmp_path):
    fake = _make_fake_tesseract(tmp_path)
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setenv("PATH", "")
    assert resolve_tesseract_path({"tesseract_path": str(fake)}) == str(fake)


def test_env_invalid_fallback_to_path(monkeypatch, tmp_path):
    fake = _make_fake_tesseract(tmp_path)
    monkeypatch.setenv("TESSERACT_CMD", str(tmp_path / "missing"))
    monkeypatch.setenv("PATH", str(tmp_path))
    assert resolve_tesseract_path({}) == str(fake)


def test_resolve_missing_raises(monkeypatch):
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RuntimeError):
        resolve_tesseract_path({"tesseract_path": "/invalid/path"})
