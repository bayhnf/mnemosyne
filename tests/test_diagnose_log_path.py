"""Regression tests for diagnostics path resolution."""

import mnemosyne.diagnose as diagnose

def test_diagnose_log_dir_honors_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    import mnemosyne.diagnose as diagnose

    assert diagnose._default_log_dir() == tmp_path / "hermes" / "mnemosyne" / "logs"


def test_read_only_diagnostics_never_creates_log_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(diagnose, "LOG_DIR", tmp_path / "hermes" / "mnemosyne" / "logs")

    summary = diagnose.run_diagnostics(read_only=True)

    assert summary["log_path"] is None
    assert not (tmp_path / "hermes").exists()
