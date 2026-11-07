from pathlib import Path

import pandas as pd

from rst_audit.checks import run_checks
from rst_audit.cli import main
from rst_audit.figures import build_all


def test_shipped_checks_pass():
    out = run_checks(strict=True)
    assert len(out) >= 50
    assert (out["status"] == "PASS").all()


def test_cli_reproduce_runs_without_data(capsys):
    assert main(["reproduce"]) == 0
    assert "checks passed" in capsys.readouterr().out


def test_figures_exist_after_build():
    names = build_all()
    root = Path(__file__).resolve().parents[1] / "figures"
    for name in names:
        for ext in ("pdf", "png"):
            path = root / f"{name}.{ext}"
            assert path.exists() and path.stat().st_size > 1000
