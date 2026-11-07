"""Fail the build if identifying strings or raw data slip into the release.

The denylist is deliberately kept out of the repository: set
``RST_ANON_PATTERNS`` to a semicolon-separated list (for example in CI or in a
local shell) to enable the identifying-string checks.  Without it those two
tests are skipped, so this file itself never stores the names it looks for.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXT = {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".tex", ".csv",
            ".json", ".cff", ".cfg", ".ini", ".gitignore"}
RAW_EXT = {".duckdb", ".parquet", ".feather", ".db", ".sqlite"}


def _patterns() -> list[str]:
    raw = os.environ.get("RST_ANON_PATTERNS", "")
    return [p.strip().lower() for p in raw.split(";") if p.strip()]


def _text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        if path.suffix.lower() in TEXT_EXT or path.name == ".gitignore":
            yield path


def test_no_identifying_strings():
    patterns = _patterns()
    if not patterns:
        import pytest

        pytest.skip("RST_ANON_PATTERNS not set")
    hits = []
    for path in _text_files():
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for pat in patterns:
            if pat in text:
                hits.append(f"{path.relative_to(ROOT)}: {pat}")
    assert not hits, "identifying strings found:\n" + "\n".join(hits)


def test_pdf_metadata_has_no_author_identity():
    patterns = _patterns()
    if not patterns:
        import pytest

        pytest.skip("RST_ANON_PATTERNS not set")
    for path in (ROOT / "figures").glob("*.pdf"):
        raw = path.read_bytes().lower()
        for pat in patterns:
            assert pat.encode("utf-8") not in raw, f"{path.name}: {pat}"


def test_no_raw_trade_data_is_shipped():
    raw = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in RAW_EXT]
    assert not raw, f"raw data files present: {raw}"


def test_no_file_larger_than_10mb():
    big = [(str(p.relative_to(ROOT)), p.stat().st_size)
           for p in ROOT.rglob("*") if p.is_file() and p.stat().st_size > 10_000_000]
    assert not big, f"oversized files: {big}"


def test_local_paths_are_not_baked_into_source():
    bad = []
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "D:\\" in text or "/home/" in text or "/Users/" in text:
            bad.append(str(path.relative_to(ROOT)))
    assert not bad, f"absolute local paths in source: {bad}"


def test_citation_is_anonymous():
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "Anonymous" in text, "CITATION.cff must not name the authors"
