"""Convert the official CITES bulk export into the DuckDB table used here.

The CITES Trade Database download is a single CSV whose header varies slightly
between releases (``App.`` vs ``Appendix``, ``Reporter type`` vs
``Reporter.type``).  This script normalises the header, checks that the columns
required by the analysis are present, and writes a table named ``trade`` with
the exact names the SQL in :mod:`rst_audit.panel` expects.

Usage
-----
    python scripts/prepare_cites_db.py --csv cites_trade.csv --out data/cites.duckdb

The conversion is streamed in chunks, so a full 29-million-row export does not
need to fit in memory.  Existing files are left untouched unless ``--force`` is
given.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

# Canonical names used by the analysis SQL.
REQUIRED = ["Year", "Exporter", "Taxon", "Quantity"]
OPTIONAL = [
    "Importer", "Origin", "Genus", "Family", "Order", "Class", "Appendix",
    "Reporter.type", "Purpose", "Source", "Term", "Unit",
]
NUMERIC = {"Year": "int32", "Quantity": "float64"}
CHUNK_ROWS = 500_000


def _key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


ALIASES = {
    "year": "Year",
    "exporter": "Exporter",
    "importer": "Importer",
    "origin": "Origin",
    "taxon": "Taxon",
    "genus": "Genus",
    "family": "Family",
    "order": "Order",
    "class": "Class",
    "app": "Appendix",
    "appendix": "Appendix",
    "reportertype": "Reporter.type",
    "purpose": "Purpose",
    "source": "Source",
    "term": "Term",
    "unit": "Unit",
    "quantity": "Quantity",
}


def normalise_columns(columns: list[str]) -> dict[str, str]:
    """Map raw CSV headers to canonical names; unknown columns pass through."""
    mapping = {}
    for col in columns:
        canonical = ALIASES.get(_key(col))
        mapping[col] = canonical if canonical else col.strip()
    return mapping


def _check_columns(columns: list[str]) -> None:
    missing = [c for c in REQUIRED if c not in columns]
    if missing:
        raise SystemExit(
            "missing required column(s): " + ", ".join(missing)
            + "\nexpected at least: " + ", ".join(REQUIRED)
            + "\nfound: " + ", ".join(columns)
        )
    if "Reporter.type" not in columns:
        raise SystemExit(
            "missing required column 'Reporter.type' (or 'Reporter type'); "
            "the analysis separates exporter- and importer-reported flows"
        )


def _prepare_chunk(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in NUMERIC:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Year"] = df["Year"].astype("Int64")
    df["Quantity"] = df["Quantity"].fillna(0).astype("int64")
    for col in columns:
        if col not in NUMERIC:
            # Keep NULLs in Origin: the SQL distinguishes NULL from filled values.
            df[col] = df[col].astype("object")
    ordered = REQUIRED + [c for c in OPTIONAL if c in df.columns]
    return df[ordered]


def convert(csv_path: Path, out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        raise SystemExit(f"{out_path} exists; pass --force to overwrite")

    header = pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig")
    mapping = normalise_columns(list(header.columns))
    probe = [mapping[c] for c in header.columns]
    _check_columns(probe)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(out_path))
    con.execute("DROP TABLE IF EXISTS trade")
    columns = REQUIRED + [c for c in OPTIONAL if c in probe]
    ddl = ",\n  ".join(
        f'"{c}" ' + ("INTEGER" if c == "Year" else "BIGINT" if c == "Quantity"
                     else "VARCHAR")
        for c in columns
    )
    con.execute(f"CREATE TABLE trade (\n  {ddl}\n)")

    insert = (
        f'INSERT INTO trade SELECT {", ".join(chr(34) + c + chr(34) for c in columns)} '
        "FROM chunk"
    )
    total = 0
    reader = pd.read_csv(
        csv_path,
        chunksize=CHUNK_ROWS,
        encoding="utf-8-sig",
        low_memory=False,
    )
    for i, chunk in enumerate(reader, start=1):
        chunk = chunk.rename(columns=mapping)
        chunk = _prepare_chunk(chunk, columns)
        con.register("chunk", chunk)
        con.execute(insert)
        con.unregister("chunk")
        total += len(chunk)
        print(f"[chunk {i}] +{len(chunk):,} rows (total {total:,})", flush=True)

    con.execute('CREATE INDEX IF NOT EXISTS idx_trade_year ON trade("Year")')
    con.execute('CREATE INDEX IF NOT EXISTS idx_trade_unit ON trade("Exporter", "Taxon")')
    years = con.execute('SELECT min("Year"), max("Year") FROM trade').fetchone()
    rows = con.execute("SELECT count(*) FROM trade").fetchone()[0]
    con.close()

    if rows == 0:
        raise SystemExit("converted table is empty; check the CSV and header")
    print(f"[done] {out_path}  rows={rows:,}  years={years[0]}–{years[1]}")
    print("next: CITES_DB=%s rst-audit pipeline" % out_path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path, help="official CITES bulk export (CSV)")
    ap.add_argument("--out", required=True, type=Path, help="output DuckDB file")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    args = ap.parse_args(argv)
    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")
    convert(args.csv, args.out, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
