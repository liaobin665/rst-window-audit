"""Small I/O, logging and fingerprint helpers used by every stage."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def sha1_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def df_fingerprint(df: pd.DataFrame) -> str:
    """Deterministic content fingerprint: row count plus sorted column hashes."""
    h = hashlib.sha1()
    h.update(str(len(df)).encode())
    for col in df.columns:
        values = pd.util.hash_pandas_object(df[col].astype(str), index=False).values
        h.update(col.encode())
        h.update(np.asarray(values, dtype=np.uint64).tobytes())
    return h.hexdigest()[:16]


def write_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def write_json(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)
    return path


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing input: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def read_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def upsert_csv(path: str | Path, rows, keys: list[str]) -> pd.DataFrame:
    """Atomic upsert so interrupted runs can resume without duplicate cells."""
    new = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if new.empty:
        return pd.DataFrame()
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path, encoding="utf-8")
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(subset=keys, keep="last").sort_values(keys).reset_index(drop=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    new.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)
    return new


class Logger:
    """Print to stdout and append the same lines to a log file."""

    def __init__(self, path: str | Path, echo: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self.t0 = time.time()
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)

    def __call__(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')} +{time.time() - self.t0:.0f}s] {message}"
        if self.echo:
            print(line, flush=True)
        self._fh.write(line + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
