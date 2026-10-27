"""Reader for the EClog dataset (Chodak, Suchacka & Chawla, 2020)."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

COLUMNS = [
    "IpId", "UserId", "TimeStamp", "HttpMethod", "Uri", "HttpVersion",
    "ResponseCode", "Bytes", "Referrer", "UserAgent",
]

# .NET ticks: 100-ns intervals since 0001-01-01. Offset is the Unix epoch in ticks.
_UNIX_EPOCH_TICKS = 621_355_968_000_000_000
_COUNTRY_RE = re.compile(r"([A-Za-z]{2})")


def _sniff(path: Path) -> tuple[str, bool]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        sample = f.read(64 * 1024)
    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    first = sample.splitlines()[0]
    has_header = "TimeStamp" in first or "UserAgent" in first
    return dialect.delimiter, has_header


def ticks_to_datetime(ticks: pd.Series) -> pd.Series:
    seconds = (ticks.astype("int64") - _UNIX_EPOCH_TICKS) // 10_000_000
    return pd.to_datetime(seconds, unit="s", utc=True)


def read_eclog(path: str | Path) -> pd.DataFrame:
    """Load a raw EClog file into a request table.

    Returns one row per request with the original columns plus
    ``ts`` (UTC datetime) and ``country`` (two-letter code parsed from IpId).
    """
    path = Path(path)
    sep, has_header = _sniff(path)
    df = pd.read_csv(
        path,
        sep=sep,
        header=0 if has_header else None,
        names=None if has_header else COLUMNS,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        quoting=csv.QUOTE_MINIMAL,
        encoding="utf-8",
        encoding_errors="replace",
        on_bad_lines="warn",
    )
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"EClog columns not found: {missing}; got {list(df.columns)}")

    df["ts"] = ticks_to_datetime(pd.to_numeric(df["TimeStamp"], errors="coerce").fillna(0))
    df["ResponseCode"] = pd.to_numeric(df["ResponseCode"], errors="coerce").astype("Int64")
    df["Bytes"] = pd.to_numeric(df["Bytes"], errors="coerce").astype("Int64")
    df["country"] = df["IpId"].str.extract(_COUNTRY_RE, expand=False).str.upper()
    return df.sort_values("ts", kind="stable").reset_index(drop=True)
