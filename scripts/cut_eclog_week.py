"""Cut a calendar window out of the full EClog archive (Section 2.7).

Usage:
    python scripts/cut_eclog_week.py /path/to/eclog.zip --start 2020-01-20
    python scripts/cut_eclog_week.py /path/to/eclog.csv --start 2020-02-17 --days 7

The source may be the Dataverse archive (eclog.zip), an extracted CSV, or a
directory of CSV files. Boundaries are local platform time (Europe/Warsaw),
from 00:00 of --start for --days days; the output keeps the raw EClog rows
and header unchanged, so it goes through run_eclog.py like the one-day slice.
Lines are streamed, nothing is loaded into memory.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
UNIX_EPOCH_TICKS = 621_355_968_000_000_000
HEADER = "IpId,UserId,TimeStamp,HttpMethod,Uri,HttpVersion,ResponseCode,Bytes,Referrer,UserAgent\n"
TZ = ZoneInfo("Europe/Warsaw")


def local_bounds_ticks(start: date, days: int) -> tuple[int, int]:
    t0 = datetime.combine(start, datetime.min.time(), tzinfo=TZ)
    t1 = t0 + timedelta(days=days)
    to_ticks = lambda t: int(t.timestamp()) * 10_000_000 + UNIX_EPOCH_TICKS
    return to_ticks(t0), to_ticks(t1)


def open_sources(src: Path):
    """Yield (name, text stream) for every CSV-like member of the source."""
    if src.is_dir():
        for p in sorted(src.iterdir()):
            if p.suffix.lower() in (".csv", ".tab", ".txt"):
                with open(p, encoding="utf-8", errors="replace", newline="") as f:
                    yield p.name, f
    elif src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            for info in z.infolist():
                if info.is_dir() or not info.filename.lower().endswith((".csv", ".tab", ".txt")):
                    continue
                with z.open(info) as raw:
                    yield info.filename, io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
    else:
        with open(src, encoding="utf-8", errors="replace", newline="") as f:
            yield src.name, f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path, help="eclog.zip, an extracted CSV, or a directory of CSVs")
    ap.add_argument("--start", type=date.fromisoformat, required=True, help="first local day, YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    lo, hi = local_bounds_ticks(args.start, args.days)
    out = args.out or ROOT / "data" / "raw" / f"eclog_week_{args.start.isoformat()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    seen = kept = 0
    with open(out, "w", encoding="utf-8", newline="") as w:
        w.write(HEADER)
        for name, stream in open_sources(args.source):
            for line in stream:
                if line.startswith("IpId"):
                    continue  # header of this member
                seen += 1
                # IpId and UserId never contain commas, so the third field is the timestamp
                parts = line.split(",", 3)
                if len(parts) < 4:
                    continue
                try:
                    ticks = int(parts[2])
                except ValueError:
                    continue
                if lo <= ticks < hi:
                    w.write(line)
                    kept += 1
                if seen % 5_000_000 == 0:
                    print(f"  {name}: {seen:,} lines read, {kept:,} kept ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)

    end = args.start + timedelta(days=args.days - 1)
    print(f"window {args.start} .. {end} (local time), {args.days} days")
    print(f"lines read: {seen:,}   kept: {kept:,}   ({time.time() - t0:.0f}s)")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
