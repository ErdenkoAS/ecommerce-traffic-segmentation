"""Preprocess an EClog slice and report the counts used in Section 2.7.

Usage:
    python scripts/run_eclog.py data/raw/eclog_2019-12-11.csv

Writes the request-level and session-level tables to data/processed/ and
prints a summary. Reference values for the one-day slice 2019-12-11 with the
current configs: 81 905 action-level requests, 2 962 sessions, 597 with a bot
user-agent, 1 281 with >= 5 requests, 758 with >= 5 non-beacon actions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trafseg import preprocess as pp
from trafseg.io_eclog import read_eclog

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path, help="raw EClog file")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    args = ap.parse_args()

    raw = read_eclog(args.raw)
    print(f"requests:            {len(raw):>9,}")
    print(f"client ids:          {raw['IpId'].nunique():>9,}")
    print(f"user agents:         {raw['UserAgent'].nunique():>9,}")
    print(f"time span:           {raw['ts'].min()} .. {raw['ts'].max()}")

    rules = pp.FilterRules.from_yaml(CONFIG / "static_patterns.yaml")
    req = pp.filter_requests(raw, rules)
    print(f"after filtering:     {len(req):>9,}  ({len(req) / len(raw):.1%} of raw)")

    req = pp.type_actions(req, pp.load_typing_rules(CONFIG / "eclog_uri_types.yaml"))
    req = pp.sessionize(req)
    sessions = pp.session_table(req)
    sessions["bot_ua"] = pp.flag_bot_ua(
        sessions["user_agent"], pp.load_bot_signatures(CONFIG / "bot_ua_signatures.yaml")
    )

    print(f"sessions:            {len(sessions):>9,}")
    print(f"  bot user-agent:    {int(sessions['bot_ua'].sum()):>9,}")
    print(f"  >= 5 requests:     {int((sessions['n_requests'] >= 5).sum()):>9,}")
    print(f"  >= 5 actions:      {int((sessions['n_actions'] >= 5).sum()):>9,}")
    print("action types:")
    print(req["action"].value_counts().to_string())

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.raw.stem
    req.to_parquet(args.out / f"{stem}_requests.parquet", index=False)
    sessions.to_parquet(args.out / f"{stem}_sessions.parquet", index=False)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
