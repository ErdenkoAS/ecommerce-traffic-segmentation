"""Build the session feature matrix from a processed request table.

Usage:
    python scripts/build_features.py data/processed/eclog_1day_requests.parquet

Writes <stem>_features.parquet next to the input and prints a per-family
summary. Weak labels (bot user-agent) are attached as a separate column
``label_bot_ua`` for evaluation only; they are never part of the features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trafseg import preprocess as pp
from trafseg.features import FAMILIES, build_features, load_feature_config, select_families

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("requests", type=Path, help="<stem>_requests.parquet from run_eclog.py")
    args = ap.parse_args()

    req = pd.read_parquet(args.requests)
    X = build_features(req, load_feature_config(CONFIG / "features.yaml"))

    ua = req.groupby("session_id")["UserAgent"].first()
    X["label_bot_ua"] = pp.flag_bot_ua(ua, pp.load_bot_signatures(CONFIG / "bot_ua_signatures.yaml")).astype(int)

    print(f"sessions: {len(X):,}   features: {X.shape[1] - 1}")
    for fam in FAMILIES:
        cols = select_families(X, [fam]).columns
        print(f"  {fam:<5} {len(cols):>2}  {', '.join(c.split('_', 1)[1] for c in cols)}")
    print(f"sequence-eligible sessions: {int(X['seq_valid'].sum()):,}")

    out = args.requests.with_name(args.requests.name.replace("_requests", "_features"))
    X.to_parquet(out)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
