"""Build the session feature matrix from a processed request table.

Usage:
    python scripts/build_features.py data/processed/eclog_1day_requests.parquet

With --clients (the generator's client table) the account families are added
and the true class is attached as ``label_class``. Writes <stem>_features.parquet
next to the input and prints a per-family summary. Labels (``label_bot_ua``,
``label_class``) are for evaluation only; they are never part of the features.
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
    ap.add_argument("--clients", type=Path, default=None, help="<stem>_clients.parquet from run_generator.py")
    args = ap.parse_args()

    req = pd.read_parquet(args.requests)
    clients = pd.read_parquet(args.clients) if args.clients else None
    X = build_features(req, load_feature_config(CONFIG / "features.yaml"), clients)
    if clients is not None:
        ip = req.groupby("session_id")["IpId"].first()
        X["label_class"] = ip.map(clients.set_index("IpId")["class"])

    ua = req.groupby("session_id")["UserAgent"].first()
    X["label_bot_ua"] = pp.flag_bot_ua(ua, pp.load_bot_signatures(CONFIG / "bot_ua_signatures.yaml")).astype(int)

    n_labels = sum(c.startswith("label_") for c in X.columns)
    print(f"sessions: {len(X):,}   features: {X.shape[1] - n_labels}")
    for fam in FAMILIES:
        cols = select_families(X, [fam]).columns
        if len(cols):
            print(f"  {fam:<5} {len(cols):>2}  {', '.join(c.split('_', 1)[1] for c in cols)}")
    if "label_class" in X:
        print("classes:", X["label_class"].value_counts().to_dict())
    print(f"sequence-eligible sessions: {int(X['seq_valid'].sum()):,}")

    out = args.requests.with_name(args.requests.name.replace("_requests", "_features"))
    X.to_parquet(out)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
