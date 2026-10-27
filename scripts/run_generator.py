"""Generate a synthetic benchmark log in the EClog format.

Usage:
    python scripts/run_generator.py --requests 100000 --seed 0 [--bot-share 0.2]

Writes to data/synthetic/:
    synth_<N>k_s<seed>[_b<share>].csv        request log, EClog columns
    synth_<N>k_s<seed>[_b<share>]_clients.parquet   class label and account fields per client

Then run the usual pipeline on the csv:
    python scripts/run_eclog.py data/synthetic/synth_100k_s0.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trafseg.generator import generate, load_generator_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bot-share", type=float, default=None, help="override total share of automated sessions")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic")
    args = ap.parse_args()

    cfg = load_generator_config(ROOT / "config" / "generator.yaml")
    req, clients = generate(cfg, args.requests, args.seed, args.bot_share)

    stem = f"synth_{args.requests // 1000}k_s{args.seed}"
    if args.bot_share is not None:
        stem += f"_b{int(round(args.bot_share * 100))}"
    args.out.mkdir(parents=True, exist_ok=True)
    req.to_csv(args.out / f"{stem}.csv", index=False)
    clients.to_parquet(args.out / f"{stem}_clients.parquet", index=False)

    print(f"requests: {len(req):,}   clients: {len(clients):,}")
    print(clients["class"].value_counts().rename("clients").to_string())
    print(f"written to {args.out / stem}.csv")


if __name__ == "__main__":
    main()
