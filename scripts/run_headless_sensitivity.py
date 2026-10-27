"""Sensitivity of the headless class to its rhythm regularity (Section 3.3).

The headless class is the one whose parameters were not observed in EClog
(Section 2.6). This script regenerates the benchmark with a grid of beacon
jitter and inter-action spread for that class and reports how often it is
separated from legitimate traffic.

Usage:
    python scripts/run_headless_sensitivity.py [--gen-seeds 0 1 2] [--methods kmeans gmm]
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from run_sweep import CONFIG, ROOT, evaluate, make_features
from trafseg import pipeline as pl
from trafseg.generator import load_generator_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=int, default=100_000)
    ap.add_argument("--gen-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--jitter", type=float, nargs="+", default=[0.0, 0.1, 0.25])
    ap.add_argument("--sigma", type=float, nargs="+", default=[0.2, 0.5, 1.0])
    ap.add_argument("--methods", nargs="+", default=["kmeans", "gmm"])
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "headless_sensitivity.csv")
    args = ap.parse_args()

    cfg = pl.load_pipeline_config(CONFIG / "pipeline.yaml")
    base = load_generator_config(CONFIG / "generator.yaml")
    rows = []
    for jitter in args.jitter:
        for sigma in args.sigma:
            gen_cfg = copy.deepcopy(base)
            gen_cfg["classes"]["headless"]["beacon"]["jitter"] = jitter
            gen_cfg["classes"]["headless"]["gap_s"]["sigma"] = sigma
            for seed in args.gen_seeds:
                stem = f"synth_{args.volume // 1000}k_s{seed}_hj{int(jitter * 100)}_hs{int(sigma * 100)}"
                X = make_features(args.volume, seed, 0.2, args.data, gen_cfg, stem)
                tag = {"jitter": jitter, "sigma": sigma, "gen_seed": seed, "n_sessions": len(X)}
                rows += evaluate(X, cfg, args.methods, seed, tag)
                last = [r for r in rows if r["gen_seed"] == seed and r["jitter"] == jitter and r["sigma"] == sigma]
                print(f"jitter={jitter:.2f} sigma={sigma:.1f} seed={seed}  " +
                      " ".join(f"{r['method']}:{r['recall_headless']:.2f}" for r in last), flush=True)
    res = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print("\nmean recall of headless sessions:")
    print(res.pivot_table(index=["jitter", "sigma"], columns="method", values="recall_headless").round(3).to_string())
    print("\nmean FPR:")
    print(res.pivot_table(index=["jitter", "sigma"], columns="method", values="fpr").round(3).to_string())
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
