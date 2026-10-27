"""Method x metric matrix on the synthetic benchmark (Sections 3.1-3.3).

Usage:
    python scripts/run_benchmark.py data/synthetic/synth_100k_s0_features.parquet
        [--seeds 0 1 2] [--ablation full no_sequential ...] [--methods kmeans gmm ...]
        [--out results/benchmark_100k.csv]

Every (ablation row, method, seed) is one segmentation of the whole feature
matrix; the rule anchor is evaluated once per ablation row. Results are
appended to the CSV; the summary printed at the end is the mean over seeds.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from trafseg import pipeline as pl
from trafseg.evaluation import external_metrics, internal_metrics, rule_metrics
from trafseg.generator import BOT_CLASSES

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ["k", "stable", "silhouette", "davies_bouldin", "ari", "ari_binary", "homogeneity",
           "completeness", "bot_purity", "fpr", "bot_recall", "recall_crawler", "recall_scraper", "recall_headless"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("features", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--ablation", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-rule-constraint", action="store_true", help="select k from silhouette and stability only")
    args = ap.parse_args()

    cfg = pl.load_pipeline_config(ROOT / "config" / "pipeline.yaml")
    if args.no_rule_constraint:
        cfg["rule_segment_min_share"] = 0.0
    X = pd.read_parquet(args.features)
    truth = X["label_class"]
    rows_cfg = args.ablation or list(cfg["ablation"])
    methods = args.methods or list(cfg["methods"])
    out = args.out or ROOT / "results" / f"benchmark_{args.features.stem.replace('_features', '')}.csv"

    rows = []
    for row in rows_cfg:
        families = cfg["ablation"][row]
        Z, cols = pl.prepare(X, families, cfg)
        rule = pl.rule_anchor(X, cfg)
        rows.append({"ablation": row, "method": "rule", "seed": 0, "n_features": len(cols), "k": 2,
                     "stable": True, **rule_metrics(rule, truth, BOT_CLASSES)})
        for method in methods:
            Zm = pl.reduce(Z, method, cfg, seed=0)
            for seed in args.seeds:
                t0 = time.time()
                seg = pl.segment(Zm, method, cfg, seed, rule)
                rec = {"ablation": row, "method": method, "seed": seed, "n_features": len(cols),
                       "n_dims": Zm.shape[1], "k": seg.k, "stable": seg.stable, "stability": seg.stability,
                       **internal_metrics(Zm, seg.labels, cfg["silhouette_sample"], seed),
                       **external_metrics(seg.labels, truth, BOT_CLASSES), "seconds": round(time.time() - t0, 1)}
                rows.append(rec)
                print(f"{row:<15} {method:<8} seed={seed} k={seg.k:<3} stable={seg.stable!s:<5} "
                      f"sil={rec['silhouette']:.3f} ari={rec['ari']:.3f} purity={rec['bot_purity']:.3f} "
                      f"fpr={rec['fpr']:.3f} recall={rec['bot_recall']:.3f} ({rec['seconds']}s)")

    res = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    summary = res.groupby(["ablation", "method"])[[c for c in SUMMARY if c in res]].mean(numeric_only=True)
    print("\nmean over seeds:")
    print(summary.round(3).to_string())
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
