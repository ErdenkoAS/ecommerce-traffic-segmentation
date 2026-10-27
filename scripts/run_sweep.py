"""Robustness sweep on the synthetic benchmark (Section 3.3).

For every (volume, generator seed, bot share) the script generates the log,
runs the usual preprocessing and feature extraction, and evaluates the rule
anchor and every clustering method on the full feature set. Results are
appended to one CSV so an interrupted sweep can be resumed.

Usage:
    python scripts/run_sweep.py                       # default grid, see below
    python scripts/run_sweep.py --volumes 100000 --gen-seeds 0 1 --bot-shares 0.2
    python scripts/run_sweep.py --methods kmeans gmm  # skip the slow agglomerative method

Default grid: volumes 100k/200k/400k x generator seeds 0-9 at bot share 0.2,
plus bot shares 0.05/0.10/0.40 at 100k x seeds 0-9. One clustering seed per
configuration (the generator seed); initialisation stability is already part
of the k selection.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from trafseg import pipeline as pl
from trafseg import preprocess as pp
from trafseg.evaluation import automated_flag, external_metrics, internal_metrics, rule_metrics
from trafseg.features import build_features, load_feature_config
from trafseg.generator import BOT_CLASSES, generate, load_generator_config
from trafseg.io_eclog import read_eclog

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"


def make_features(volume: int, gen_seed: int, bot_share: float, out_dir: Path,
                  gen_cfg: dict | None = None, stem: str | None = None) -> pd.DataFrame:
    """Generate -> preprocess -> features, cached on disk by configuration."""
    stem = stem or f"synth_{volume // 1000}k_s{gen_seed}_b{int(round(bot_share * 100))}"
    cached = out_dir / f"{stem}_features.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    req_raw, clients = generate(gen_cfg or load_generator_config(CONFIG / "generator.yaml"), volume, gen_seed, bot_share)
    out_dir.mkdir(parents=True, exist_ok=True)
    req_raw.to_csv(out_dir / f"{stem}.csv", index=False)
    raw = read_eclog(out_dir / f"{stem}.csv")
    req = pp.filter_requests(raw, pp.FilterRules.from_yaml(CONFIG / "static_patterns.yaml"))
    req = pp.type_actions(req, pp.load_typing_rules(CONFIG / "eclog_uri_types.yaml"))
    req = pp.sessionize(req)
    X = build_features(req, load_feature_config(CONFIG / "features.yaml"), clients)
    ip = req.groupby("session_id")["IpId"].first()
    X["label_class"] = ip.map(clients.set_index("IpId")["class"])
    ua = req.groupby("session_id")["UserAgent"].first()
    X["label_bot_ua"] = pp.flag_bot_ua(ua, pp.load_bot_signatures(CONFIG / "bot_ua_signatures.yaml")).astype(int)
    X.to_parquet(cached)
    return X


def evaluate(X: pd.DataFrame, cfg: dict, methods: list[str], seed: int, tag: dict, ablation: str = "full") -> list[dict]:
    truth = X["label_class"]
    Z, cols = pl.prepare(X, cfg["ablation"][ablation], cfg)
    tag = {**tag, "ablation": ablation, "n_features": len(cols)}
    rule = pl.rule_anchor(X, cfg)
    rule_ua = pl.rule_anchor(X, cfg, signature_only=True)
    rows = [{**tag, "method": "rule", "k": 2, "stable": True, **rule_metrics(rule, truth, BOT_CLASSES)},
            {**tag, "method": "rule_ua", "k": 2, "stable": True, **rule_metrics(rule_ua, truth, BOT_CLASSES)}]
    for method in methods:
        t0 = time.time()
        Zm = pl.reduce(Z, method, cfg, seed)
        seg = pl.segment(Zm, method, cfg, seed, rule)
        rows.append({**tag, "method": method, "k": seg.k, "stable": seg.stable, "stability": seg.stability,
                     **internal_metrics(Zm, seg.labels, cfg["silhouette_sample"], seed),
                     **external_metrics(seg.labels, truth, BOT_CLASSES), "seconds": round(time.time() - t0, 1)})
        # operational mode of Algorithm 2: rule OR segmentation, for both rule variants
        flag = automated_flag(seg.labels, truth, BOT_CLASSES)
        for name, r in (("rule", rule), ("rule_ua", rule_ua)):
            union = r.astype(bool) | flag
            rows.append({**tag, "method": f"{name}+{method}", "k": seg.k, "stable": seg.stable,
                         **rule_metrics(union.astype(int), truth, BOT_CLASSES)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", type=int, nargs="+", default=[100_000, 200_000, 400_000])
    ap.add_argument("--gen-seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--bot-shares", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.40])
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--ablation", nargs="+", default=["full"], help="rows of config/pipeline.yaml:ablation")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "sweep.csv")
    ap.add_argument("--no-rule-constraint", action="store_true", help="select k from silhouette and stability only")
    args = ap.parse_args()

    cfg = pl.load_pipeline_config(CONFIG / "pipeline.yaml")
    if args.no_rule_constraint:
        cfg["rule_segment_min_share"] = 0.0
    methods = args.methods or list(cfg["methods"])
    grid = [(v, s, 0.2) for v in args.volumes for s in args.gen_seeds]
    grid += [(args.volumes[0], s, b) for b in args.bot_shares if b != 0.2 for s in args.gen_seeds]

    done = pd.read_csv(args.out) if args.out.exists() else pd.DataFrame()
    for volume, gen_seed, bot_share in grid:
        tag = {"volume": volume, "gen_seed": gen_seed, "bot_share": bot_share}
        if len(done) and ((done["volume"] == volume) & (done["gen_seed"] == gen_seed)
                          & (done["bot_share"] == bot_share)).any():
            continue
        t0 = time.time()
        X = make_features(volume, gen_seed, bot_share, args.data)
        rows = []
        for row in args.ablation:
            rows += evaluate(X, cfg, methods, gen_seed, {**tag, "n_sessions": len(X)}, row)
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        done.to_csv(args.out, index=False)
        summary = " ".join(f"{r['method']}:{r['bot_recall']:.2f}/{r['fpr']:.3f}" for r in rows if r["ablation"] == args.ablation[0])
        print(f"{volume // 1000:>4}k seed={gen_seed} bots={bot_share:.2f} sessions={len(X):,}  "
              f"recall/fpr  {summary}  ({time.time() - t0:.0f}s)", flush=True)

    pd.set_option("display.width", 250)
    cols = ["k", "bot_purity", "fpr", "bot_recall", "recall_crawler", "recall_scraper", "recall_headless"]
    print("\nmean by volume (bot share 0.2):")
    base = done[done["ablation"] == "full"] if "ablation" in done else done
    print(base[base["bot_share"] == 0.2].groupby(["volume", "method"])[cols].mean().round(3).to_string())
    print("\nmean by bot share (smallest volume):")
    print(base[base["volume"] == args.volumes[0]].groupby(["bot_share", "method"])[cols].mean().round(3).to_string())
    if len(args.ablation) > 1:
        print("\nmean by ablation row:")
        print(done.groupby(["ablation", "method"])[cols].mean().round(3).to_string())
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
