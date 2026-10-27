"""Figures for Section 3 from the result tables in results/.

Usage:
    python scripts/make_figures.py            # writes results/fig_*.png and .pdf

Figure 2: recall, purity and false-positive rate against the share of
automated sessions: rule anchor and the three centroid/hierarchical methods
(results/sweep.csv, mean and standard deviation over benchmark seeds).
Figure 3: segment profiles of the Ward segmentation on the 100k benchmark
(seed 0): median of selected features per segment and class composition.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from trafseg import pipeline as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
COLORS = {"rule": "#eb6834", "agglo": "#2a78d6", "gmm": "#1baf7a", "kmeans": "#eda100"}
LABELS = {"rule": "rule anchor", "agglo": "Ward", "gmm": "GMM", "kmeans": "k-means"}
SERIES = ["rule", "agglo", "gmm", "kmeans"]
MAIN_METHOD = "agglo"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.6, "axes.edgecolor": "#888888", "axes.linewidth": 0.6})


def fig_bot_share() -> None:
    d = pd.read_csv(RESULTS / "sweep.csv")
    d = d[(d["volume"] == d["volume"].min()) & d["method"].isin(SERIES)]
    g = d.groupby(["bot_share", "method"])
    mean, std = g[["bot_recall", "fpr", "bot_purity"]].mean(), g[["bot_recall", "fpr", "bot_purity"]].std()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    for ax, metric, title in zip(axes, ["bot_recall", "bot_purity", "fpr"],
                                 ["Recall of automated sessions", "Purity of automated segments", "False-positive rate"]):
        for i, m in enumerate(SERIES):
            x = mean.xs(m, level="method").index.to_numpy() * 100 + (i - 1.5) * 0.6   # small offset so error bars do not overlap
            y = mean.xs(m, level="method")[metric].to_numpy()
            e = std.xs(m, level="method")[metric].to_numpy()
            ax.errorbar(x, y, yerr=e, color=COLORS[m], lw=2, marker="o", ms=4, capsize=2, elinewidth=0.8, label=LABELS[m])
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("automated sessions, %")
        ax.set_xticks([5, 10, 20, 40])
        ax.set_xlim(0, 45)
        ax.set_ylim(0, 1.05 if metric != "fpr" else 0.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.14))
    fig.savefig(RESULTS / "fig_bot_share.png", dpi=300, bbox_inches="tight")
    fig.savefig(RESULTS / "fig_bot_share.pdf", bbox_inches="tight")


def fig_profiles() -> None:
    candidates = [ROOT / "data" / "synthetic" / n for n in ("synth_100k_s0_b20_features.parquet", "synth_100k_s0_features.parquet")]
    X = pd.read_parquet(next(c for c in candidates if c.exists()))
    cfg = pl.load_pipeline_config(ROOT / "config" / "pipeline.yaml")
    Z, cols = pl.prepare(X, cfg["ablation"]["full"], cfg)
    seg = pl.segment(Z, MAIN_METHOD, cfg, 0, pl.rule_anchor(X, cfg))
    feats = ["beh_n_actions", "beh_actions_per_min", "beh_iat_cv", "beh_beacon_share", "tech_no_referrer_share",
             "beh_product_share", "beh_commerce_share", "seq_self_transition_share", "id_foreign", "id_sessions_per_client"]
    names = ["actions", "actions / min", "interval CV", "beacon share", "no-referrer share", "product share",
             "commerce share", "self-transitions", "foreign client", "sessions / client"]
    prof = pl.profiles(X, seg.labels, feats)
    comp = pd.crosstab(seg.labels, X["label_class"], normalize="index")[["human", "crawler", "scraper", "headless"]]
    order = comp["human"].sort_values(ascending=False).index
    prof, comp = prof.loc[order], comp.loc[order]
    rule_share = pd.Series(pl.rule_anchor(X, cfg)).groupby(seg.labels).mean().loc[order]

    # medians scaled 0..1 per feature for a sequential colour map; values printed in cells
    M = prof[feats].to_numpy(dtype=float)
    rng = np.nanmax(M, axis=0) - np.nanmin(M, axis=0)
    S = (M - np.nanmin(M, axis=0)) / np.where(rng == 0, 1, rng)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [10, 4.6]}, constrained_layout=True)
    ax1.imshow(S, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}".rstrip("0").rstrip(".")
            ax1.text(j, i, txt, ha="center", va="center", fontsize=7, color="#0b0b0b" if S[i, j] < 0.6 else "white")
    ax1.set_xticks(range(len(names)), names, rotation=40, ha="right")
    ax1.set_yticks(range(len(order)), [f"S{k}  ({prof.loc[k, 'share'] * 100:.0f}%)" for k in order])
    ax1.set_title("Segment profiles (median per segment)", fontsize=9, loc="left")
    ax1.grid(False)

    left = np.zeros(len(order))
    cls_colors = {"human": "#c3c2b7", "crawler": "#2a78d6", "scraper": "#eb6834", "headless": "#1baf7a"}
    for c in comp.columns:
        ax2.barh(range(len(order)), comp[c].to_numpy(), left=left, color=cls_colors[c], label=c, height=0.7,
                 edgecolor="white", linewidth=1)
        left += comp[c].to_numpy()
    for i, k in enumerate(order):
        ax2.text(1.02, i, f"rules {rule_share.loc[k]:.2f}", va="center", fontsize=7, color="#52514e")

    ax2.set_yticks(range(len(order)), [])
    ax2.set_ylim(len(order) - 0.5, -0.5)
    ax2.set_xlim(0, 1.3)
    ax2.set_xticks([0, 0.5, 1], ["0", "50", "100 %"])
    ax2.set_title("Class composition", fontsize=9, loc="left")
    ax2.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax2.grid(False)
    fig.savefig(RESULTS / "fig_profiles.png", dpi=300, bbox_inches="tight")
    fig.savefig(RESULTS / "fig_profiles.pdf", bbox_inches="tight")


if __name__ == "__main__":
    fig_bot_share()
    fig_profiles()
    print("written", RESULTS / "fig_bot_share.png", RESULTS / "fig_profiles.png")
