"""Segmentation of one window (Section 2.5, Algorithm 1, steps 3-6).

``prepare`` turns the feature matrix into the standardised input, ``segment``
runs one method with number-of-segments selection, ``rule_anchor`` is the
knowledge-driven baseline, ``profiles`` describes segments in the original
features (R2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

LOG_IDENTITY = ["id_sessions_per_client", "id_ua_per_client", "id_foreign"]


def load_pipeline_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def family_columns(X: pd.DataFrame, families: list[str]) -> list[str]:
    cols = []
    for c in X.columns:
        if c.startswith("label_"):
            continue
        fam = c.split("_", 1)[0]
        if fam in families or ("id_log" in families and c in LOG_IDENTITY):
            cols.append(c)
    return cols


def prepare(X: pd.DataFrame, families: list[str], cfg: dict) -> tuple[np.ndarray, list[str]]:
    """Log-transform, standardise, impute. Returns the matrix and its column names."""
    cols = family_columns(X, families)
    M = X[cols].astype(float).copy()
    for c in cfg["log_columns"]:
        if c in M:
            M[c] = np.log1p(M[c].clip(lower=0))
    Z = StandardScaler().fit_transform(M)
    return np.nan_to_num(Z, nan=0.0), cols


def reduce(Z: np.ndarray, method: str, cfg: dict, seed: int) -> np.ndarray:
    if method in cfg["density_methods"]:
        return PCA(n_components=cfg["pca_variance"], svd_solver="full", random_state=seed).fit_transform(Z)
    return Z


# ----- clustering methods ----------------------------------------------------------

def _fit(method: str, Z: np.ndarray, k: int, seed: int, params: dict) -> np.ndarray:
    if method == "kmeans":
        return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(Z)
    if method == "gmm":
        return GaussianMixture(n_components=k, random_state=seed, **params).fit(Z).predict(Z)
    if method == "hdbscan":   # k is min_cluster_size here
        return HDBSCAN(min_cluster_size=k, min_samples=params["min_samples"]).fit_predict(Z)
    if method == "agglo":
        rng = np.random.default_rng(seed)
        n = min(params["subsample"], len(Z))
        idx = rng.choice(len(Z), n, replace=False)
        sub = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z[idx])
        return NearestCentroid().fit(Z[idx], sub).predict(Z)
    raise ValueError(method)


def _silhouette(Z: np.ndarray, labels: np.ndarray, sample: int, seed: int) -> float:
    mask = labels >= 0  # HDBSCAN noise excluded
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return float("nan")
    return float(silhouette_score(Z[mask], labels[mask], sample_size=min(sample, mask.sum()), random_state=seed))


@dataclass
class Segmentation:
    method: str
    labels: np.ndarray
    k: int
    silhouette: float
    stability: float
    stable: bool
    grid: pd.DataFrame  # one row per candidate k


def segment(Z: np.ndarray, method: str, cfg: dict, seed: int, rule: np.ndarray | None = None) -> Segmentation:
    """Algorithm 1, steps 4-6: pick k by silhouette subject to stability >= tau.

    With ``rule`` (the rule-anchor flags) a candidate partition is admissible
    only if some segment is covered by the rules (rule share >= the Algorithm 2
    threshold): the data-driven split must at least reproduce what the
    knowledge-driven layer already knows (R5). Without it, silhouette alone
    tends to pick the dominant split inside legitimate traffic.
    """
    params = cfg["methods"][method]
    grid = params.get("min_cluster_size_grid", cfg["k_grid"])
    rows = []
    for k in grid:
        runs = [_fit(method, Z, k, seed * 100 + i, params) for i in range(cfg["n_inits"] if method != "hdbscan" else 1)]
        sil = _silhouette(Z, runs[0], cfg["silhouette_sample"], seed)
        if len(runs) > 1:
            aris = [adjusted_rand_score(runs[i], runs[j]) for i in range(len(runs)) for j in range(i + 1, len(runs))]
            stab = float(np.mean(aris))
        else:
            stab = 1.0  # deterministic
        covered = float(pd.Series(rule).groupby(runs[0]).mean().max()) if rule is not None else np.nan
        rows.append({"k": k, "silhouette": sil, "stability": stab,
                     "n_segments": int(len(set(runs[0])) - (runs[0] == -1).any()),
                     "noise_share": float((runs[0] == -1).mean()), "max_rule_share": covered})
    g = pd.DataFrame(rows)
    if rule is not None:
        adm = g["max_rule_share"] >= cfg["rule_segment_min_share"]
        g = g[adm] if adm.any() else g
    # R2: at most max(k_grid) segments; candidates beyond it are not interpretable
    g = g[g["n_segments"].between(2, max(cfg["k_grid"]))] if g["n_segments"].between(2, max(cfg["k_grid"])).any() else g
    ok = g[(g["stability"] >= cfg["stability_threshold"]) & g["silhouette"].notna()]
    if len(ok):
        best, stable = ok.loc[ok["silhouette"].idxmax()], True
    else:
        best, stable = g.loc[g["stability"].idxmax()], False
    labels = _fit(method, Z, int(best["k"]), seed * 100, params)
    return Segmentation(method, labels, int(best["k"]), float(best["silhouette"]), float(best["stability"]), stable, g)


# ----- knowledge-driven layer and profiles -------------------------------------------

def rule_anchor(X: pd.DataFrame, cfg: dict) -> np.ndarray:
    """1 = automated by the rule: bot user-agent signature or all requests without referrer."""
    return ((X["label_bot_ua"] == 1) | (X["tech_no_referrer_share"] >= cfg["rule_anchor"]["no_referrer_share"])).astype(int).to_numpy()


def profiles(X: pd.DataFrame, labels: np.ndarray, cols: list[str]) -> pd.DataFrame:
    """Median of each feature per segment plus segment share (Algorithm 1, step 6)."""
    df = X[cols].copy()
    df["segment"] = labels
    out = df.groupby("segment").median()
    out.insert(0, "share", df.groupby("segment").size() / len(df))
    return out
