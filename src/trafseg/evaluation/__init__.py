"""Metrics for one segmentation (Section 3.1).

Internal metrics describe the partition in the space it was computed in;
external metrics compare it with the true class of the benchmark. Segments
are interpreted through their majority class: a segment whose majority is
not ``human`` counts as automated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, completeness_score,
                             davies_bouldin_score, homogeneity_score, silhouette_score)


def internal_metrics(Z: np.ndarray, labels: np.ndarray, sample: int = 5000, seed: int = 0) -> dict:
    mask = labels >= 0
    if len(np.unique(labels[mask])) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan, "noise_share": float((~mask).mean())}
    Zm, lm = Z[mask], labels[mask]
    return {
        "silhouette": float(silhouette_score(Zm, lm, sample_size=min(sample, len(Zm)), random_state=seed)),
        "davies_bouldin": float(davies_bouldin_score(Zm, lm)),
        "calinski_harabasz": float(calinski_harabasz_score(Zm, lm)),
        "noise_share": float((~mask).mean()),
    }


def segment_majority(labels: np.ndarray, truth: pd.Series) -> pd.Series:
    """Majority true class per segment (noise segment -1 included)."""
    return pd.Series(truth.to_numpy()).groupby(labels).agg(lambda s: s.value_counts().idxmax())


def external_metrics(labels: np.ndarray, truth: pd.Series, bot_classes: tuple[str, ...]) -> dict:
    """ARI/homogeneity/completeness against the class, purity and FPR-analogue
    against the binary human/automated interpretation of segments."""
    t = truth.to_numpy()
    is_bot = np.isin(t, bot_classes)
    major = segment_majority(labels, truth)
    seg_is_bot = pd.Series(labels).map(major).isin(bot_classes).to_numpy()
    out = {
        "ari": float(adjusted_rand_score(t, labels)),
        "ari_binary": float(adjusted_rand_score(is_bot, labels)),
        "homogeneity": float(homogeneity_score(t, labels)),
        "completeness": float(completeness_score(t, labels)),
        # purity of automated segments: share of truly automated sessions in them
        "bot_purity": float(is_bot[seg_is_bot].mean()) if seg_is_bot.any() else np.nan,
        # FPR analogue: humans placed in automated segments
        "fpr": float(seg_is_bot[~is_bot].mean()),
        # recall of automated sessions overall and per class
        "bot_recall": float(seg_is_bot[is_bot].mean()),
    }
    for c in bot_classes:
        m = t == c
        out[f"recall_{c}"] = float(seg_is_bot[m].mean()) if m.any() else np.nan
    return out


def rule_metrics(rule: np.ndarray, truth: pd.Series, bot_classes: tuple[str, ...]) -> dict:
    """The same external quantities for the binary rule anchor."""
    t = truth.to_numpy()
    is_bot = np.isin(t, bot_classes)
    flag = rule.astype(bool)
    out = {
        "ari": float(adjusted_rand_score(t, rule)),
        "ari_binary": float(adjusted_rand_score(is_bot, rule)),
        "homogeneity": float(homogeneity_score(t, rule)),
        "completeness": float(completeness_score(t, rule)),
        "bot_purity": float(is_bot[flag].mean()) if flag.any() else np.nan,
        "fpr": float(flag[~is_bot].mean()),
        "bot_recall": float(flag[is_bot].mean()),
    }
    for c in bot_classes:
        m = t == c
        out[f"recall_{c}"] = float(flag[m].mean()) if m.any() else np.nan
    return out
