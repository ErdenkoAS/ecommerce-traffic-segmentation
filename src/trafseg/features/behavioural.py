"""Behavioural / temporal family: intensity, timing, beacon activity, diversity.

Timing features are computed on non-beacon actions; beacon requests are
periodic background traffic and get their own share and regularity features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _entropy(counts: pd.Series) -> float:
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _iat_stats(ts: pd.Series) -> tuple[float, float]:
    """Median and coefficient of variation of inter-arrival times, seconds."""
    if len(ts) < 2:
        return np.nan, np.nan
    d = ts.diff().dt.total_seconds().to_numpy()[1:]
    med = float(np.median(d))
    mean = d.mean()
    cv = float(d.std() / mean) if mean > 0 else 0.0
    return med, cv


def behavioural_features(req: pd.DataFrame) -> pd.DataFrame:
    g_all = req.groupby("session_id", sort=True)
    acts = req[~req["is_beacon"]]
    g = acts.groupby("session_id", sort=True)

    out = pd.DataFrame(index=g_all.size().index)
    out["beh_n_requests"] = g_all.size()
    out["beh_n_actions"] = g.size().reindex(out.index).fillna(0).astype(int)
    span = (g_all["ts"].max() - g_all["ts"].min()).dt.total_seconds()
    out["beh_duration_s"] = span
    out["beh_actions_per_min"] = out["beh_n_actions"] / (span / 60).clip(lower=1 / 60)

    iat = g["ts"].apply(_iat_stats)
    out["beh_iat_median_s"] = iat.str[0].reindex(out.index)
    out["beh_iat_cv"] = iat.str[1].reindex(out.index)

    out["beh_beacon_share"] = g_all["is_beacon"].mean()
    beacon_iat = req[req["is_beacon"]].groupby("session_id")["ts"].apply(_iat_stats)
    out["beh_beacon_iat_cv"] = beacon_iat.str[1].reindex(out.index)

    out["beh_action_entropy"] = g["action"].agg(lambda s: _entropy(s.value_counts())).reindex(out.index)
    out["beh_n_action_types"] = g["action"].nunique().reindex(out.index).fillna(0).astype(int)
    for name, types in {"product": ["product"], "commerce": ["cart", "checkout", "account"], "error": ["error"]}.items():
        share = acts["action"].isin(types).groupby(acts["session_id"]).mean()
        out[f"beh_{name}_share"] = share.reindex(out.index)
    out["beh_unique_uri_share"] = (g["Uri"].nunique() / g.size()).reindex(out.index)

    hour = g_all["ts"].min().dt.hour + g_all["ts"].min().dt.minute / 60
    out["beh_start_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["beh_start_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out
