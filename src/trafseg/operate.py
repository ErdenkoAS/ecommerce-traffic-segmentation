"""Periodic operation of the subsystem (Section 2.5, Algorithm 2).

Algorithm 1 segments one window; this module links windows together:
segment matching between consecutive windows, per-client aggregation over
the last ``n`` windows, and the status of every segment relative to the
knowledge-driven rules. Nothing here uses true classes: on the real log the
only labels are the rule flags.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def match_segments(prev: pd.DataFrame, cur: pd.DataFrame) -> dict[int, int]:
    """Map current segment ids to previous ones by profile similarity (step 2).

    Profiles are median feature vectors (``pipeline.profiles`` without the
    share column); both are standardised with the pooled scale so that heavy
    features do not dominate. Unmatched current segments (if the window has
    more segments than the previous one) get fresh ids after the old ones.
    """
    cols = [c for c in cur.columns if c != "share" and c in prev.columns]
    P, C = prev[cols].astype(float), cur[cols].astype(float)
    scale = pd.concat([P, C]).std().replace(0, 1).fillna(1)
    A = (P / scale).fillna(0).to_numpy(dtype=float)
    B = (C / scale).fillna(0).to_numpy(dtype=float)
    cost = ((B[:, None, :] - A[None, :, :]) ** 2).sum(axis=2)
    rows, cols_ = linear_sum_assignment(cost)
    mapping = {int(cur.index[r]): int(prev.index[c]) for r, c in zip(rows, cols_)}
    nxt = int(prev.index.max()) + 1
    for seg in cur.index:
        if int(seg) not in mapping:
            mapping[int(seg)] = nxt
            nxt += 1
    return mapping


def segment_status(rule_share: pd.Series, covered: float = 0.9, outside: float = 0.1) -> pd.Series:
    """'covered' / 'partial' / 'outside' relative to the rules (step 4)."""
    return pd.Series(np.where(rule_share >= covered, "covered",
                              np.where(rule_share <= outside, "outside", "partial")), index=rule_share.index)


def interpret_segments(prof: pd.DataFrame, rule_share: pd.Series, cfg: dict) -> pd.Series:
    """Boolean per segment: interpreted as automated (step 5).

    Either the rules already flag most of its sessions, or its median profile
    shows multi-action sessions without a single client-side beacon request:
    the structural property of scripted traffic the paper builds on.
    """
    by_rule = rule_share >= cfg["automated_rule_share"]
    by_profile = (prof["beh_beacon_share"] <= cfg["automated_beacon_share_max"]) & \
                 (prof["beh_n_actions"] >= cfg["automated_min_actions"])
    return (by_rule | by_profile.reindex(rule_share.index, fill_value=False)).astype(bool)


def aggregate_clients(sessions: pd.DataFrame, n_windows: int, mode_share: float = 0.5) -> pd.DataFrame:
    """Per-client aggregated segment over the last ``n_windows`` windows (step 3).

    ``sessions`` has columns client, window, segment (matched ids). The
    aggregated segment is the mode of the client's session distribution when
    its share is >= ``mode_share``, otherwise 'mixed'.
    """
    last = sorted(sessions["window"].unique())[-n_windows:]
    s = sessions[sessions["window"].isin(last)]
    counts = s.groupby(["client", "segment"]).size().rename("n").reset_index()
    total = counts.groupby("client")["n"].transform("sum")
    counts["share"] = counts["n"] / total
    top = counts.sort_values(["client", "share"], ascending=[True, False]).drop_duplicates("client")
    out = top.set_index("client")[["segment", "share"]].copy()
    out["n_sessions"] = total.groupby(counts["client"]).first()
    out["aggregated"] = np.where(out["share"] >= mode_share, out["segment"].astype(str), "mixed")
    return out[["n_sessions", "segment", "share", "aggregated"]]
