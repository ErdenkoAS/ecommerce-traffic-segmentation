"""Sequential family: the session as a string of action types.

Beacon requests are excluded from the string (they are background traffic,
not user actions). Sessions shorter than ``min_actions`` get NaN features
and seq_valid = 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = [
    "seq_markov_entropy",       # entropy rate of the first-order chain, bits
    "seq_self_transition_share",
    "seq_dominant_transition_share",
    "seq_bigram_diversity",     # distinct bigrams / number of transitions
    "seq_max_run_share",        # longest run of one action type / length
]


def sequence_features(actions: np.ndarray) -> list[float]:
    n = len(actions)
    a, b = actions[:-1], actions[1:]
    pairs = pd.MultiIndex.from_arrays([a, b])
    counts = pairs.value_counts()
    n_trans = n - 1

    # entropy rate: sum_i p(i) * H(next | i)
    src_counts = pd.Series(a).value_counts()
    h = 0.0
    for src, c in src_counts.items():
        p_next = counts[counts.index.get_level_values(0) == src] / c
        h += (c / n_trans) * float(-(p_next * np.log2(p_next)).sum())

    self_share = float((a == b).mean())
    dominant = float(counts.iloc[0] / n_trans)
    diversity = float(len(counts) / n_trans)

    change = np.flatnonzero(a != b)
    bounds = np.concatenate([[-1], change, [n - 1]])
    max_run = int(np.diff(bounds).max())
    return [h, self_share, dominant, diversity, max_run / n]


def sequential_features(req: pd.DataFrame, min_actions: int) -> pd.DataFrame:
    acts = req[~req["is_beacon"]].sort_values(["session_id", "seq"])
    sessions = req["session_id"].unique()
    out = pd.DataFrame(np.nan, index=pd.Index(sorted(sessions), name="session_id"), columns=COLUMNS)

    sizes = acts.groupby("session_id").size()
    eligible = sizes[sizes >= min_actions].index
    rows = acts[acts["session_id"].isin(eligible)].groupby("session_id")["action"].apply(
        lambda s: sequence_features(s.to_numpy())
    )
    if len(rows):
        out.loc[rows.index, COLUMNS] = np.vstack(rows.to_numpy())
    out["seq_valid"] = out.index.isin(eligible).astype(int)
    return out
