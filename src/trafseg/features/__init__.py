"""Session feature families (Section 2.4).

Every function takes the request table produced by ``trafseg.preprocess``
(one row per action-level request with ``session_id``, ``action``,
``is_beacon``, ``ts`` ...) and returns one row per session. Column prefixes
name the family, so ablation is a column selection:

    tech_  technical          beh_  behavioural / temporal
    seq_   sequential         id_   identity (log part; account part needs a client table)
    rep_   reputation (account table only)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .account import account_features
from .behavioural import behavioural_features
from .identity import identity_features
from .sequential import sequential_features
from .technical import technical_features

FAMILIES = ("tech", "beh", "seq", "id", "rep")


def load_feature_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def build_features(req: pd.DataFrame, cfg: dict, clients: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full feature matrix indexed by ``session_id``; ``clients`` adds the account families."""
    parts = [
        technical_features(req, cfg["response_codes"]),
        behavioural_features(req),
        sequential_features(req, cfg["min_actions_for_sequence"]),
        identity_features(req, cfg["platform_country"]),
    ]
    if clients is not None:
        parts.append(account_features(req, clients))
    return pd.concat(parts, axis=1).sort_index()


def select_families(X: pd.DataFrame, families: list[str]) -> pd.DataFrame:
    """Columns of the given families, e.g. ['tech', 'beh'] for an ablation row."""
    keep = [c for c in X.columns if c.split("_", 1)[0] in families]
    return X[keep]
