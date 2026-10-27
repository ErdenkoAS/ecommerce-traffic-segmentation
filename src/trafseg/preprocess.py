"""Filtering, action typing and sessionization of request logs (Section 2.3).

The output is a request table restricted to action-level requests, with one
session id per (client id, user-agent) pair separated by inactivity gaps.
The same functions are applied to both data circuits, so the session tables
are comparable by construction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SESSION_TIMEOUT = pd.Timedelta(minutes=30)


@dataclass
class FilterRules:
    static: list[re.Pattern] = field(default_factory=list)
    service: list[re.Pattern] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FilterRules":
        cfg = yaml.safe_load(Path(path).read_text())
        ext = "|".join(cfg.get("static_extensions", []))
        static = [re.compile(rf"\.(?:{ext})$", re.I)] if ext else []
        static += [re.compile(p, re.I) for p in cfg.get("static_paths", [])]
        service = [re.compile(p, re.I) for p in cfg.get("service_paths", [])]
        return cls(static=static, service=service)


@dataclass
class TypingRule:
    type: str
    pattern: re.Pattern
    method: str | None = None


def load_typing_rules(path: str | Path) -> list[TypingRule]:
    cfg = yaml.safe_load(Path(path).read_text())
    return [
        TypingRule(r["type"], re.compile(r["pattern"], re.I), r.get("method"))
        for r in cfg["rules"]
    ]


def load_bot_signatures(path: str | Path) -> list[str]:
    return [s.lower() for s in yaml.safe_load(Path(path).read_text())["signatures"]]


def uri_path(uri: pd.Series) -> pd.Series:
    return uri.str.split("?", n=1).str[0]


def _matches_any(path: pd.Series, patterns: list[re.Pattern]) -> pd.Series:
    mask = pd.Series(False, index=path.index)
    for p in patterns:
        mask |= path.str.contains(p, na=False)
    return mask


def filter_requests(df: pd.DataFrame, rules: FilterRules) -> pd.DataFrame:
    """Drop static resources and service traffic; keep action-level requests."""
    path = uri_path(df["Uri"])
    drop = _matches_any(path, rules.static) | _matches_any(path, rules.service)
    return df.loc[~drop].copy()


def type_actions(df: pd.DataFrame, rules: list[TypingRule]) -> pd.DataFrame:
    """Assign ``action`` to each request by the first matching rule, else 'other'."""
    path = uri_path(df["Uri"])
    action = pd.Series("other", index=df.index, dtype="object")
    unassigned = pd.Series(True, index=df.index)
    for r in rules:
        hit = unassigned & path.str.contains(r.pattern, na=False, regex=True)
        if r.method:
            hit &= df["HttpMethod"].str.upper() == r.method.upper()
        action[hit] = r.type
        unassigned &= ~hit
    out = df.copy()
    out["action"] = action
    out["is_beacon"] = action == "beacon"
    return out


def sessionize(df: pd.DataFrame, timeout: pd.Timedelta = SESSION_TIMEOUT) -> pd.DataFrame:
    """Add ``session_id``: same IpId and UserAgent, gaps between requests <= timeout."""
    df = df.sort_values(["IpId", "UserAgent", "ts"], kind="stable").copy()
    same_client = (df["IpId"].eq(df["IpId"].shift())
                   & df["UserAgent"].eq(df["UserAgent"].shift()))
    gap = df["ts"].diff()
    new_session = ~same_client | (gap > timeout)
    df["session_id"] = np.cumsum(new_session.to_numpy())
    df["seq"] = df.groupby("session_id").cumcount()
    return df.reset_index(drop=True)


def flag_bot_ua(ua: pd.Series, signatures: list[str]) -> pd.Series:
    """True for a known-bot signature or an empty user-agent."""
    pattern = "|".join(re.escape(s) for s in signatures)
    return ua.isin(["", "-"]) | ua.str.lower().str.contains(pattern, regex=True, na=False)


def session_table(req: pd.DataFrame) -> pd.DataFrame:
    """One row per session with basic counts; feature families are built elsewhere."""
    g = req.groupby("session_id", sort=True)
    out = g.agg(
        ip_id=("IpId", "first"),
        country=("country", "first"),
        user_agent=("UserAgent", "first"),
        start=("ts", "min"),
        end=("ts", "max"),
        n_requests=("Uri", "size"),
        n_beacon=("is_beacon", "sum"),
        n_no_referrer=("Referrer", lambda s: (s.isin(["", "-"])).sum()),
    )
    out["n_actions"] = out["n_requests"] - out["n_beacon"]
    out["duration_s"] = (out["end"] - out["start"]).dt.total_seconds()
    return out.reset_index()
