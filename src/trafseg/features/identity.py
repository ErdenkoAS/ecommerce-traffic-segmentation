"""Identity family, the part computable from a server log: client-level
diversity within the window and geography. Account-based features
(authentication, account age, device history) exist only on the synthetic
benchmark and are added there by the generator."""

from __future__ import annotations

import pandas as pd


def identity_features(req: pd.DataFrame, platform_country: str) -> pd.DataFrame:
    s = req.groupby("session_id", sort=True).agg(ip=("IpId", "first"),
                                                 ua=("UserAgent", "first"),
                                                 country=("country", "first"))
    per_ip = s.groupby("ip").agg(n_sessions=("ua", "size"), n_ua=("ua", "nunique"))
    out = pd.DataFrame(index=s.index)
    out["id_sessions_per_client"] = s["ip"].map(per_ip["n_sessions"])
    out["id_ua_per_client"] = s["ip"].map(per_ip["n_ua"])
    out["id_foreign"] = (s["country"] != platform_country).astype(int)
    return out
