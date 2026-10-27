"""Technical family: request-level and user-agent string properties."""

from __future__ import annotations

import numpy as np
import pandas as pd

_UA_TOKEN = r"[A-Za-z][A-Za-z0-9._-]*/[0-9][0-9A-Za-z.]*"  # e.g. Chrome/78.0.3904


def technical_features(req: pd.DataFrame, response_codes: list[int]) -> pd.DataFrame:
    g = req.groupby("session_id", sort=True)
    n = g.size()

    out = pd.DataFrame(index=n.index)
    out["tech_no_referrer_share"] = g["Referrer"].agg(lambda s: s.isin(["", "-"]).mean())
    method = req["HttpMethod"].str.upper()
    out["tech_post_share"] = (method == "POST").groupby(req["session_id"]).mean()
    out["tech_head_share"] = (method == "HEAD").groupby(req["session_id"]).mean()
    for code in response_codes:
        out[f"tech_share_{code}"] = (req["ResponseCode"] == code).groupby(req["session_id"]).mean()
    out["tech_bytes_median"] = g["Bytes"].median().astype(float)

    ua = g["UserAgent"].first()
    # share of sessions in the window that carry the same user-agent string
    out["tech_ua_popularity"] = ua.map(ua.value_counts(normalize=True))
    out["tech_ua_tokens"] = ua.str.count(_UA_TOKEN)
    out["tech_ua_compatible"] = ua.str.contains("compatible;", case=False, regex=False).astype(int)
    out["tech_ua_length"] = ua.str.len()
    return out.fillna(0.0)
