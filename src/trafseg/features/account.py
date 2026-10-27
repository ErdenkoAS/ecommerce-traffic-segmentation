"""Account-based identity and reputation features.

Available only where the platform's account system is: on the synthetic
benchmark these come from the generator's client table. The columns join on
the client id, so every session of a client shares them.
"""

from __future__ import annotations

import pandas as pd

ACCOUNT_COLUMNS = {
    "authenticated": "id_authenticated",
    "account_age_days": "id_account_age_days",
    "n_devices": "id_n_devices",
    "complaints": "rep_complaints",
    "suspicious_email": "rep_suspicious_email",
}


def account_features(req: pd.DataFrame, clients: pd.DataFrame) -> pd.DataFrame:
    ip = req.groupby("session_id", sort=True)["IpId"].first()
    table = clients.set_index("IpId")[list(ACCOUNT_COLUMNS)].rename(columns=ACCOUNT_COLUMNS)
    return table.reindex(ip.to_numpy()).set_index(ip.index)
