import numpy as np
import pandas as pd
import pytest

from trafseg.features import build_features, select_families
from trafseg.features.sequential import sequence_features

CFG = {"platform_country": "PL", "min_actions_for_sequence": 5, "response_codes": [404, 499]}


def _req(rows):
    cols = ["session_id", "seq", "IpId", "country", "UserAgent", "ts", "HttpMethod", "Uri",
            "Referrer", "ResponseCode", "Bytes", "action"]
    df = pd.DataFrame(rows, columns=cols)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["is_beacon"] = df["action"] == "beacon"
    return df


UA = "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0 Safari/537.36"
BOT = "Mozilla/4.0 (compatible; MSIE 5.01; Windows NT 5.0)"


def test_sequence_features_degenerate_and_diverse():
    h, self_share, dominant, diversity, max_run = sequence_features(np.array(["product"] * 6))
    assert (h, self_share, dominant, diversity, max_run) == (0.0, 1.0, 1.0, 0.2, 1.0)

    h, self_share, dominant, diversity, max_run = sequence_features(
        np.array(["home", "category", "product", "cart", "checkout"]))
    assert h == 0.0 and self_share == 0.0            # every state has one deterministic successor
    assert dominant == 0.25 and diversity == 1.0 and max_run == 0.2


def test_sequence_entropy_positive_for_mixed_transitions():
    h, *_ = sequence_features(np.array(["product", "category", "product", "product", "category", "search"]))
    assert h > 0


def test_build_features_two_sessions():
    t = "2019-12-11 10:00:"
    rows = [
        # human-like session: 3 actions + 2 beacons, referrer present, commerce action
        (1, 0, "1PL", "PL", UA, t + "00", "GET", "/", "-", 200, 5000, "home"),
        (1, 1, "1PL", "PL", UA, t + "10", "POST", "/inne/informacja_online.php", "http://x", 200, 0, "beacon"),
        (1, 2, "1PL", "PL", UA, t + "20", "GET", "/p-1.html", "http://x", 200, 6000, "product"),
        (1, 3, "1PL", "PL", UA, t + "30", "POST", "/inne/informacja_online.php", "http://x", 200, 0, "beacon"),
        (1, 4, "1PL", "PL", UA, t + "40", "GET", "/koszyk.html", "http://x", 200, 4000, "cart"),
        # scraper-like session: 5 product pages, no referrer, foreign, old UA
        (2, 0, "2DE", "DE", BOT, t + "00", "GET", "/p-1.html", "-", 200, 9000, "product"),
        (2, 1, "2DE", "DE", BOT, t + "01", "GET", "/p-2.html", "-", 200, 9000, "product"),
        (2, 2, "2DE", "DE", BOT, t + "02", "GET", "/p-3.html", "-", 404, 9000, "product"),
        (2, 3, "2DE", "DE", BOT, t + "03", "GET", "/p-4.html", "-", 200, 9000, "product"),
        (2, 4, "2DE", "DE", BOT, t + "04", "GET", "/p-5.html", "-", 200, 9000, "product"),
    ]
    X = build_features(_req(rows), CFG)
    assert list(X.index) == [1, 2]
    h, b = X.loc[1], X.loc[2]

    assert (h["beh_n_requests"], h["beh_n_actions"], b["beh_n_actions"]) == (5, 3, 5)
    assert h["beh_beacon_share"] == pytest.approx(0.4) and b["beh_beacon_share"] == 0
    assert h["beh_commerce_share"] == pytest.approx(1 / 3) and b["beh_commerce_share"] == 0
    assert h["beh_iat_median_s"] == 20 and b["beh_iat_median_s"] == 1
    assert h["tech_no_referrer_share"] == pytest.approx(0.2) and b["tech_no_referrer_share"] == 1
    assert b["tech_share_404"] == pytest.approx(0.2)
    assert (h["tech_ua_compatible"], b["tech_ua_compatible"]) == (0, 1)
    assert (h["id_foreign"], b["id_foreign"]) == (0, 1)

    assert h["seq_valid"] == 0 and np.isnan(h["seq_markov_entropy"])
    assert b["seq_valid"] == 1 and b["seq_self_transition_share"] == 1.0


def test_select_families_by_prefix():
    X = pd.DataFrame(columns=["tech_a", "beh_b", "seq_c", "seq_valid", "id_d", "label_bot_ua"])
    assert list(select_families(X, ["tech", "id"]).columns) == ["tech_a", "id_d"]
    assert list(select_families(X, ["seq"]).columns) == ["seq_c", "seq_valid"]
