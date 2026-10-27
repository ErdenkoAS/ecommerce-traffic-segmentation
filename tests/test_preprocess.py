import pandas as pd
import pytest

from trafseg import preprocess as pp

CONFIG = "config"


def _requests(rows):
    df = pd.DataFrame(rows, columns=["IpId", "UserAgent", "ts", "HttpMethod", "Uri", "Referrer"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["country"] = "PL"
    return df


def test_filter_drops_static_and_admin():
    df = _requests([
        ("1_PL", "ua", "2019-12-11 10:00:00", "GET", "/p-1234", "-"),
        ("1_PL", "ua", "2019-12-11 10:00:01", "GET", "/img/logo.png?v=2", "-"),
        ("1_PL", "ua", "2019-12-11 10:00:02", "GET", "/zarzadzanie/index.php", "-"),
    ])
    out = pp.filter_requests(df, pp.FilterRules.from_yaml(f"{CONFIG}/static_patterns.yaml"))
    assert out["Uri"].tolist() == ["/p-1234"]


def test_typing_first_match_and_beacon_method():
    df = _requests([
        ("1_PL", "ua", "2019-12-11 10:00:00", "GET", "/p-1234", "-"),
        ("1_PL", "ua", "2019-12-11 10:00:01", "GET", "/c-12_3", "-"),
        ("1_PL", "ua", "2019-12-11 10:00:02", "POST", "/inne/informacja_online.php", "-"),
        ("1_PL", "ua", "2019-12-11 10:00:03", "GET", "/inne/informacja_online.php", "-"),
        ("1_PL", "ua", "2019-12-11 10:00:04", "GET", "/unknown", "-"),
    ])
    out = pp.type_actions(df, pp.load_typing_rules(f"{CONFIG}/eclog_uri_types.yaml"))
    assert out["action"].tolist() == ["product", "category", "beacon", "info", "other"]
    assert out["is_beacon"].sum() == 1


def test_sessionize_splits_on_client_ua_and_gap():
    df = _requests([
        ("1_PL", "ua", "2019-12-11 10:00:00", "GET", "/", "-"),
        ("1_PL", "ua", "2019-12-11 10:29:00", "GET", "/p-1", "-"),
        ("1_PL", "ua", "2019-12-11 11:00:00", "GET", "/p-2", "-"),   # gap 31 min -> new session
        ("1_PL", "ub", "2019-12-11 10:00:30", "GET", "/p-3", "-"),   # other UA
        ("2_PL", "ua", "2019-12-11 10:00:30", "GET", "/p-4", "-"),   # other client
    ])
    out = pp.sessionize(df)
    assert out["session_id"].nunique() == 4
    s = out.set_index("Uri")["session_id"]
    assert s["/"] == s["/p-1"] != s["/p-2"]
    assert out.loc[out["Uri"] == "/p-1", "seq"].item() == 1


def test_session_table_counts():
    df = _requests([
        ("1_PL", "ua", "2019-12-11 10:00:00", "GET", "/p-1", "-"),
        ("1_PL", "ua", "2019-12-11 10:00:05", "POST", "/inne/informacja_online.php", "http://x"),
        ("1_PL", "ua", "2019-12-11 10:00:20", "GET", "/p-2", ""),
    ])
    df = pp.type_actions(df, pp.load_typing_rules(f"{CONFIG}/eclog_uri_types.yaml"))
    out = pp.session_table(pp.sessionize(df))
    row = out.iloc[0]
    assert (row["n_requests"], row["n_beacon"], row["n_actions"], row["n_no_referrer"]) == (3, 1, 2, 2)
    assert row["duration_s"] == 20


def test_bot_ua_flag():
    ua = pd.Series(["Mozilla/5.0 (Windows NT 10.0)", "Googlebot/2.1", "python-requests/2.22"])
    flags = pp.flag_bot_ua(ua, pp.load_bot_signatures(f"{CONFIG}/bot_ua_signatures.yaml"))
    assert flags.tolist() == [False, True, True]
