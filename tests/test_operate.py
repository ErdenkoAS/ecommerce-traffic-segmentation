import pandas as pd

from trafseg import operate as op

OCFG = {"automated_rule_share": 0.5, "automated_beacon_share_max": 0.0, "automated_min_actions": 5}


def _prof(rows):
    return pd.DataFrame(rows, columns=["share", "beh_n_actions", "beh_beacon_share"])


def test_match_segments_recovers_permutation_and_adds_new_ids():
    prev = _prof([[0.6, 1, 0.5], [0.3, 12, 0.0], [0.1, 20, 0.0]])
    cur = _prof([[0.1, 19, 0.0], [0.6, 1, 0.5], [0.3, 12, 0.0], [0.0, 3, 0.2]])  # permuted + one new
    m = op.match_segments(prev, cur)
    assert m[0] == 2 and m[1] == 0 and m[2] == 1 and m[3] == 3


def test_segment_status_thresholds():
    s = op.segment_status(pd.Series([0.95, 0.5, 0.05]))
    assert list(s) == ["covered", "partial", "outside"]


def test_interpret_segments_by_rule_or_profile():
    prof = _prof([[0.6, 1, 0.5], [0.3, 12, 0.0], [0.1, 20, 0.0], [0.0, 3, 0.0]])
    rule = pd.Series([0.05, 0.95, 0.3, 0.1])
    auto = op.interpret_segments(prof, rule, OCFG)
    assert list(auto) == [False, True, True, False]   # 2: beacon-free multi-action, 3: too short


def test_aggregate_clients_mode_and_mixed():
    s = pd.DataFrame({"client": ["a", "a", "a", "b", "b", "b", "c"],
                      "window": ["d2", "d2", "d3", "d2", "d3", "d3", "d1"], "segment": [1, 1, 0, 0, 1, 2, 1]})
    agg = op.aggregate_clients(s, n_windows=2)
    assert agg.loc["a", "aggregated"] == "1" and agg.loc["a", "n_sessions"] == 3
    assert agg.loc["b", "aggregated"] == "mixed"
    assert "c" not in agg.index                       # outside the last two windows
