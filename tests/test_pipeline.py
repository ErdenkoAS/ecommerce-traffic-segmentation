import numpy as np
import pandas as pd
import pytest

from trafseg import pipeline as pl
from trafseg.evaluation import external_metrics, internal_metrics, rule_metrics

CFG = pl.load_pipeline_config("config/pipeline.yaml")
BOT = ("crawler", "scraper", "headless")


def _toy(n=300, seed=0):
    """Three well-separated blobs: humans, crawlers, scrapers."""
    rng = np.random.default_rng(seed)
    cls = np.array(["human"] * n + ["crawler"] * (n // 3) + ["scraper"] * (n // 3))
    centre = {"human": (0, 0, 0), "crawler": (8, 0, 0), "scraper": (0, 8, 0)}
    pts = np.vstack([rng.normal(centre[c], 0.5, 3) for c in cls])
    X = pd.DataFrame(pts, columns=["tech_a", "beh_b", "seq_c"])
    X["seq_valid"] = 1
    X["beh_n_actions"] = np.where(cls == "human", 2, 20)
    X.loc[::7, "seq_c"] = np.nan                       # some missing values
    X["tech_no_referrer_share"] = np.where(cls == "crawler", 1.0, 0.0)
    X["label_bot_ua"] = (cls == "crawler").astype(int)
    X["label_class"] = cls
    return X


def test_prepare_selects_families_logs_and_imputes():
    X = _toy()
    Z, cols = pl.prepare(X, ["tech", "beh"], CFG)
    assert cols == ["tech_a", "beh_b", "beh_n_actions", "tech_no_referrer_share"]
    assert Z.shape == (len(X), 4) and not np.isnan(Z).any()
    Z2, cols2 = pl.prepare(X, ["seq"], CFG)
    assert cols2 == ["seq_c", "seq_valid"] and not np.isnan(Z2).any()


def test_family_columns_log_identity_subset():
    X = pd.DataFrame(columns=["id_foreign", "id_authenticated", "rep_complaints", "label_class"])
    assert pl.family_columns(X, ["id_log"]) == ["id_foreign"]
    assert pl.family_columns(X, ["id", "rep"]) == ["id_foreign", "id_authenticated", "rep_complaints"]


def test_rule_anchor_signature_only():
    X = _toy()
    X.loc[0, "tech_no_referrer_share"] = 1.0          # a human session without referrer
    assert pl.rule_anchor(X, CFG)[0] == 1 and pl.rule_anchor(X, CFG, signature_only=True)[0] == 0


@pytest.mark.parametrize("method", ["kmeans", "gmm", "agglo", "hdbscan"])
def test_segment_recovers_blobs(method):
    X = _toy()
    Z, _ = pl.prepare(X, ["tech", "beh", "seq"], CFG)
    Zm = pl.reduce(Z, method, CFG, 0)
    seg = pl.segment(Zm, method, CFG, seed=0, rule=pl.rule_anchor(X, CFG))
    ext = external_metrics(seg.labels, X["label_class"], BOT)
    assert ext["ari"] > 0.9 and ext["fpr"] == 0 and ext["bot_recall"] > 0.95
    assert seg.stable
    if method != "hdbscan":
        assert seg.k == 3


def test_rule_anchor_and_rule_metrics():
    X = _toy()
    rule = pl.rule_anchor(X, CFG)
    m = rule_metrics(rule, X["label_class"], BOT)
    assert m["recall_crawler"] == 1.0 and m["recall_scraper"] == 0.0 and m["fpr"] == 0.0


def test_internal_metrics_ignore_noise():
    Z = np.vstack([np.zeros((10, 2)), np.ones((10, 2)) * 5])
    labels = np.array([0] * 10 + [1] * 9 + [-1])
    m = internal_metrics(Z, labels)
    assert m["noise_share"] == pytest.approx(0.05) and m["silhouette"] > 0.9


def test_profiles_have_share_and_medians():
    X = _toy()
    labels = (X["label_class"] != "human").astype(int).to_numpy()
    p = pl.profiles(X, labels, ["beh_n_actions"])
    assert list(p.columns) == ["share", "beh_n_actions"]
    assert p.loc[1, "beh_n_actions"] == 20 and p["share"].sum() == pytest.approx(1.0)
