import pandas as pd

from trafseg.generator import BOT_CLASSES, COLUMNS, generate, load_generator_config
from trafseg.io_eclog import ticks_to_datetime

CFG = load_generator_config("config/generator.yaml")


def test_output_format_matches_eclog():
    req, clients = generate(CFG, target_requests=3000, seed=1)
    assert list(req.columns) == COLUMNS
    assert req["IpId"].str.match(r"^\d+[A-Z]{2}$").all()
    assert req["TimeStamp"].is_monotonic_increasing
    ts = ticks_to_datetime(req["TimeStamp"])
    assert ts.min().strftime("%Y-%m-%d") == CFG["day"]
    assert set(clients["class"]) == {"human", "crawler", "scraper", "headless"}
    assert set(clients["IpId"]) >= set(req["IpId"])


def test_seed_is_reproducible():
    a, _ = generate(CFG, target_requests=3000, seed=7)
    b, _ = generate(CFG, target_requests=3000, seed=7)
    c, _ = generate(CFG, target_requests=3000, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_bot_share_override_rescales_bot_classes():
    _, clients = generate(CFG, target_requests=20000, seed=2, bot_share=0.4)
    sessions = clients.groupby("class")["n_sessions"].sum()
    bot = sessions[list(BOT_CLASSES)].sum() / sessions.sum()
    assert 0.3 < bot < 0.5


def test_class_signatures():
    req, clients = generate(CFG, target_requests=20000, seed=3)
    cls = req["IpId"].map(clients.set_index("IpId")["class"])
    beacon = req["Uri"] == CFG["beacon_uri"]
    assert beacon[cls == "crawler"].sum() == 0 and beacon[cls == "scraper"].sum() == 0
    assert beacon[cls == "human"].mean() > 0.2 and beacon[cls == "headless"].mean() > 0.2
    assert (req.loc[cls == "crawler", "Referrer"] == "-").all()
    campaigns = clients[clients["campaign_id"] > 0]
    assert (campaigns["class"] == "scraper").all()
    for _, g in campaigns.groupby("campaign_id"):
        assert g["client_idx"].diff().dropna().eq(1).all()   # consecutive ids
        assert g["user_agent"].nunique() == 1
