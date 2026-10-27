"""Synthetic benchmark generator (Section 2.6).

``generate`` returns a request log in the EClog column layout plus a client
table with the true class and account attributes. Because the log has the
same format as the real data, it goes through ``trafseg.preprocess`` and
``trafseg.features`` unchanged; the class is the ground truth per client.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .site import HOST, Site

CLASSES = ("human", "crawler", "scraper", "headless")
BOT_CLASSES = ("crawler", "scraper", "headless")
UNIX_EPOCH_TICKS = 621_355_968_000_000_000
COLUMNS = ["IpId", "UserId", "TimeStamp", "HttpMethod", "Uri", "HttpVersion",
           "ResponseCode", "Bytes", "Referrer", "UserAgent"]


def load_generator_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _weighted(rng, pairs):
    w = np.array([p[0] for p in pairs], dtype=float)
    return pairs[rng.choice(len(pairs), p=w / w.sum())][1]


def _choice(rng, dist: dict):
    keys = list(dist)
    p = np.array([dist[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


class Generator:
    def __init__(self, cfg: dict, seed: int, bot_share: float | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.site = Site(cfg["site"], cfg["beacon_uri"], self.rng)
        self.day_start = int(pd.Timestamp(cfg["day"], tz="UTC").timestamp())
        self.shares = dict(cfg["class_shares"])
        if bot_share is not None:
            base = sum(self.shares[c] for c in BOT_CLASSES)
            for c in BOT_CLASSES:
                self.shares[c] *= bot_share / base
            self.shares["human"] = 1 - bot_share
        self.diurnal = np.array(cfg["diurnal_weights"], dtype=float)
        self.diurnal /= self.diurnal.sum()

    # ----- client-level sampling -------------------------------------------------

    def _user_agent(self, cls: str) -> str:
        ua = self.cfg["user_agents"]
        if cls == "scraper":
            pool = _choice(self.rng, self.cfg["classes"]["scraper"]["ua_pool"])
            return _weighted(self.rng, ua["human"] if pool == "modern" else ua[f"scraper_{pool}"])
        return _weighted(self.rng, ua[cls])

    def _n_sessions(self, spec: dict) -> int:
        if "poisson_mean" in spec:
            return max(spec["min"], int(self.rng.poisson(spec["poisson_mean"])))
        return 1 if self.rng.random() < spec["one"] else int(self.rng.integers(2, spec["max"] + 1))

    def _account(self, spec: dict) -> dict:
        r = self.rng
        auth = r.random() < spec["authenticated"]
        return {
            "authenticated": int(auth),
            "account_age_days": int(r.lognormal(np.log(max(spec["age_days_median"], 1)), 1.0)) if auth else 0,
            "n_devices": int(1 + r.poisson(spec["devices_mean"])) if auth else 0,
            "complaints": int(r.poisson(spec["complaints_mean"])) if auth else 0,
            "suspicious_email": int(auth and r.random() < spec["suspicious_email"]),
        }

    def make_clients(self, n_per_class: dict) -> pd.DataFrame:
        """Client table with sequential ids; scraper campaigns occupy consecutive ids."""
        blocks = []  # each block: list of client dicts sharing consecutive ids
        for cls, n in n_per_class.items():
            spec = self.cfg["classes"][cls]
            i = 0
            while i < n:
                dist = spec.get("distributed")
                if dist and self.rng.random() < dist["share"]:
                    size = int(self.rng.integers(dist["ids_min"], dist["ids_max"] + 1))
                    ua, country = self._user_agent(cls), _choice(self.rng, spec["countries"])
                    blocks.append([dict(cls=cls, user_agent=ua, country=country, campaign=True)
                                   for _ in range(min(size, n - i))])
                    i += size
                else:
                    blocks.append([dict(cls=cls, user_agent=self._user_agent(cls),
                                        country=_choice(self.rng, spec["countries"]), campaign=False)])
                    i += 1
        self.rng.shuffle(blocks)
        rows, campaign_id, idx = [], 0, 1
        for block in blocks:
            if block[0]["campaign"]:
                campaign_id += 1
            for c in block:
                spec = self.cfg["classes"][c["cls"]]
                rows.append({"IpId": f"{idx}{c['country']}", "client_idx": idx, "class": c["cls"],
                             "country": c["country"], "user_agent": c["user_agent"],
                             "campaign_id": campaign_id if c["campaign"] else 0,
                             "n_sessions": self._n_sessions(spec["sessions_per_client"]),
                             **self._account(spec["account"])})
                idx += 1
        return pd.DataFrame(rows)

    # ----- session-level simulation ---------------------------------------------

    def _session_start(self, cls: str) -> float:
        if cls == "human":
            hour = self.rng.choice(24, p=self.diurnal)
            return self.day_start + hour * 3600 + self.rng.random() * 3600
        return self.day_start + self.rng.random() * 86400

    def _n_actions(self, spec: dict) -> int:
        if self.rng.random() < spec["single"]:
            return 1
        return 1 + int(self.rng.geometric(1 / spec["geom_mean"]))

    def _gap(self, spec: dict) -> float:
        return float(self.rng.lognormal(np.log(spec["median"]), spec["sigma"]))

    def _next_action(self, cls_spec: dict, state: str | None) -> str:
        if state is None:
            return _choice(self.rng, cls_spec["start"])
        rows = cls_spec["transitions"]
        return _choice(self.rng, rows.get(state) or rows["any"])

    def simulate_session(self, client: dict) -> list[dict]:
        cls = client["class"]
        spec = self.cfg["classes"][cls]
        static_mean = self.cfg["site"]["static_per_page"][cls]
        codes = spec["response_codes"]
        ref = spec["referrer"]
        beacon = spec["beacon"]
        t = self._session_start(cls)
        state, prev_url, out = None, None, []

        def emit(time, method, uri, code, action, referrer):
            out.append({"IpId": client["IpId"], "UserId": "-",
                        "TimeStamp": UNIX_EPOCH_TICKS + int(time) * 10_000_000,
                        "HttpMethod": method, "Uri": uri, "HttpVersion": "HTTP/1.1",
                        "ResponseCode": code, "Bytes": self.site.response_bytes(action, self.cfg["bytes_median"]),
                        "Referrer": referrer, "UserAgent": client["user_agent"]})

        for i in range(self._n_actions(spec["actions_per_session"])):
            state = self._next_action(spec, state)
            uri = self.site.uri(state, seo_search=(cls == "crawler"))
            if i == 0:
                referrer = _weighted(self.rng, self.cfg["external_referrers"]) if self.rng.random() < ref["external_first"] else "-"
            else:
                referrer = prev_url if self.rng.random() < ref["keep"] else "-"
            emit(t, "GET", uri, int(_choice(self.rng, {str(k): v for k, v in codes.items()})), state, referrer)
            page_url = HOST + uri
            page_ref = page_url if self.rng.random() < ref["keep"] else "-"
            for _ in range(self.rng.poisson(static_mean)):
                emit(t + self.rng.random(), "GET", self.site.uri("static"), 200, "static", page_ref)
            gap = self._gap(spec["gap_s"])
            if beacon:
                k, period = 1, beacon["period_s"]
                while k * period < gap:
                    jitter = 1 + beacon["jitter"] * self.rng.standard_normal() if beacon["jitter"] else 1
                    emit(t + k * period * jitter, "POST", self.site.beacon_uri, 200, "beacon", page_ref)
                    k += 1
            t += gap
            prev_url = page_url
        return out

    def generate(self, target_requests: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        # pilot run to learn requests per session per class, then size the client table
        per_session = {}
        for cls in CLASSES:
            probe = {"IpId": "0XX", "class": cls, "user_agent": self._user_agent(cls)}
            per_session[cls] = np.mean([len(self.simulate_session(probe)) for _ in range(300)])
        mean_sessions = {c: self._mean_sessions(c) for c in CLASSES}
        req_per_session = sum(self.shares[c] * per_session[c] for c in CLASSES)
        n_sessions = target_requests / req_per_session
        n_per_class = {c: max(1, int(round(n_sessions * self.shares[c] / mean_sessions[c]))) for c in CLASSES}

        clients = self.make_clients(n_per_class)
        rows = []
        for client in clients.to_dict("records"):
            for _ in range(client["n_sessions"]):
                rows.extend(self.simulate_session(client))
        req = pd.DataFrame(rows, columns=COLUMNS).sort_values("TimeStamp", kind="stable").reset_index(drop=True)
        return req, clients

    def _mean_sessions(self, cls: str) -> float:
        spec = self.cfg["classes"][cls]["sessions_per_client"]
        if "poisson_mean" in spec:
            return max(spec["min"], spec["poisson_mean"])
        return spec["one"] * 1 + (1 - spec["one"]) * (2 + spec["max"]) / 2


def generate(cfg: dict, target_requests: int, seed: int, bot_share: float | None = None):
    return Generator(cfg, seed, bot_share).generate(target_requests)
