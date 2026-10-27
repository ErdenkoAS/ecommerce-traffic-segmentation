"""Algorithm 2 on a real EClog window (Section 3.4).

Usage:
    python scripts/run_eclog_week.py data/raw/eclog_week_2020-01-20.csv [--window day|week]
        [--method agglo] [--audit-n 60] [--seed 0]

The raw slice goes through the same preprocessing and feature extraction as
the benchmark. Each local calendar day is one window: Algorithm 1 segments
it, segments are matched to the previous day's by profile, every segment gets
its status relative to the rules, and clients are aggregated over the week.
There is no ground truth, so the script reports signature coverage (share of
bot-user-agent sessions that land in segments interpreted as automated) and
the share of browser-user-agent sessions in those segments, and writes a
blind stratified sample of the latter for manual audit.

Outputs (results/eclog_<stem>_*.csv): windows, segments, clients,
audit_sample (blind), audit_key.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from trafseg import operate as op
from trafseg import pipeline as pl
from trafseg import preprocess as pp
from trafseg.features import build_features, load_feature_config
from trafseg.io_eclog import read_eclog

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
TZ = "Europe/Warsaw"


def preprocess(raw_path: Path) -> pd.DataFrame:
    raw = read_eclog(raw_path)
    req = pp.filter_requests(raw, pp.FilterRules.from_yaml(CONFIG / "static_patterns.yaml"))
    req = pp.type_actions(req, pp.load_typing_rules(CONFIG / "eclog_uri_types.yaml"))
    req = pp.sessionize(req)
    print(f"requests {len(raw):,} -> actions {len(req):,}, sessions {req['session_id'].nunique():,}")
    return req


def window_of(req: pd.DataFrame, window: str) -> pd.Series:
    """Window label per session: local date of its first request, or one week."""
    start = req.groupby("session_id")["ts"].min().dt.tz_convert(TZ)
    if window == "week":
        return pd.Series("week", index=start.index)
    return start.dt.strftime("%Y-%m-%d")


def session_features(req: pd.DataFrame, fcfg: dict, signatures: list[str]) -> pd.DataFrame:
    X = build_features(req, fcfg)
    ua = req.groupby("session_id")["UserAgent"].first()
    X["label_bot_ua"] = pp.flag_bot_ua(ua, signatures).astype(int)
    return X


_BROWSERS = ["OPR", "Edge", "Edg", "SamsungBrowser", "YaBrowser", "CriOS", "FxiOS", "Chrome", "Firefox", "Safari",
             "MSIE", "Trident", "Opera"]


def ua_family_of(ua: str, signatures: list[str]) -> str:
    """Matched bot signature, or the browser product with its major version."""
    low = ua.lower()
    if ua in ("", "-"):
        return "(empty)"
    for sig in signatures:
        if sig.lower() in low:
            return sig
    for b in _BROWSERS:
        m = re.search(rf"{b}[/ ]([0-9]+)", ua)
        if m:
            return f"{b}/{m.group(1)}"
    return ua[:40]


def audit_record(req: pd.DataFrame) -> pd.DataFrame:
    """Per-session facts outside the feature matrix, for the manual audit."""
    g = req.sort_values("ts").groupby("session_id")
    return pd.DataFrame({
        "ip_id": g["IpId"].first(), "country": g["country"].first(), "user_agent": g["UserAgent"].first(),
        "start": g["ts"].min().dt.tz_convert(TZ).dt.strftime("%Y-%m-%d %H:%M:%S"),
        "n_requests": g.size(), "n_actions": g["is_beacon"].apply(lambda s: int((~s).sum())),
        "duration_s": (g["ts"].max() - g["ts"].min()).dt.total_seconds(),
        "response_codes": g["ResponseCode"].apply(lambda s: " ".join(map(str, s.tolist()[:40]))),
        "uri_sequence": g["Uri"].apply(lambda s: " | ".join(s[~req.loc[s.index, "is_beacon"]].tolist()[:25])),
        "unique_uri_share": g["Uri"].apply(lambda s: s.nunique() / len(s)),
        "referrer_present_share": g["Referrer"].apply(lambda s: float((~s.isin(["", "-"])).mean())),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("--window", choices=["day", "week"], default="day")
    ap.add_argument("--method", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit-n", type=int, default=60, help="browser-UA sessions from automated segments to audit")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    cfg = pl.load_pipeline_config(CONFIG / "pipeline.yaml")
    ocfg = cfg["operate"]
    method = args.method or ocfg["method"]
    fcfg = load_feature_config(CONFIG / "features.yaml")
    signatures = pp.load_bot_signatures(CONFIG / "bot_ua_signatures.yaml")
    stem = args.raw.stem.replace("eclog_", "")
    tag = f"{stem}_{args.window}_{method}"

    req = preprocess(args.raw)
    win = window_of(req, args.window)
    req["window"] = req["session_id"].map(win)

    windows, segments, sessions = [], [], []
    prev_prof = None
    for w, req_w in req.groupby("window", sort=True):
        X = session_features(req_w, fcfg, signatures)
        Z, cols = pl.prepare(X, cfg["ablation"][ocfg["families"]], cfg)
        rule = pl.rule_anchor(X, cfg)
        seg = pl.segment(pl.reduce(Z, method, cfg, args.seed), method, cfg, args.seed, rule)
        prof = pl.profiles(X, seg.labels, cols)
        mapping = op.match_segments(prev_prof, prof) if prev_prof is not None else {int(s): int(s) for s in prof.index}
        labels = pd.Series(seg.labels, index=X.index).map(mapping)
        prof.index = [mapping[int(s)] for s in prof.index]
        prev_prof = prof.sort_index()

        rule_share = pd.Series(rule, index=X.index).groupby(labels).mean()
        sig_share = X["label_bot_ua"].groupby(labels).mean()
        status = op.segment_status(rule_share, ocfg["covered_share"], ocfg["outside_share"])
        automated = op.interpret_segments(prev_prof, rule_share, ocfg)
        auto_flag = labels.map(automated).astype(bool)
        bot_ua = X["label_bot_ua"] == 1

        windows.append({"window": w, "n_sessions": len(X), "n_bot_ua": int(bot_ua.sum()), "k": seg.k,
                        "stable": seg.stable, "silhouette": round(seg.silhouette, 3),
                        "n_automated_segments": int(automated.sum()),
                        "signature_coverage": float(auto_flag[bot_ua].mean()),
                        "rule_coverage": float(auto_flag[rule.astype(bool)].mean()),
                        "automated_share": float(auto_flag.mean()),
                        "browser_in_automated": int((auto_flag & ~bot_ua).sum()),
                        "browser_in_automated_share": float(auto_flag[~bot_ua].mean())})
        for s in prev_prof.index:
            segments.append({"window": w, "segment": s, "share": prev_prof.loc[s, "share"],
                             "rule_share": rule_share[s], "signature_share": sig_share[s], "status": status[s],
                             "automated": bool(automated[s]),
                             **{c: prev_prof.loc[s, c] for c in ["beh_n_actions", "beh_beacon_share", "beh_iat_median_s",
                                                                 "tech_no_referrer_share", "seq_self_transition_share",
                                                                 "beh_product_share", "id_foreign"] if c in prev_prof}})
        first = req_w.groupby("session_id")[["IpId", "UserAgent"]].first().reindex(X.index)
        sessions.append(pd.DataFrame({"session_id": X.index, "window": w, "segment": labels.to_numpy(),
                                      "automated": auto_flag.to_numpy(), "bot_ua": bot_ua.to_numpy(),
                                      "rule": rule.astype(bool), "n_actions": X["beh_n_actions"].to_numpy(),
                                      "no_referrer": X["tech_no_referrer_share"].to_numpy(),
                                      "beacon_share": X["beh_beacon_share"].to_numpy(),
                                      "client": first["IpId"].to_numpy(), "user_agent": first["UserAgent"].to_numpy()}))
        r = windows[-1]
        print(f"{w}: sessions={r['n_sessions']:,} bot_ua={r['n_bot_ua']:,} k={seg.k} stable={seg.stable} "
              f"automated_segments={r['n_automated_segments']} signature_coverage={r['signature_coverage']:.3f} "
              f"browser_in_automated={r['browser_in_automated']} ({r['browser_in_automated_share']:.3%})", flush=True)

    W, S, SS = pd.DataFrame(windows), pd.DataFrame(segments), pd.concat(sessions, ignore_index=True)
    clients = op.aggregate_clients(SS, ocfg["aggregation_windows"], ocfg["client_mode_share"])
    seg_auto = SS.groupby("segment")["automated"].mean()  # a matched segment id may flip between windows
    clients["automated"] = clients["segment"].map(seg_auto).ge(0.5) & (clients["aggregated"] != "mixed")

    # blind audit sample: browser-UA sessions in automated segments (stratified by window x segment)
    # plus the same number of browser-UA multi-action sessions from the other segments
    rng = np.random.default_rng(args.seed)
    cand = SS[SS["automated"] & ~SS["bot_ua"]]
    per = max(1, args.audit_n // max(1, cand.groupby(["window", "segment"]).ngroups))
    pick = cand.groupby(["window", "segment"], group_keys=False).apply(
        lambda g: g.sample(min(len(g), per), random_state=int(rng.integers(1 << 31))))
    ctrl_pool = SS[~SS["automated"] & ~SS["bot_ua"] & (SS["n_actions"] >= fcfg["min_actions_for_sequence"])]
    ctrl = ctrl_pool.sample(min(len(ctrl_pool), len(pick)), random_state=args.seed)
    audit = pd.concat([pick.assign(group="automated"), ctrl.assign(group="control")])
    audit = audit.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    audit.insert(0, "audit_id", [f"A{i:03d}" for i in range(len(audit))])
    facts = audit_record(req[req["session_id"].isin(audit["session_id"])])
    sample = audit[["audit_id", "session_id"]].join(facts, on="session_id").drop(columns="session_id")
    sample["verdict"] = ""  # to be filled by the auditor: automated / human / unclear
    key = audit[["audit_id", "session_id", "window", "segment", "group"]]

    # what escapes and what is found beyond the signatures: user-agent families and session shape
    ua_family = SS["user_agent"].map(lambda u: ua_family_of(u, signatures))
    escaped = SS[SS["bot_ua"] & ~SS["automated"]]
    beyond = SS[~SS["bot_ua"] & SS["automated"]]
    diag = pd.concat([
        ua_family[escaped.index].value_counts().head(15).rename("n").reset_index().assign(group="bot_ua_not_automated"),
        ua_family[beyond.index].value_counts().head(15).rename("n").reset_index().assign(group="browser_ua_automated"),
    ]).rename(columns={"index": "user_agent_family", "user_agent": "user_agent_family"})
    shape = pd.DataFrame({
        "group": ["bot_ua_not_automated", "browser_ua_automated"],
        "n": [len(escaped), len(beyond)],
        "single_action_share": [(escaped["n_actions"] == 1).mean(), (beyond["n_actions"] == 1).mean()],
        "ge5_actions_share": [(escaped["n_actions"] >= 5).mean(), (beyond["n_actions"] >= 5).mean()],
        "no_referrer_share": [escaped["no_referrer"].mean(), beyond["no_referrer"].mean()],
        "zero_beacon_share": [(escaped["beacon_share"] == 0).mean(), (beyond["beacon_share"] == 0).mean()],
    })

    args.out.mkdir(parents=True, exist_ok=True)
    for name, df in (("windows", W), ("segments", S), ("clients", clients.reset_index()),
                     ("audit_sample", sample), ("audit_key", key), ("diagnostics", diag)):
        df.to_csv(args.out / f"eclog_{tag}_{name}.csv", index=False)

    pd.set_option("display.width", 250)
    print("\nwindows:")
    print(W.set_index("window").round(3).to_string())
    print("\nsegments:")
    print(S.round(3).to_string(index=False))
    print(f"\nover all windows: sessions={len(SS):,} bot_ua={int(SS['bot_ua'].sum()):,} "
          f"signature_coverage={SS.loc[SS['bot_ua'], 'automated'].mean():.3f} "
          f"rule_coverage={SS.loc[SS['rule'], 'automated'].mean():.3f} "
          f"browser_in_automated={int((SS['automated'] & ~SS['bot_ua']).sum()):,} "
          f"({SS.loc[~SS['bot_ua'], 'automated'].mean():.3%} of browser-UA sessions)")
    print("\nsession shape of the two off-diagonal groups:")
    print(shape.round(3).to_string(index=False))
    print("\nuser-agent families:")
    print(diag.to_string(index=False))
    print(f"clients: {len(clients):,}, aggregated automated {int(clients['automated'].sum()):,}, "
          f"mixed {int((clients['aggregated'] == 'mixed').sum()):,}")
    print(f"audit sample: {len(sample)} sessions ({(audit['group'] == 'automated').sum()} from automated segments)")
    print(f"written to {args.out}/eclog_{tag}_*.csv")


if __name__ == "__main__":
    main()
