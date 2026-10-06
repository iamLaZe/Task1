"""Run with `pytest` or `python tests/test_metrics.py` (no pytest needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import metrics as M, scoring as S, text_analysis as T  # noqa: E402


def test_csat_excludes_blanks():
    s = M.clean_csat(pd.Series(["5", None, "", "3", "9", "0"]))
    st = M.csat_stats(s)
    assert st["rated"] == 2 and st["avg_rating"] == 4.0   # blanks and invalid not counted as 0


def test_scores_4_and_5_are_positive():
    st = M.csat_stats(pd.Series([1, 2, 3, 4, 5, np.nan]))
    assert st["rated"] == 5 and st["csat_pct"] == 40.0


def test_legacy_timestamp_normalization():
    r = pd.Series(["2025-01-01 03:05", "2025-01-01 03:05"])
    out = M.normalize_resolved_at(r, pd.Series(["legacy_fd", "helpdesk"]))
    assert out[0] == pd.Timestamp("2025-01-01 08:35") and out[1] == pd.Timestamp("2025-01-01 03:05")


def test_aht_calculation_and_negative_fix():
    fr = pd.Series(pd.to_datetime(["2025-01-01 08:22"]))
    raw = M.aht_minutes(fr, pd.Series(pd.to_datetime(["2025-01-01 03:05"])))
    fixed = M.aht_minutes(fr, M.normalize_resolved_at(pd.Series(["2025-01-01 03:05"]), pd.Series(["legacy_fd"])))
    assert raw[0] < 0 and fixed[0] == 13.0


def test_missing_resolution_is_nan_not_zero():
    fr = pd.Series(pd.to_datetime(["2025-01-01 08:22"]))
    aht = M.aht_minutes(fr, M.normalize_resolved_at(pd.Series([pd.NaT]), pd.Series(["helpdesk"])))
    assert np.isnan(aht[0])
    g = pd.DataFrame({"csat": [np.nan], "aht_min": aht, "sla_breach": [False], "replacement_flag": [False],
                      "replacement_cost": [np.nan], "transfers": [0]})
    assert np.isnan(M.summarise(g)["avg_aht"])


def test_sla_breach_by_channel():
    created = pd.Series(pd.to_datetime(["2025-01-01 10:00"] * 5))
    first = pd.Series(pd.to_datetime(["2025-01-01 10:15", "2025-01-01 10:16", "2025-01-01 12:00",
                                      "2025-01-01 14:01", "2025-01-01 18:00"]))
    ch = pd.Series(["chat", "chat", "voice", "social", "email"])
    assert M.sla_breach(created, first, ch).tolist() == [False, True, False, True, False]


def test_replacement_cost_uses_unit_cost_plus_340():
    c = M.replacement_cost([1480, 90, 2650], [True, True, False])
    assert c[0] == 1820 and c[1] == 430 and np.isnan(c[2])   # never the Rs 2,500 email figure


def _agents_frame():
    rows = []
    def add(aid, tier, team, n_rated, pos):
        for i in range(n_rated):
            rows.append(dict(agent_id=aid, agent_name=aid, agent_team=team, tier=tier, site="x", shift="x",
                             assigned_team=team, csat=5.0 if i < pos else 1.0, aht_min=10.0, sla_breach=False,
                             replacement_flag=False, replacement_cost=np.nan, planned_replacement_cost=500.0,
                             transfers=0))
    add("T1_LOW", 1, "A", 60, 6)      # 10% CSAT, enough sample
    add("T1_OK", 1, "A", 60, 54)      # 90% CSAT
    add("T1_SMALL", 1, "A", 49, 0)    # 0% CSAT but only 49 rated -> must not be ranked
    add("T2_LOW", 2, "Esc", 80, 0)    # Tier 2 with 0% CSAT -> must not be ranked
    return pd.DataFrame(rows)


def test_tier2_excluded_from_tier1_ranking():
    ag = S.agent_table(_agents_frame(), tiers=(1, 2))
    r = S.ranking(ag)
    assert "T2_LOW" not in r["agent_id"].tolist() and (r["tier"] == 1).all()


def test_minimum_rated_threshold():
    ag = S.agent_table(_agents_frame())
    r = S.ranking(ag)
    assert "T1_SMALL" not in r["agent_id"].tolist()
    assert ag.set_index("agent_id").loc["T1_SMALL", "status"] == S.STATUS_LOW_N
    assert r["agent_id"].tolist()[0] == "T1_LOW"


# ---- leave-one-out / case-mix tests -------------------------------------------------
def _peer_frame():
    """Team X: agent CAND (cat 'a' 10 tix, cat 'b' 10 tix), peers P1+P2 (cat 'a' 40 tix, cat 'b' 10 tix)."""
    rows = []
    def add(aid, cat, n, n_repl, n_rated=None, pos=0):
        for i in range(n):
            rows.append(dict(ticket_id=f"{aid}{cat}{i}", agent_id=aid, agent_name=aid, agent_team="X", tier=1,
                             site="x", shift="x", assigned_team="X", category=cat,
                             csat=(5.0 if i < pos else 1.0), aht_min=10.0, sla_breach=False,
                             replacement_flag=i < n_repl, replacement_cost=(1000.0 if i < n_repl else np.nan),
                             planned_replacement_cost=1000.0, transfers=0, created_at=pd.Timestamp("2026-01-01")))
    add("CAND", "a", 10, 5)      # 50% replacement in 'a'
    add("CAND", "b", 10, 5)      # 50% replacement in 'b'
    add("P1", "a", 20, 2); add("P2", "a", 20, 2)   # peers 'a': 4/40 = 10%
    add("P1", "b", 5, 1); add("P2", "b", 5, 1)     # peers 'b': only 10 tickets (<20) -> fallback
    return pd.DataFrame(rows)


def test_leave_one_out_team_baseline():
    df = _peer_frame()
    ag = S.agent_table(df, min_rated=1).set_index("agent_id")
    # peers P1+P2 overall: 6 replacements / 50 tickets = 12%; pooled incl. CAND would be 16/70
    assert abs(ag.loc["CAND", "team_replacement_rate"] - 12.0) < 1e-9
    assert abs(ag.loc["CAND", "pooled_team_replacement_rate"] - 16 / 70 * 100) < 1e-9
    # P1's baseline excludes P1 (CAND + P2: 10+... ) -> must differ from pooled
    assert abs(ag.loc["P1", "team_replacement_rate"] - ag.loc["P1", "pooled_team_replacement_rate"]) > 1e-6


def test_category_level_replacement_baseline():
    overall, cat = S.peer_replacement_rates(_peer_frame(), "X", "CAND")
    assert abs(cat["a"] - 0.10) < 1e-9          # 4 / 40


def test_fallback_when_category_peer_count_below_20():
    overall, cat = S.peer_replacement_rates(_peer_frame(), "X", "CAND")
    assert "b" not in cat and abs(overall - 0.12) < 1e-9
    e = S.agent_exposure(_peer_frame(), "CAND", "X")
    assert e["tickets_category_baseline"] == 10 and e["tickets_team_fallback"] == 10


def test_case_mix_adjusted_excess_exposure():
    e = S.agent_exposure(_peer_frame(), "CAND", "X")
    # expected: 10 'a' tickets x 10% x 1000 + 10 'b' tickets x 12% x 1000 = 1000 + 1200
    assert abs(e["expected_case_mix"] - 2200.0) < 1e-6
    assert e["actual_replacement_spend"] == 10000.0
    assert abs(e["excess_exposure"] - 7800.0) < 1e-6
    assert abs(e["expected_team_rate_only"] - 2400.0) < 1e-6   # flat-rate comparison differs


def test_excess_never_negative():
    df = _peer_frame()
    df.loc[df.agent_id == "CAND", ["replacement_flag", "replacement_cost"]] = [False, np.nan]
    assert S.agent_exposure(df, "CAND", "X")["excess_exposure"] == 0.0


def test_scaled_quarterly_opportunity():
    # recovered Rs 10,000 observed over 1,000 tickets -> Rs 10/ticket x 650 x 13 = 84,500
    assert abs(S.scale_to_quarter(10000, 1000) - 84500) < 1e-6
    assert S.RECOVERY_RATE == 0.5


def test_customer_name_masking_before_nlp():
    df = pd.DataFrame({"customer_message": ["Hi, I am Zzyx Qwerty, order VR123456, call 9876543210 or a@b.com. left bud dead"],
                       "agent_notes": ["called zzyx back"]})
    docs = T.build_docs(df, {"Zzyx", "Qwerty"})
    d = docs.iloc[0]
    for leak in ("zzyx", "qwerty", "123456", "9876543210", "a b com"):
        assert leak not in d
    assert "bud" in d


# ---- final-patch tests ---------------------------------------------------------------
def test_nlp_peer_baseline_excludes_selected_agent():
    rows = []
    for aid, theme, n in [("A", "x", 10), ("P1", "y", 10), ("P2", "y", 10)]:
        for i in range(n):
            rows.append(dict(ticket_id=f"{aid}{i}", agent_id=aid, agent_team="X", tier=1, replacement_flag=False))
    w = pd.DataFrame(rows)
    tt = pd.DataFrame({"ticket_id": w["ticket_id"], "topic_label": np.where(w["agent_id"] == "A", "x", "y"),
                       "topic_strength": 1.0})
    th = T.agent_themes(tt, w, "A").set_index("topic_label")
    # peers (P1+P2) have 0% of theme 'x'; a pooled baseline incl. A would give 33%
    assert th.loc["x", "peer_share_pct"] == 0.0 and th.loc["x", "agent_share_pct"] == 100.0


def test_recovery_sensitivity_scenarios():
    imp = {"excess_exposure": 10000.0, "window_tickets_all_tiers": 1000}
    sc = S.recovery_scenarios(imp).set_index("recovery_scenario")["modeled_quarterly_opportunity_inr"]
    assert abs(sc["25%"] - 21125) < 1e-6 and abs(sc["50%"] - 42250) < 1e-6 and abs(sc["75%"] - 63375) < 1e-6


def test_category_detail_baseline_used():
    cd = S.category_detail(_peer_frame(), "CAND", "X").set_index("category")
    assert cd.loc["a", "baseline_used"] == "category" and cd.loc["b", "baseline_used"] == "team fallback"
    assert cd.loc["a", "peer_tickets"] == 40 and abs(cd.loc["a", "peer_replacement_rate"] - 10.0) < 1e-9


def test_export_selection_tier1_only():
    ag = S.agent_table(_agents_frame(), tiers=(1, 2))
    ex = S.export_tables(ag)
    assert (ag.loc[ag["tier"] == 2, "agent_id"].isin(ex["tier1_agent_metrics"]["agent_id"]) == False).all()
    assert set(ex["raw_bottom10"]["agent_id"]) <= set(ex["tier1_agent_metrics"]["agent_id"])
    assert "T1_SMALL" not in ex["raw_bottom10"]["agent_id"].tolist()


def test_review_sample_balanced_and_masked():
    from src import validation as V
    import tempfile
    df = _peer_frame().assign(customer_message="call 9876543210", agent_notes="ok")
    tt = pd.DataFrame({"ticket_id": df["ticket_id"], "topic_label": "t"})
    p = V.make_review_sample(df, tt, ["P1", "P2"], n=10, path=Path(tempfile.mkdtemp()) / "s.csv")
    r = pd.read_csv(p)
    assert r["agent_id"].value_counts().to_dict() == {"P1": 5, "P2": 5}
    assert "9876543210" not in r.to_csv() and r["judged_relevant"].isna().all()


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f(); print("PASS", f.__name__)
    print(f"{len(fns)} tests passed")
