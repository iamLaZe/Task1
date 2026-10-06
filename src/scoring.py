"""Transparent rule-based ranking, coaching shortlist and business impact. No ML scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M

MIN_RATED = 50
CSAT_GAP_PP = 5.0          # agent CSAT must be >= 5pp BELOW team baseline
REPL_GAP_PP = 3.0          # agent replacement rate must be >= 3pp ABOVE team baseline
RECOVERY_RATE = 0.50
TICKETS_PER_WEEK = 650
WEEKS_PER_QUARTER = 13

STATUS_PRIORITY = "Coaching priority"
STATUS_B10 = "Bottom 10 — review context"
STATUS_LOW_N = "Insufficient CSAT sample"
STATUS_OK = "Not flagged"


def latest_window(tickets: pd.DataFrame, months: int = 12) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = tickets["created_at"].max().normalize()
    start = end - pd.DateOffset(months=months) + pd.Timedelta(days=1)
    return start, end


def apply_window(tickets: pd.DataFrame, mode: str) -> pd.DataFrame:
    """mode: 'latest12' (default for the current decision) or 'historical' (all 18 months)."""
    if mode == "historical":
        return tickets
    start, _ = latest_window(tickets)
    return tickets[tickets["created_at"] >= start]


def tier1(tickets: pd.DataFrame) -> pd.DataFrame:
    return tickets[tickets["tier"] == 1]


def agent_table(window: pd.DataFrame, tiers=(1,), min_rated: int = MIN_RATED) -> pd.DataFrame:
    """One row per agent (grouped by agent_id) with team baselines, gaps and status.

    Baselines are LEAVE-ONE-OUT peers: same Tier-1 team, excluding the agent being evaluated.
    Tier 2 agents are never given Tier-1 baselines or a coaching status.
    """
    base = window[window["tier"].isin(tiers)]
    meta = (base.groupby("agent_id")[["agent_name", "agent_team", "tier", "site", "shift"]].first())
    rows = base.groupby("agent_id").apply(M.summarise, include_groups=False)
    ag = meta.join(rows).reset_index().rename(columns={"agent_name": "name", "agent_team": "team"})

    # assigned_team context: share of the agent's tickets first routed to a different team
    mism = base.assign(m=base["assigned_team"] != base["agent_team"]).groupby("agent_id")["m"].mean() * 100
    ag["routed_elsewhere_pct"] = ag["agent_id"].map(mism)

    t1 = window[window["tier"] == 1]
    # Pooled team metrics (display only) ...
    pooled = t1.groupby("agent_team").apply(M.summarise, include_groups=False)
    ag["pooled_team_csat_pct"] = ag["team"].map(pooled["csat_pct"])
    ag["pooled_team_replacement_rate"] = ag["team"].map(pooled["replacement_rate"])
    # ... but every agent-vs-peer comparison uses LEAVE-ONE-OUT peers: same Tier-1 team, excluding the agent.
    peer_rows = {}
    for _, r in ag[ag["tier"] == 1].iterrows():
        peers = t1[(t1["agent_team"] == r["team"]) & (t1["agent_id"] != r["agent_id"])]
        p = M.summarise(peers) if len(peers) else pd.Series(dtype=float)
        peer_rows[r["agent_id"]] = {
            "team_csat_pct": p.get("csat_pct", np.nan), "team_replacement_rate": p.get("replacement_rate", np.nan),
            "team_avg_aht": p.get("avg_aht", np.nan), "team_median_aht": p.get("median_aht", np.nan),
            "team_sla_breach_pct": p.get("sla_breach_pct", np.nan), "team_rated": p.get("rated", 0),
            "peer_agents": peers["agent_id"].nunique()}
    peer = pd.DataFrame.from_dict(peer_rows, orient="index")
    ag = ag.merge(peer, left_on="agent_id", right_index=True, how="left")
    ag["csat_gap"] = ag["csat_pct"] - ag["team_csat_pct"]
    ag["replacement_gap"] = ag["replacement_rate"] - ag["team_replacement_rate"]
    ag["eligible"] = ag["rated"] >= min_rated
    ag.loc[ag["tier"] != 1, ["team_csat_pct", "team_replacement_rate", "csat_gap", "replacement_gap"]] = np.nan

    # ---- ranking (Tier 1, eligible only)
    ag["rank_low_to_high"] = np.nan
    el = ag[(ag["tier"] == 1) & ag["eligible"]].sort_values(["csat_pct", "avg_rating"])
    ag.loc[el.index, "rank_low_to_high"] = np.arange(1, len(el) + 1)
    ag["in_bottom10"] = ag["rank_low_to_high"] <= 10
    ag["in_top5"] = ag["rank_low_to_high"] > (len(el) - 5)

    # ---- transparent rules
    ag["meets_csat_rule"] = ag["csat_gap"] <= -CSAT_GAP_PP
    ag["meets_repl_rule"] = ag["replacement_gap"] >= REPL_GAP_PP
    ag["coaching_priority"] = (ag["tier"] == 1) & ag["eligible"] & ag["meets_csat_rule"] & ag["meets_repl_rule"]

    def status(r):
        if r["tier"] != 1:
            return "Tier 2 — not ranked"
        if not r["eligible"]:
            return STATUS_LOW_N
        if r["coaching_priority"]:
            return STATUS_PRIORITY
        if r["in_bottom10"]:
            return STATUS_B10
        return STATUS_OK
    ag["status"] = ag.apply(status, axis=1)
    return ag.sort_values("agent_id").reset_index(drop=True)


def ranking(ag: pd.DataFrame, min_rated: int = MIN_RATED) -> pd.DataFrame:
    """Tier-1 agents with >= min_rated rated tickets, lowest CSAT % first."""
    r = ag[(ag["tier"] == 1) & (ag["rated"] >= min_rated)]
    return r.sort_values(["csat_pct", "avg_rating"]).reset_index(drop=True)


def bottom10(ag: pd.DataFrame, min_rated: int = MIN_RATED) -> pd.DataFrame:
    return ranking(ag, min_rated).head(10)


def top5(ag: pd.DataFrame, min_rated: int = MIN_RATED) -> pd.DataFrame:
    return ranking(ag, min_rated).sort_values(["csat_pct", "avg_rating"], ascending=False).head(5)


def pooled_csat(window: pd.DataFrame, agent_ids) -> float:
    return M.csat_stats(window[window["agent_id"].isin(agent_ids)]["csat"])["csat_pct"]


# ---- Business impact ---------------------------------------------------------------
MIN_CATEGORY_PEER_TICKETS = 20


def peer_replacement_rates(t1: pd.DataFrame, team: str, agent_id: str, min_n: int = MIN_CATEGORY_PEER_TICKETS):
    """Replacement rate (0-1) among OTHER Tier-1 agents of the same team: overall and per category.
    Categories with fewer than min_n peer tickets are omitted (caller falls back to the overall rate)."""
    peers = t1[(t1["agent_team"] == team) & (t1["agent_id"] != agent_id)]
    overall = peers["replacement_flag"].mean() if len(peers) else np.nan
    g = peers.groupby("category")["replacement_flag"].agg(["mean", "size"])
    cat = g.loc[g["size"] >= min_n, "mean"].to_dict()
    return overall, cat


def agent_exposure(window: pd.DataFrame, agent_id: str, team: str, min_n: int = MIN_CATEGORY_PEER_TICKETS) -> dict:
    """Case-mix-aware expected replacement cost for one agent.

    expected(ticket) = peer rate for the ticket's category (if >= min_n peer tickets, else leave-one-out
    team-wide rate) x (unit cost + Rs 340). Excess = max(0, actual - expected).
    """
    t1 = window[window["tier"] == 1]
    overall, cat = peer_replacement_rates(t1, team, agent_id, min_n)
    g = window[window["agent_id"] == agent_id]
    rate = g["category"].map(cat)
    used_cat = rate.notna()
    rate = rate.fillna(overall)
    expected = float((rate * g["planned_replacement_cost"]).sum())
    expected_simple = float((overall * g["planned_replacement_cost"]).sum())
    actual = float(g["replacement_cost"].sum())
    return {
        "tickets": len(g), "replacements": int(g["replacement_flag"].sum()),
        "actual_replacement_spend": actual,
        "expected_case_mix": expected,
        "excess_exposure": max(0.0, actual - expected),
        "expected_team_rate_only": expected_simple,
        "excess_team_rate_only": max(0.0, actual - expected_simple),
        "tickets_category_baseline": int(used_cat.sum()),
        "tickets_team_fallback": int((~used_cat).sum()),
    }


def scale_to_quarter(recovered: float, window_tickets: int,
                     tickets_per_week: int = TICKETS_PER_WEEK, weeks: int = WEEKS_PER_QUARTER) -> float:
    """Observed per-ticket opportunity x stated current quarterly ticket volume."""
    return recovered / window_tickets * tickets_per_week * weeks if window_tickets else 0.0


def business_impact(window: pd.DataFrame, ag: pd.DataFrame, all_tickets_in_window: int | None = None) -> dict:
    """Modeled direct-cost opportunity (case-mix adjusted) for the coaching-priority cohort.

    Associated replacement-cost exposure only - not proven agent-caused cost, not guaranteed savings.
    Opportunity = 50% of positive excess, scaled from observed all-tier ticket volume (preserving the
    observed Tier-1 share) to 650 tickets/week x 13 weeks.
    """
    cand = ag[ag["coaching_priority"]]
    detail = []
    for _, r in cand.iterrows():
        d = agent_exposure(window, r["agent_id"], r["team"])
        detail.append({"agent_id": r["agent_id"], "name": r["name"], "team": r["team"], **d})
    det = pd.DataFrame(detail)
    total_tickets = all_tickets_in_window if all_tickets_in_window is not None else len(window)
    t1_tickets = int((window["tier"] == 1).sum())
    t1_repl_spend = float(window.loc[window["tier"] == 1, "replacement_cost"].sum())
    col = lambda c: float(det[c].sum()) if len(det) else 0.0
    excess = col("excess_exposure")
    recovered = excess * RECOVERY_RATE
    cand_tickets = int(col("tickets"))
    cand_spend = col("actual_replacement_spend")
    return {
        "detail": det,
        "n_candidates": len(det),
        "candidate_ticket_share_t1": cand_tickets / t1_tickets * 100 if t1_tickets else np.nan,
        "candidate_spend_share_t1": cand_spend / t1_repl_spend * 100 if t1_repl_spend else np.nan,
        "actual_spend": cand_spend,
        "expected_case_mix": col("expected_case_mix"),
        "excess_exposure": excess,
        "excess_team_rate_only": col("excess_team_rate_only"),
        "modeled_recovery": recovered,
        "tickets_category_baseline": int(col("tickets_category_baseline")),
        "tickets_team_fallback": int(col("tickets_team_fallback")),
        "window_tickets_all_tiers": total_tickets,
        "window_tickets_tier1": t1_tickets,
        "tier1_share": t1_tickets / total_tickets * 100 if total_tickets else np.nan,
        "quarter_tickets": TICKETS_PER_WEEK * WEEKS_PER_QUARTER,
        "quarterly_opportunity": scale_to_quarter(recovered, total_tickets),
        "quarterly_opportunity_team_rate_only": scale_to_quarter(col("excess_team_rate_only") * RECOVERY_RATE, total_tickets),
        "annual_opportunity": scale_to_quarter(recovered, total_tickets) * 4,
        "opportunity_at_dataset_volume_quarter": recovered / 52.18 * WEEKS_PER_QUARTER,
        "sla_credit_secondary": sla_secondary(window, cand["agent_id"]),
    }


def sla_secondary(window: pd.DataFrame, agent_ids) -> dict:
    """SLA credit cost shown as a secondary metric only (not part of the headline)."""
    g = window[window["agent_id"].isin(agent_ids)]
    n = int(g["sla_breach"].sum())
    return {"breaches": n, "credit_inr": n * M.SLA_CREDIT_INR}


def recovery_scenarios(imp: dict, rates=(0.25, 0.50, 0.75)) -> pd.DataFrame:
    """Scenario table (not forecasts): quarterly opportunity at different recovery assumptions."""
    return pd.DataFrame([{
        "recovery_scenario": f"{int(r * 100)}%",
        "modeled_quarterly_opportunity_inr": scale_to_quarter(imp["excess_exposure"] * r, imp["window_tickets_all_tiers"]),
        "headline": r == RECOVERY_RATE,
    } for r in rates])


def category_detail(window: pd.DataFrame, agent_id: str, team: str, min_n: int = MIN_CATEGORY_PEER_TICKETS) -> pd.DataFrame:
    """Per-category case-mix table for one Tier-1 agent vs leave-one-out same-team peers."""
    t1 = window[window["tier"] == 1]
    peers = t1[(t1["agent_team"] == team) & (t1["agent_id"] != agent_id)]
    me = window[window["agent_id"] == agent_id]
    a = me.groupby("category").agg(agent_tickets=("ticket_id", "size"), agent_replacement_rate=("replacement_flag", "mean"))
    p = peers.groupby("category").agg(peer_tickets=("ticket_id", "size"), peer_replacement_rate=("replacement_flag", "mean"))
    d = a.join(p).fillna({"peer_tickets": 0})
    d["peer_tickets"] = d["peer_tickets"].astype(int)
    d["baseline_used"] = np.where(d["peer_tickets"] >= min_n, "category", "team fallback")
    d[["agent_replacement_rate", "peer_replacement_rate"]] *= 100
    return d.sort_values("agent_tickets", ascending=False).reset_index()


EXPORT_COLS = ["agent_id", "name", "team", "tier", "tickets", "rated", "csat_pct", "avg_rating", "avg_aht",
               "median_aht", "sla_breach_pct", "replacement_rate", "replacement_spend", "team_csat_pct",
               "team_replacement_rate", "csat_gap", "replacement_gap", "status"]


def export_tables(ag: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tables offered as CSV downloads. Tier 1 only; no customer data."""
    t1 = ag[ag["tier"] == 1]
    return {
        "tier1_agent_metrics": t1[EXPORT_COLS].round(2),
        "raw_bottom10": bottom10(ag)[EXPORT_COLS].round(2),
        "coaching_priority": ag[ag["coaching_priority"]][EXPORT_COLS].round(2),
    }
