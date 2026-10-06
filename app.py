"""Vireo Support Coach - Streamlit app. Run: streamlit run app.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import data, metrics as M, scoring as S, text_analysis as T, validation as V

st.set_page_config(page_title="Vireo Support Coach", page_icon="🎧", layout="wide")

COLS = {
    "agent_id": "agent_id", "name": "name", "team": "team", "tier": "tier", "tickets": "tickets",
    "rated": "rated", "csat_pct": "CSAT %", "avg_rating": "avg CSAT rating", "avg_aht": "avg AHT (min)",
    "median_aht": "median AHT (min)", "sla_breach_pct": "SLA breach %", "replacement_rate": "replacement rate %",
    "replacement_spend": "replacement spend ₹", "csat_gap": "CSAT gap (pp)",
    "replacement_gap": "replacement gap (pp)", "status": "coaching status",
}


@st.cache_data(show_spinner="Loading data and fitting local NLP topics…")
def load_all():
    raw, t = data.load()
    names = T.name_tokens(raw["customers"])
    tt, topics = T.fit_topics(t, names)
    return raw, t, tt, topics, names


def show(df: pd.DataFrame, cols=None, rename=True):
    d = df[cols] if cols else df
    d = d.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(2 if c == "avg_rating" else 1)
    if rename:
        d = d.rename(columns=COLS)
    st.dataframe(d, hide_index=True, width="stretch")


def inr(x):
    return "n/a" if pd.isna(x) else f"₹{x:,.0f}"


def pct(x):
    return "n/a" if pd.isna(x) else f"{x:.1f}%"


def minutes(x):
    if pd.isna(x):
        return "n/a"
    return f"{x:,.0f} min" if x < 180 else f"{x:,.0f} min (~{x/60:.1f} h)"


try:
    raw, tickets, tt, topics, names = load_all()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Copy tickets.csv, agents.csv and products.csv (customers.csv optional) into the data/ folder, "
            "or set VIREO_DATA_DIR. See README.")
    st.stop()

start12, end12 = S.latest_window(tickets)
page = st.sidebar.radio("Page", ["1 · Executive overview", "2 · Agent performance", "3 · Agent detail",
                                 "4 · Data quality & validation"])
mode = "latest12"
if page[0] in "23":
    choice = st.sidebar.radio("Time window", ["Latest 12 months (current decision)", "Historical (all 18 months)"])
    mode = "latest12" if choice.startswith("Latest") else "historical"
st.sidebar.caption(f"Data: {tickets['created_at'].min():%d %b %Y} – {tickets['created_at'].max():%d %b %Y}. "
                   f"Latest-12-month window starts {start12:%d %b %Y}.")
st.sidebar.caption("Paid AI/API calls: 0. NLP runs locally.")

# =========================================================== PAGE 1
if page.startswith("1"):
    w = S.apply_window(tickets, "latest12")
    ag = S.agent_table(w)
    b10 = S.bottom10(ag)
    pri = ag[ag["coaching_priority"]]
    imp = S.business_impact(w, ag)
    t1w = S.tier1(w)

    st.title("Vireo Support Coach")
    st.caption(f"Current view = latest 12 months ({start12:%d %b %Y} – {end12:%d %b %Y}), Tier 1 only. "
               "The CSAT trend chart uses the full 18 months.")
    st.info(f"**Decision summary:** Review **{len(pri)} Coaching Priority agents** first; keep the raw Bottom 10 "
            f"for context. Modeled, not guaranteed; associated exposure, not proven agent-caused cost.")
    k = st.columns(6)
    k[0].metric("Tier-1 CSAT", pct(M.csat_stats(t1w["csat"])["csat_pct"]))
    k[1].metric("Tier-1 median AHT", minutes(t1w["aht_min"].median()))
    k[2].metric("Tier-1 tickets", f"{len(t1w):,}")
    k[3].metric("Bottom-10 CSAT", pct(S.pooled_csat(w, b10["agent_id"])))
    k[4].metric("Coaching-priority agents", len(pri))
    k[5].metric("Modeled quarterly opportunity", inr(imp["quarterly_opportunity"]))
    st.caption("Modeled direct-cost opportunity (case-mix adjusted): 50% recovery of excess associated replacement-cost "
               "exposure, scaled to 650 tickets/week × 13 weeks. A model, not a guaranteed saving. The dataset itself averages only "
               f"~{imp['window_tickets_all_tiers']/(365/7):.0f} tickets/week; at that observed volume the same "
               f"logic gives {inr(imp['opportunity_at_dataset_volume_quarter'])} per quarter.")

    c1, c2 = st.columns(2)
    m = (tickets[tickets["tier"] == 1].groupby("month")
         .apply(lambda g: pd.Series(M.csat_stats(g["csat"])), include_groups=False).reset_index())
    f1 = px.line(m, x="month", y="csat_pct", markers=True, hover_data=["rated"],
                 title="Tier-1 CSAT % by month (all 18 months)", labels={"csat_pct": "CSAT %", "month": ""})
    f1.add_vrect(x0=start12, x1=end12, fillcolor="orange", opacity=0.12, line_width=0)
    f1.update_yaxes(range=[0, 100])
    c1.plotly_chart(f1, width="stretch")

    el = S.ranking(ag)
    f2 = px.scatter(el, x="median_aht", y="csat_pct", color="status", size="rated", hover_name="name",
                    hover_data=["agent_id", "team"], log_x=True,
                    title="CSAT % vs median AHT (Tier 1, ≥50 rated, log x-axis)",
                    labels={"median_aht": "median AHT (min, log)", "csat_pct": "CSAT %"})
    c2.plotly_chart(f2, width="stretch")

    if len(pri):
        long = pri.melt(id_vars=["agent_id", "name"], value_vars=["replacement_rate", "team_replacement_rate"],
                        var_name="series", value_name="rate")
        long["series"] = long["series"].map({"replacement_rate": "Agent", "team_replacement_rate": "Own team baseline"})
        long["label"] = long["name"] + " (" + long["agent_id"] + ")"
        f3 = px.bar(long, x="label", y="rate", color="series", barmode="group",
                    title="Replacement rate %: coaching candidates vs own team baseline",
                    labels={"rate": "replacement rate %", "label": ""})
        st.plotly_chart(f3, width="stretch")

    st.subheader("Raw Bottom 10 (as requested by the client)")
    show(b10, ["agent_id", "name", "team", "tickets", "rated", "csat_pct", "avg_rating", "avg_aht", "median_aht",
               "sla_breach_pct", "replacement_rate", "replacement_spend", "status"])
    st.subheader("Coaching priority (team-context rules)")
    if len(pri):
        show(pri, ["agent_id", "name", "team", "tickets", "rated", "csat_pct", "team_csat_pct", "csat_gap",
                   "replacement_rate", "team_replacement_rate", "replacement_gap", "replacement_spend"],
             rename=False)
    else:
        st.write("No agent meets all four rules in this window.")

    st.subheader("Business case: associated replacement-cost exposure (latest 12 months)")
    dd = imp["detail"]
    if len(dd):
        show(dd, ["agent_id", "name", "tickets", "replacements", "actual_replacement_spend", "expected_case_mix",
                  "excess_exposure"], rename=False)
    st.caption(f"Expected = peer replacement rate for each ticket's category (other Tier-1 agents on the same team, "
               f"≥{S.MIN_CATEGORY_PEER_TICKETS} peer tickets, else leave-one-out team rate) × (unit cost + ₹340). "
               f"Total actual {inr(imp['actual_spend'])}, expected {inr(imp['expected_case_mix'])}, "
               f"excess {inr(imp['excess_exposure'])}. Using one flat team rate instead would give "
               f"{inr(imp['excess_team_rate_only'])} excess.")

    st.markdown("**Flat team rate vs case-mix-adjusted excess exposure**")
    cm = pd.DataFrame({
        "method": ["Flat leave-one-out team rate (ignores ticket mix)", "Case-mix-adjusted (headline)"],
        "excess exposure ₹": [imp["excess_team_rate_only"], imp["excess_exposure"]],
        "modeled quarterly opportunity ₹ (50%)": [imp["quarterly_opportunity_team_rate_only"], imp["quarterly_opportunity"]],
    }).round(0)
    st.dataframe(cm, hide_index=True, width="stretch")
    st.caption("Candidates handle a different mix of ticket categories (e.g. more charging/battery and warranty issues) "
               "than their peers. A flat team rate would overstate the excess; comparing like-for-like categories "
               "prevents that overstatement. The business calculation uses the case-mix-adjusted figure.")

    st.markdown("**Recovery scenarios (illustrative scenarios — not forecasts, not guaranteed savings)**")
    sc = S.recovery_scenarios(imp)
    sc["modeled_quarterly_opportunity_inr"] = sc["modeled_quarterly_opportunity_inr"].round(0)
    sc["headline"] = sc["headline"].map({True: "headline", False: ""})
    st.dataframe(sc.rename(columns={"recovery_scenario": "recovery of excess",
                                    "modeled_quarterly_opportunity_inr": "modeled quarterly opportunity ₹"}),
                 hide_index=True, width="stretch")

    st.markdown("**Downloads (CSV)**")
    ex = S.export_tables(ag)
    d = st.columns(3)
    for col, (key, label) in zip(d, [("tier1_agent_metrics", "Full Tier-1 agent metrics"),
                                     ("raw_bottom10", "Raw Bottom 10"),
                                     ("coaching_priority", "Coaching Priority shortlist")]):
        col.download_button(label, ex[key].to_csv(index=False).encode("utf-8"), file_name=f"{key}.csv",
                            mime="text/csv", key=f"dl_{key}")

    st.subheader("Recommendation")
    st.markdown(
        f"- The client's **raw Bottom 10** is kept above. Some specialised queues receive more difficult customer "
        f"interactions by design, so raw CSAT should be interpreted with team and case-mix context "
        f"(those agents are flagged *Bottom 10 — review context* unless the rules below are met).\n"
        f"- Comparing each agent with **peers on the same team (excluding the agent)**: CSAT ≥{S.CSAT_GAP_PP:.0f}pp below "
        f"**and** replacement rate ≥{S.REPL_GAP_PP:.0f}pp above narrows the shortlist to **{len(pri)} agents** "
        f"({', '.join(pri['name'] + ' ' + pri['agent_id'])}). Review them for coaching.\n"
        f"- They handle **{imp['candidate_ticket_share_t1']:.1f}%** of Tier-1 tickets but carry "
        f"**{imp['candidate_spend_share_t1']:.1f}%** of Tier-1 replacement spend (associated exposure — not proven "
        f"agent-caused).\n"
        f"- **Modeled direct-cost opportunity:** about **{inr(imp['quarterly_opportunity'])} per quarter** "
        f"(50% recovery of {inr(imp['excess_exposure'])} case-mix-adjusted excess over 12 months, scaled to "
        f"650 tickets/week). Modeled, not guaranteed. No ₹ value is placed on CSAT, and AHT is not converted to savings.")

# =========================================================== PAGE 2
elif page.startswith("2"):
    st.title("Agent performance")
    w = S.apply_window(tickets, mode)
    ag = S.agent_table(w, tiers=(1, 2))
    st.caption(f"Window: {w['created_at'].min():%d %b %Y} – {w['created_at'].max():%d %b %Y}. "
               "Tier 1 ranked on CSAT among agents with ≥50 rated tickets. Tier 2 shown separately, never ranked.")
    teams = sorted(ag.loc[ag["tier"] == 1, "team"].unique())
    sel = st.multiselect("Team (Tier 1)", teams, default=teams)
    main = ag[(ag["tier"] == 1) & ag["eligible"] & ag["team"].isin(sel)].sort_values("csat_pct")
    tab = ["agent_id", "name", "team", "tier", "tickets", "rated", "csat_pct", "avg_aht", "median_aht",
           "sla_breach_pct", "replacement_rate", "replacement_spend", "csat_gap", "replacement_gap", "status"]
    show(main, tab)
    st.caption("Gaps are vs. leave-one-out peers: the agent's own Tier-1 team excluding the agent. AHT is skewed: use the median. "
               "AHT differs structurally by team (Logistics/Returns work is asynchronous), so compare within team.")

    st.subheader("Raw Bottom 10 (client request)")
    show(S.bottom10(ag), ["agent_id", "name", "team", "tickets", "rated", "csat_pct", "avg_rating", "avg_aht",
                          "median_aht", "sla_breach_pct", "replacement_rate", "replacement_spend", "status"])
    st.subheader("Top 5 (same rule: Tier 1, ≥50 rated) — reference only, no bonus allocation")
    show(S.top5(ag), ["agent_id", "name", "team", "tickets", "rated", "csat_pct", "median_aht", "sla_breach_pct",
                      "replacement_rate"])

    low = ag[(ag["tier"] == 1) & ~ag["eligible"] & ag["team"].isin(sel)]
    st.subheader("Insufficient CSAT sample (<50 rated tickets — not ranked)")
    if len(low):
        show(low, ["agent_id", "name", "team", "tickets", "rated", "csat_pct", "median_aht", "status"])
    else:
        st.write("None.")

    st.subheader("Tier 2 — Escalations & Warranty (shown separately, not compared with Tier 1)")
    t2 = ag[ag["tier"] == 2].copy()
    t2["median_aht_days"] = t2["median_aht"] / 1440
    show(t2, ["agent_id", "name", "team", "tickets", "rated", "csat_pct", "median_aht_days", "sla_breach_pct",
              "replacement_rate"])
    st.caption("Policy §6: Tier 2 cases are multi-touch and measured on resolution in days, not volume.")

# =========================================================== PAGE 3
elif page.startswith("3"):
    st.title("Agent detail")
    w = S.apply_window(tickets, mode)
    ag = S.agent_table(w, tiers=(1, 2))
    ag["label"] = ag["agent_id"] + " — " + ag["name"] + " (" + ag["team"] + ")"
    default = ag.index[ag["coaching_priority"]][0] if ag["coaching_priority"].any() else 0
    r = ag.loc[st.selectbox("Agent", ag.index, index=int(default), format_func=lambda i: ag.loc[i, "label"])]
    st.caption(f"Window: {w['created_at'].min():%d %b %Y} – {w['created_at'].max():%d %b %Y}")

    st.subheader(f"{r['name']} · {r['agent_id']} · {r['status']}")
    a = st.columns(4)
    a[0].metric("Team", r["team"]); a[1].metric("Tier", int(r["tier"]))
    a[2].metric("Tickets", int(r["tickets"])); a[3].metric("Rated tickets", int(r["rated"]))
    b = st.columns(4)
    b[0].metric("CSAT %", pct(r["csat_pct"]), None if pd.isna(r["csat_gap"]) else f"{r['csat_gap']:+.1f} pp vs team")
    b[1].metric("Avg CSAT rating", "n/a" if pd.isna(r["avg_rating"]) else f"{r['avg_rating']:.2f}")
    b[2].metric("Median AHT", minutes(r["median_aht"])); b[3].metric("Avg AHT", minutes(r["avg_aht"]))
    c = st.columns(4)
    c[0].metric("SLA breach %", pct(r["sla_breach_pct"]))
    c[1].metric("Replacement rate", pct(r["replacement_rate"]),
                None if pd.isna(r["replacement_gap"]) else f"{r['replacement_gap']:+.1f} pp vs team",
                delta_color="inverse")
    c[2].metric("Replacement spend", inr(r["replacement_spend"]))
    c[3].metric("Transfers", int(r["transfers"]))

    if r["tier"] == 1:
        st.markdown("**Peer baseline (own Tier-1 team, excluding this agent)**")
        base = pd.DataFrame({
            "metric": ["CSAT %", "replacement rate %", "avg AHT (min)", "median AHT (min)", "SLA breach %"],
            "agent": [r["csat_pct"], r["replacement_rate"], r["avg_aht"], r["median_aht"], r["sla_breach_pct"]],
            "peer team": [r["team_csat_pct"], r["team_replacement_rate"], r["team_avg_aht"], r["team_median_aht"],
                     r["team_sla_breach_pct"]]})
        base["gap"] = base["agent"] - base["peer team"]
        show(base, rename=False)
        st.caption(f"{r['routed_elsewhere_pct']:.1f}% of this agent's tickets were first routed to a different team "
                   f"(assigned_team ≠ resolving agent's team).")

        st.markdown("#### Why this agent is flagged")
        ok = lambda v: "✅" if v else "❌"
        st.markdown(
            f"- {ok(r['eligible'])} Sample: {int(r['rated'])} rated tickets (minimum {S.MIN_RATED}).\n"
            f"- {ok(r['meets_csat_rule'])} CSAT {pct(r['csat_pct'])} vs team {pct(r['team_csat_pct'])} → gap "
            f"{r['csat_gap']:+.1f} pp (rule: ≤ −{S.CSAT_GAP_PP:.0f} pp).\n"
            f"- {ok(r['meets_repl_rule'])} Replacement rate {pct(r['replacement_rate'])} vs team "
            f"{pct(r['team_replacement_rate'])} → gap {r['replacement_gap']:+.1f} pp (rule: ≥ +{S.REPL_GAP_PP:.0f} pp).\n"
            f"- Raw Bottom 10 rank: {'#%d' % r['rank_low_to_high'] if r['in_bottom10'] else 'not in Bottom 10'}.")
        if r["coaching_priority"]:
            d = S.agent_exposure(w, r["agent_id"], r["team"])
            st.success(f"Coaching priority: all four rules met. Associated replacement-cost exposure: actual "
                       f"{inr(d['actual_replacement_spend'])} vs {inr(d['expected_case_mix'])} expected at peer category rates "
                       f"→ excess {inr(d['excess_exposure'])}. This is exposure associated with the agent's "
                       f"tickets, not proven agent-caused cost.")
        elif r["in_bottom10"]:
            st.warning("Bottom 10 — review context: low CSAT, but the team-context rules are not both met. "
                       "Check team and case-mix effects before any training decision.")
        elif not r["eligible"]:
            st.info("Insufficient CSAT sample: not ranked.")
        else:
            st.write("Not flagged.")
    else:
        st.info("Tier 2: measured on resolution in days (policy §6); not compared with Tier 1 and no coaching rule applied.")

    ag_w = w[w["agent_id"] == r["agent_id"]]
    st.markdown("#### Case mix: replacement rate by ticket category vs peers (excluding this agent)")
    if r["tier"] == 1:
        cd = S.category_detail(w, r["agent_id"], r["team"])
        show(cd.round(1), rename=False)
        st.caption(f"Peer baseline is used per category when there are ≥{S.MIN_CATEGORY_PEER_TICKETS} peer tickets; "
                   "otherwise the leave-one-out team-wide rate is the fallback.")
    else:
        cat = ag_w.groupby("category").agg(tickets=("ticket_id", "size"), replacement_rate=("replacement_flag", "mean"))
        cat["replacement_rate"] = (cat["replacement_rate"] * 100).round(1)
        show(cat.reset_index(), rename=False)

    st.markdown("#### Recurring ticket themes (local NLP — explanatory only)")
    th = T.agent_themes(tt, w, r["agent_id"])
    if len(th):
        show(th.round(1), rename=False)
    st.caption("Themes are data-derived (TF-IDF + NMF) and compared with leave-one-out same-team peers. They suggest coaching topics; they do not decide who is coached.")
    ex = T.examples(tt, w, r["agent_id"], names)
    if len(ex):
        st.markdown("**Representative tickets (PII masked)**")
        show(ex, rename=False)

# =========================================================== PAGE 4
else:
    st.title("Data quality & validation")
    w = S.apply_window(tickets, "latest12")
    ag = S.agent_table(w)
    q = V.data_quality(raw, tickets, ranked_agent_tiers=S.ranking(ag)["tier"])
    r1 = st.columns(4)
    r1[0].metric("Tickets", f"{q['tickets']:,}"); r1[1].metric("Agents", q["agents"])
    r1[2].metric("Tier 1 / Tier 2", f"{q['tier1_agents']} / {q['tier2_agents']}")
    r1[3].metric("Legacy (legacy_fd) tickets", f"{q['legacy_tickets']:,}")
    r2 = st.columns(4)
    r2[0].metric("Missing CSAT", f"{q['missing_csat']:,}"); r2[1].metric("CSAT response rate", pct(q["csat_response_rate"]))
    r2[2].metric("Invalid CSAT", q["invalid_csat"]); r2[3].metric("Unknown agent IDs", q["unknown_agent_ids"])
    r3 = st.columns(4)
    r3[0].metric("Unknown product SKUs", q["unknown_product_skus"])
    r3[1].metric("Raw negative AHT (before)", f"{q['raw_negative_aht']:,}")
    r3[2].metric("Normalized negative AHT (after)", q["normalized_negative_aht"])
    r3[3].metric("Missing resolved_at", q["missing_resolved_at"])
    r4 = st.columns(3)
    r4[0].metric("Open / pending", q["open_pending"])
    r4[1].metric("Refund + replacement anomalies", q["refund_replacement_anomalies"])
    r4[2].metric("Duplicate display names", ", ".join(q["duplicate_agent_names"]) or "none")

    st.subheader("Case-mix baseline coverage (coaching-priority tickets, latest 12 months)")
    imp = S.business_impact(w, ag)
    z = st.columns(3)
    z[0].metric("Tickets using category-level peer baseline", f"{imp['tickets_category_baseline']:,}")
    z[1].metric("Tickets using team-level fallback", f"{imp['tickets_team_fallback']:,}")
    z[2].metric("Minimum peer tickets per category", S.MIN_CATEGORY_PEER_TICKETS)
    r5 = st.columns(1)
    r5[0].metric("Duplicate ticket IDs (unique = 0 duplicates)", q["duplicate_ticket_ids"])

    st.subheader("Validation checks")
    st.dataframe(pd.DataFrame([{"check": n, "result": "PASS" if ok else "FAIL"} for n, ok in q["checks"]]),
                 hide_index=True, width="stretch")

    st.subheader("Legacy timestamp normalisation — before / after")
    lg = tickets[tickets["source_system"] == M.LEGACY_SYSTEM]
    st.dataframe(pd.DataFrame({
        "": ["Legacy tickets with resolved_at", "Negative AHT (raw, UTC as exported)",
             "Negative AHT (after +5h30m to IST)", "Median AHT raw (min)", "Median AHT normalised (min)"],
        "value": [int(lg["resolved_at"].notna().sum()), int((lg["raw_aht_min"] < 0).sum()),
                  int((lg["aht_min"] < 0).sum()), round(lg["raw_aht_min"].median(), 1), round(lg["aht_min"].median(), 1)],
    }), hide_index=True, width="stretch")

    st.subheader("Refund + replacement anomalies (policy §5: never both)")
    an = tickets[tickets["refund_and_replacement"]]
    show(an[["ticket_id", "agent_id", "agent_name", "category", "refund_amount_inr", "refund_reason_code",
             "product_sku", "created_at"]], rename=False)

    st.subheader("Decisions where inputs conflicted")
    st.markdown(
        "- **Baselines:** each agent is compared with same-team Tier-1 peers excluding themselves (leave-one-out).\n"
        "- **Replacement cost:** the finance email says ~₹2,500; the policy (§5) says unit cost + ₹340. "
        "Policy followed, per-ticket via products.csv.\n"
        "- **Timestamps:** the data pack says timestamps are IST; the policy (§9) says legacy resolution times were "
        "reconstructed from UTC. Policy followed (+5h30m on resolved_at for `legacy_fd`).\n"
        "- **Festive volume +30% / replacements +100%:** client assertions, not used in any calculation.\n"
        "- **Volume:** the dataset averages far fewer than 650 tickets/week; the stated 650/week is used only to scale "
        "the modeled opportunity (the observed-volume figure is shown in the Page 1 caption).")

    st.subheader("NLP validation (manual)")
    st.markdown("Nothing here is auto-scored. Generate a fixed sample, judge each row Y/N in the "
                "`judged_relevant` column, save, and refresh this page.\n\n"
                "`python -m src.validation make-sample`  → writes `outputs/nlp_review_sample.csv` (40 masked tickets from the latest-12-month Coaching Priority agents, balanced across them).")
    rv = V.summarise_review()
    if rv is None or rv["judged"] == 0:
        st.info("No manual review recorded yet — no accuracy or error-rate figure is claimed.")
    else:
        x = st.columns(4)
        x[0].metric("Sample size", rv["sample_size"]); x[1].metric("Judged relevant", rv["relevant"])
        x[2].metric("Judged not relevant", rv["not_relevant"]); x[3].metric("Error rate", pct(rv["error_rate"]))
    st.markdown("**Themes found**")
    st.dataframe(topics[["topic", "label", "top_terms"]], hide_index=True, width="stretch")
