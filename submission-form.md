# Submission Form — Vireo Support Coach (root copy)

> Placeholders to fill before submitting: [PASTE PUBLIC GOOGLE DRIVE LINK], [PASTE PUBLIC GITHUB LINK], [PASTE SCREEN RECORDING LINK], [FILL IN ACTUAL HOURS]


**1. What did you build, and what business outcome does it move? (number and money)**
A local Streamlit app that gives the requested CSAT, handle time and raw Bottom 10 per agent, plus a rule-based Coaching Priority shortlist, evidence per agent, and recurring ticket themes. Latest 12 months, Tier 1: CSAT 47.8%; four agents (11.8% of Tier-1 tickets, 27.6% of Tier-1 replacement spend) have CSAT 34.9% vs 50–52% for same-team peers. Their replacements cost ₹5,50,280 vs ₹4,17,572 expected at peer category rates → ₹1,32,708 associated excess exposure. **Modeled direct-cost opportunity: ≈ ₹58,387 per quarter** (50% recovery, scaled to 650 tickets/week × 13 weeks). Modeled, not guaranteed. CSAT is not monetised.

**2. What does one run cost, and a month at ~650 tickets/week?**
Paid model/API calls per run: 0 (local TF-IDF + NMF). 650 × 52 / 12 ≈ 2,817 tickets/month × ₹0 per ticket = **₹0/month** paid AI/API cost. (Ordinary CPU/RAM is still used; only paid calls are zero.)

**3. How do you know it works?**
- 21 automated tests pass (CSAT blanks, 4/5 positives, legacy +5h30m, AHT, missing resolution, SLA, replacement cost, Tier 2 exclusion, 50-rating minimum, leave-one-out baselines, category/fallback baselines, case-mix excess, quarterly scaling, name masking before NLP, NLP peer baseline excluding the agent, recovery scenarios, CSV export selection, review-sample balance).
- Data sanity: 11,750 tickets, 44 agents (38/6), raw negative AHT 2,309 → 0 after normalisation, 567 unresolved, 6 refund+replacement anomalies, all agent IDs/SKUs resolve.
- NLP: **manual review not yet completed; no NLP accuracy or error rate is claimed.** A 40-ticket masked review sheet (drawn from the four Coaching Priority agents, balanced across them) can be generated (`python -m src.validation make-sample`). Expected failure kinds: multi-issue tickets, very short/vague text, failed IVR transcripts, generic agent notes.

**4. Did you change, narrow or push back on the client's ask?**
Yes. The raw Bottom 10 is delivered as asked, but the training shortlist is narrowed using leave-one-out same-team baselines and case-mix-adjusted replacement exposure (prompted by Neha Kulkarni's warning on specialised queues). Tier 2 is kept separate (policy §6). The ₹2,500 replacement cost in the finance email was replaced by policy: unit cost + ₹340. Replacement spend is framed as associated exposure, and CSAT is not turned into ₹.

**5. What is wrong with what you are handing us?**
- Streamlit UI was only smoke-tested headlessly (stubbed libraries); layout/charts need one live look.
- Thresholds (5 pp, 3 pp, 20 peer tickets, 50% recovery) are judgment calls, not statistically derived.
- Candidates' own peers can include other candidates; small teams make baselines noisy.
- The data averages ~184 tickets/week, so scaling to 650 is a 3.5× extrapolation; the ₹58k figure is ≈₹16.5k at observed volume.
- Replacement ≠ proven cause; ticket-level case mix is limited to category (product mix only enters via unit cost).
- NLP themes unvalidated by humans.

**6. What did you deliberately leave out, and why?**
Orders/lot-code, repeat-contact matching, CSAT→revenue, AHT→₹ savings, prediction, bonus allocation, database/API/auth. The data gives no defensible conversion for CSAT or AHT, and replacement spend is the only directly costed lever; the rest would add effort without changing the decision.

**7. Anything built or found that nobody asked for?**
Case-mix comparison on the overview page showing a flat team rate would double the excess (₹2.55 lakh vs ₹1.33 lakh); 25/50/75% recovery scenario table; CSV downloads; category-level case-mix table per agent; a flag that the candidates issue many warranty-category replacements though policy §6 reserves warranty replacement approval for Tier 2 (to be checked, not asserted); the 650 vs ~184 tickets/week mismatch; PII-masked ticket examples.

**8. What did you use AI for?**
ChatGPT: planning, analysis, QA, business reasoning. Claude Chat: implementation, tests, documentation. Antigravity: planned for final visual polish only — [FILL IN whether used]. Helped: fast build, test scaffolding, finding the flat-rate flaw. Wasted time: first-pass baselines included the agent; initial wording over-generalised "Logistics/Returns" — both corrected. Discarded: opaque scoring, revenue-from-CSAT, AHT savings. No paid runtime LLM calls. Screen recording: [PASTE SCREEN RECORDING LINK]

**9. Your Public Google Drive Link**
[PASTE PUBLIC GOOGLE DRIVE LINK]

**10. Someone picks this up Monday and you are unreachable — the three things**
1. Run: `pip install -r requirements.txt`, put tickets/agents/products CSVs in `data/`, `streamlit run app.py`; `python tests/test_metrics.py`.
2. Coaching list = Tier 1, ≥50 rated, CSAT ≥5 pp below and replacement rate ≥3 pp above leave-one-out team peers; opportunity = 50% × case-mix excess, scaled to 650/wk × 13.
3. Don't forget: modeled not guaranteed; policy (₹ unit+340, UTC→IST for legacy) beats the email; confirm 650/week; NLP never selects people; never commit raw data.

**11. Honest hours spent**
[FILL IN ACTUAL HOURS]

**12. Github Repo Link**
[PASTE PUBLIC GITHUB LINK]
