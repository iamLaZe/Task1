# Submission Form — Vireo Support Coach

**1. What did you build, and what business outcome does it move? (number and money)**
A local Streamlit app for Vireo's support team. It delivers what the client asked for (CSAT and handle time per agent, raw Bottom 10), plus a transparent Coaching Priority shortlist, the evidence behind each flag, and recurring ticket themes from local NLP.

- Latest 12 months, Tier 1: CSAT 47.8%.
- Four agents meet the rules: CSAT 34.9% vs 50–52% for same-team peers (excluding themselves), and 30.4% of their tickets end in a replacement vs about 16%. They handle 11.8% of Tier-1 tickets but carry 27.6% of Tier-1 replacement spend.
- Their replacements cost ₹5,50,280 against ₹4,17,572 expected at peer category rates, an associated excess exposure of ₹1,32,708.
- **Modeled direct-cost opportunity: about ₹58,387 per quarter** (50% recovery, scaled to 650 tickets/week × 13 weeks). Scenarios: ₹29,193 at 25% and ₹87,580 at 75%.

This is modeled, not guaranteed, and the spend is exposure associated with these agents' tickets, not proven to be caused by them. CSAT and handle time are not converted into rupees.

**2. What does one run cost, and what would a month cost at ~650 tickets/week?**
Paid model/API calls per run: 0 (TF-IDF + NMF run locally, no API key).
650 tickets/week × 52 ÷ 12 ≈ 2,817 tickets/month; 2,817 × ₹0 = **₹0/month** paid AI/API cost.
Ordinary CPU/RAM is still used; only paid calls are zero.

**3. How do you know it works?**
- **Software:** 21 automated tests pass: CSAT excludes blanks, scores 4–5 count as positive, legacy +5h30m, AHT, missing resolution, SLA, replacement cost, Tier 2 exclusion, 50-rating minimum, leave-one-out baselines, category/fallback baselines, case-mix excess, quarterly scaling, name masking before NLP, NLP peer baseline excluding the agent, recovery scenarios, CSV export selection, review-sample balance.
- **Data:** 11,750 tickets; 44 agents (38 Tier 1, 6 Tier 2); raw negative AHT 2,309 → 0 after normalisation; 567 open/pending left blank (not zero); 6 refund+replacement anomalies; all agent IDs and SKUs resolve; CSAT blanks excluded. Full-history Tier-1 CSAT was cross-checked with an independent calculation (48.63%).
- **NLP:** sample size 0 so far. **Manual NLP review not yet completed; no accuracy or error rate is claimed.** A fixed 40-ticket masked sample (10 per Coaching Priority agent) is generated with `python -m src.validation make-sample`, and the app reports the real error rate once it is judged.
- **Cases it will get wrong:** tickets with several issues, very short or vague text, failed IVR transcripts, and generic agent notes.

**4. Did you change, narrow, or push back on the client's ask?**
Yes, in the first build, once the data and policy were checked.
- The raw Bottom 10 is delivered exactly as requested and stays visible.
- The training shortlist is narrowed to agents whose CSAT is ≥5 pp below, and replacement rate ≥3 pp above, their own team peers (excluding themselves). This follows Neha Kulkarni's warning that specialised queues get harder customers.
- Replacement cost uses the policy (product unit cost + ₹340), not the ₹2,500 in Arjun's email.
- Legacy resolution times are treated as UTC per the policy, although the data pack says IST.
- Tier 2 is shown separately, never ranked against Tier 1.
- No rupee value is put on CSAT, and handle time is not turned into savings.

**5. What is wrong with what you are handing us?**
- The UI was smoke-tested headlessly with stubbed Streamlit/Plotly, never viewed live; layout and charts need one real look.
- The thresholds (5 pp, 3 pp, 20 peer tickets, 50% recovery) are judgment calls, not statistically derived.
- Peers can include other candidates, and small teams make baselines noisy.
- The data averages about 184 tickets/week, so scaling to 650 is a ~3.5× extrapolation (about ₹16.5k per quarter at observed volume). The 650 figure needs client confirmation.
- Case-mix adjustment is by ticket category only; replacement is not proof of cause.
- The NLP themes have not been checked by a human, and some themes are imperfect.

**6. What did you deliberately leave out, and why that rather than something else?**
Orders/lot codes, repeat-contact matching, CSAT→revenue, AHT→₹ savings, prediction, bonus allocation, and any database, API or authentication. The data has no defensible conversion for CSAT or AHT, and replacement spend is the only lever with a policy-defined cost. The rest would add effort without changing who gets reviewed first.

**7. Anything you built or found that nobody asked for?**
- A flat team rate would nearly double the excess (₹2.55 lakh vs ₹1.33 lakh), so the overview shows both and uses the case-mix figure.
- Recovery scenarios at 25/50/75%, CSV downloads, and a per-agent category case-mix table.
- The candidates issue many warranty-category replacements, although policy §6 reserves warranty replacements for Tier 2. This is flagged for the Team Lead to check, not asserted.
- The volume mismatch (650/week stated vs ~184/week in the data).
- 2,309 negative handle times, all legacy tickets, fixed by timezone normalisation rather than dropped.
- PII-masked ticket examples.

**8. What did you use AI for?**
- **ChatGPT:** planning, research, QA and business reasoning.
- **Claude Chat:** implementation, tests and documentation. It helped build quickly, write test scaffolding and catch the flat-rate flaw.
- **Antigravity:** planned for final visual polish only. [FILL IN whether used]
- **Wasted time / corrected:** the first baselines included the agent being judged; one flat replacement rate ignored ticket mix; early wording over-generalised about Logistics/Returns; customer names were first masked only for display.
- **Thrown away:** an opaque score, revenue-from-CSAT, AHT savings, repeat-contact matching.
- No paid runtime LLM calls. The prompt log is in `deliverables/07_Prompt_Log.md`.


**9. Your Public Google Drive Link**
https://drive.google.com/drive/folders/1VDT3zEJatMKHS7weseC_XhEY9JVA5nyJ?usp=sharing

**10. Someone picks this up on Monday and you are unreachable — the three things**
1. **Run it:** `pip install -r requirements.txt`, put tickets/agents/products CSVs in `data/`, then `streamlit run app.py`; check with `python tests/test_metrics.py` (expect 21 passed).
2. **How the list works:** Tier 1, ≥50 rated tickets, CSAT ≥5 pp below and replacement rate ≥3 pp above leave-one-out team peers = Coaching Priority. Opportunity = 50% × case-mix excess, scaled to 650/week × 13 weeks. Constants live in `src/scoring.py`.
3. **Don't forget:** it is modeled, not guaranteed; policy beats email (unit cost + ₹340, legacy UTC→IST); join on agent_id (two Kavya Pandeys); confirm 650/week; NLP explains but never selects; never commit raw data; the NLP manual review is still outstanding.

**11. Honest hours spent**
5.5 hours

**12. Github Repo Link**
https://github.com/iamLaZe/Task1
