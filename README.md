# Vireo Support Coach

A small, local Streamlit app that helps Vireo Audio decide **which support agents should get coaching** — using the
client's requested views (CSAT, handle time, bottom ten) *plus* team context, so training money isn't spent on people
who are simply working the hardest queues.

## 1. Project purpose
Turn the Vireo support export (tickets, roster, products) into: the raw Bottom 10, a transparent coaching-priority
shortlist, the evidence behind each flag, a modeled direct-cost opportunity and recurring ticket themes.

## 2. Business problem
Priya Raman (Head of CX) asked for *CSAT per agent, handle time per agent, bottom ten flagged* to spend a ₹4 lakh Q3
training budget "on the right people". A raw bottom ten is delivered as asked, but Neha Kulkarni (Support Ops) warns
that hardware-triage/warranty-type queues get the angriest customers by design. So the shortlist is narrowed using
each agent's **own team baseline**.

## 3–5. Quick setup, installation, run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put the data files in ./data (see §6), then:
streamlit run app.py
```
Tests (no pytest needed, 21 tests): `python tests/test_metrics.py` — or `pytest` if installed.
Optional NLP review sheet: `python -m src.validation make-sample`.

## 6. Expected data files (in `data/`, or set `VIREO_DATA_DIR`)
`tickets.csv`, `agents.csv`, `products.csv` (required) and `customers.csv` (optional; only used to *mask* customer names
in text examples). Export-style names with a leading dash (`-tickets.csv`) are also accepted. `orders.csv` is not used.
`data/*.csv` is git-ignored because the data contains customer PII — **do not commit it**.

## 7. Project structure
```
app.py                 Streamlit UI (4 pages)
src/data.py            load CSVs, join on agent_id, derive row-level fields
src/metrics.py         policy constants + pure metric functions
src/scoring.py         windows, ranking, Bottom 10 / Top 5, coaching rules, business impact
src/text_analysis.py   PII masking, TF-IDF + NMF themes, agent theme tables
src/validation.py      data-quality report, manual NLP review utility
tests/test_metrics.py  automated tests

```

## 8. Metric definitions (policy v3.2 is authoritative)
- **Joins:** always `agent_id` (two agents are both named "Kavya Pandey": A3006 Chat Frontline, A3029 Logistics).
- **SLA breach:** first human response later than target, measured from creation: chat 15 min, voice 120, social 240, email 480.
  Each breach = ₹350 credit (shown as a *secondary* metric only).
- **Replacement rate:** share of the agent's tickets with `replacement_issued = Y`.
- **Replacement planning cost:** `unit_cost_inr` (products.csv) + ₹340. The ₹2,500 in the finance email is **not** used.
- **Replacement spend:** sum of that cost over the agent's replacement tickets. Described as *associated replacement-cost
  exposure*, never "agent-caused".

## 9. CSAT definition
Score 1–5; blank = no response and is **excluded** (never 0). **CSAT % = share of rated tickets scoring 4 or 5.**
Values outside 1–5 would be treated as invalid (currently 0) and counted on the Data Quality page.

## 10. AHT definition
Handle time = `first_response_at` → `resolved_at`. Missing `resolved_at` (open/pending, 567 tickets) → no AHT (NaN, never 0).
AHT is very skewed (async teams take ~a day), so **median** is the primary indicator; the average is shown too. AHT is an
operating metric — it is **not** converted to ₹ with the ₹165/hour rate because elapsed time ≠ active labour time.

## 11. Legacy timestamp normalization
For `source_system = legacy_fd`, `resolved_at` was reconstructed from UTC, so **+5h30m** is added before computing AHT.
Nothing is dropped: raw negative AHT (2,309, all legacy) vs normalized negative AHT (0) is shown on the Data Quality page.

## 12. Tier 1 / Tier 2
Tier 2 (Escalations & Warranty, 6 agents) is never ranked against Tier 1 and gets no baseline or coaching label; it is
shown in its own table (policy §6). All ranking, Bottom 10, Top 5 and coaching views are Tier 1.

## 13. Bottom 10 methodology
Tier 1, ≥ 50 rated tickets, lowest CSAT % first (tie-break: average rating). Always visible. Agents under 50 rated tickets
are listed separately as **Insufficient CSAT sample**. Top 5 uses the same eligibility rule (reference only; no bonus logic).

## 14. Coaching Priority methodology
Transparent rules, no scoring model. In the selected window an agent is **Coaching priority** if all hold:
1. Tier 1; 2. ≥ 50 rated tickets; 3. CSAT ≥ 5 pp below the **leave-one-out peer** CSAT; 4. replacement rate ≥ 3 pp above the
**leave-one-out peer** replacement rate. Bottom-10 agents failing these are labelled **Bottom 10 — review context**.
**Leave-one-out peer baseline** = pooled Tier-1 metrics of the agent's `team` (agents.csv) *excluding the agent being evaluated*; used for
CSAT, replacement rate, average/median AHT and SLA breach %. (Pooled team CSAT/replacement are kept only as display columns.)
`assigned_team` (first-routed team) can differ from the resolver's team; the share is shown as context.

## 15. Business-impact methodology (case-mix aware)
Latest 12 months, coaching-priority cohort. For each candidate:
1. Peers = other Tier-1 agents on the same team (candidate excluded).
2. Per ticket category, peer replacement rate; used if the category has **≥ 20 peer tickets**, else the leave-one-out **team-wide fallback** rate.
3. Expected cost per candidate ticket = that rate × (product `unit_cost_inr` + ₹340). Actual = sum of (unit cost + ₹340) over replacement tickets.
4. Excess = max(0, actual − expected). **Modeled direct-cost opportunity = 50% × excess**, i.e. *associated replacement-cost exposure* recovered — modeled, never guaranteed, never "agent-caused".
No ₹ value is assigned to CSAT; AHT and SLA credits (secondary) are not in the headline. The Executive Overview shows flat-team-rate vs case-mix-adjusted excess side by side (the flat rate overstates the excess because candidates' ticket mix differs; the headline uses the case-mix figure), plus a **recovery scenario table (25% / 50% / 75%)** — scenarios only, not forecasts or guaranteed savings; 50% stays the headline.

## 16. Current-volume scaling
Per-ticket opportunity = modeled recovery ÷ all tickets in the 12-month window (all tiers, preserving the observed Tier-1 share) × 650 tickets/week × 13 weeks.
Current result: **≈ ₹58,387 per quarter**. **Caveat:** the file averages only ~184 tickets/week, so 650 is a ~3.5× scale-up (≈ ₹16.5k/quarter at observed volume). Confirm 650/week with the client.

## 17. NLP approach
Local, offline: text = `customer_message` + `agent_notes` → **PII masked first** (customer names from `customers.csv`, emails, phones, order and customer IDs) → lower-cased, stop-words/digits removed → TF-IDF (1–2-grams) →
NMF with 10 topics (fixed seed). Topic labels are the top terms found in the data (nothing hard-coded). Very short or junk text
(e.g. failed IVR transcripts) is left unclassified. Themes are shown per agent vs. **leave-one-out same-team Tier-1 peers** (the selected agent is excluded from the peer distribution). **NLP never selects who is coached.**

## 18. Validation approach
Data: required columns, CSAT 1–5/blank, agent IDs and SKUs resolve, replacement cost computable, no negative normalized AHT, no
Tier 2 in the Tier 1 ranking, unique ticket IDs. NLP: no accuracy % is claimed. Run `python -m src.validation make-sample`, judge
the 40 masked tickets (drawn from the latest-12-month Coaching Priority agents, balanced across them) Y/N in `outputs/nlp_review_sample.csv`; the app then shows the genuine sample size, relevant / not relevant
counts and error rate. **No manual review has been recorded yet.**

## 19. Runtime cost
**0 paid model/API calls.** 650 × 52 / 12 ≈ 2,817 tickets/month × ₹0 = **₹0/month** in paid AI/API cost. NLP runs locally
(it still uses ordinary CPU/RAM; only paid API calls are zero).

## 20. Limitations
- Replacement ≠ proven cause; case mix (product, category) differs by agent. The detail page shows category-level replacement rates.
- Leave-one-out peers on small teams are noisy, and peers can include other candidates.
- Case mix is adjusted by ticket category only; the 20-peer-ticket threshold and fallback are judgment calls.
- 50% recovery is an assumption, not an observed result. Volume scaling relies on the stated 650/week.
- AHT is elapsed time and differs structurally by team; compare within team only.
- CSAT response rate is ~44%; small samples are noisy (hence the 50-rating minimum).
- The 5 pp / 3 pp / 50% thresholds are configurable constants in `src/scoring.py`, not statistically derived.
- The 4 coaching-priority agents issue many replacements in warranty/charging categories; policy §6 says only Tier 2 may approve
  *warranty* replacements — worth checking with the Team Lead (not asserted by the app).
- Streamlit UI was smoke-tested headlessly (all pages execute); visual layout should be eyeballed once on a real run.

## 21. Intentionally omitted
Orders/lot-code analysis, customer-level analysis, repeat-contact (30-day) matching, CSAT→revenue conversion, AHT→₹ savings,
predictive models, bonus allocation, database/API/auth/Docker/LLM calls.

## 22. How to interpret the dashboard
- **Page 1** — current (latest 12 months, Tier 1) view: decision summary, KPIs, 18-month CSAT trend, CSAT vs AHT, replacement rate vs team, both lists, case-mix comparison, recovery scenarios, recommendation, and **CSV downloads** (full Tier-1 metrics, raw Bottom 10, Coaching Priority shortlist).
- **Page 2** — all Tier-1 agents with team filter and window selector; Bottom 10, Top 5, insufficient-sample and Tier 2 tables.
- **Page 3** — one agent: metrics, own-team baseline, ✅/❌ rule evidence, category case-mix table (agent vs peer tickets/rates, baseline used) and theme tables, masked examples.
- **Page 4** — data quality counts, before/after legacy fix, anomalies, conflicts resolved in favour of the policy, NLP review status.
- Read *Coaching priority* as "review for training", and *Bottom 10 — review context* as "low CSAT, check the queue first".


