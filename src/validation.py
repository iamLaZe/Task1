"""Data-quality report and the manual NLP review utility."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .data import REQUIRED_COLUMNS

REVIEW_FILE = Path(__file__).resolve().parent.parent / "outputs" / "nlp_review_sample.csv"


def data_quality(raw: dict[str, pd.DataFrame], t: pd.DataFrame, ranked_agent_tiers=None) -> dict:
    """Counts for the Data Quality page. Nothing is dropped; every issue is surfaced."""
    n = len(t)
    agents = raw["agents"]
    known_agents = set(agents["agent_id"])
    skus = set(raw["products"]["sku"])
    rated = int(t["csat"].notna().sum())
    invalid_csat = int((t["csat_raw"].notna() & t["csat"].isna()).sum())
    sku_present = t["product_sku"].notna()
    names = agents["name"].value_counts()
    dup_names = sorted(names[names > 1].index)
    legacy = t["source_system"] == M.LEGACY_SYSTEM
    missing_cols = {k: [c for c in cols if c not in raw[k].columns] for k, cols in REQUIRED_COLUMNS.items()}
    repl_missing_cost = int((t["replacement_flag"] & t["replacement_cost"].isna()).sum())
    q = {
        "tickets": n,
        "agents": int(agents["agent_id"].nunique()),
        "tier1_agents": int((pd.to_numeric(agents["tier"]) == 1).sum()),
        "tier2_agents": int((pd.to_numeric(agents["tier"]) == 2).sum()),
        "missing_csat": int(t["csat"].isna().sum()) - invalid_csat,
        "rated": rated,
        "csat_response_rate": rated / n * 100,
        "invalid_csat": invalid_csat,
        "unknown_agent_ids": int((~t["agent_id"].isin(known_agents)).sum()),
        "unknown_product_skus": int((sku_present & ~t["product_sku"].isin(skus)).sum()),
        "missing_product_sku": int((~sku_present).sum()),
        "legacy_tickets": int(legacy.sum()),
        "raw_negative_aht": int((t["raw_aht_min"] < 0).sum()),
        "normalized_negative_aht": int((t["aht_min"] < 0).sum()),
        "missing_resolved_at": int(t["resolved_at"].isna().sum()),
        "open_pending": int(t["is_open"].sum()),
        "refund_replacement_anomalies": int(t["refund_and_replacement"].sum()),
        "replacements_without_cost": repl_missing_cost,
        "duplicate_ticket_ids": int(t["ticket_id"].duplicated().sum()),
        "duplicate_agent_names": dup_names,
        "missing_columns": {k: v for k, v in missing_cols.items() if v},
        "tier2_in_tier1_ranking": 0 if ranked_agent_tiers is None else int((pd.Series(ranked_agent_tiers) != 1).sum()),
    }
    q["checks"] = [
        ("Required columns present", not q["missing_columns"]),
        ("CSAT values are 1-5 or blank", q["invalid_csat"] == 0),
        ("All agent_ids resolve to agents.csv", q["unknown_agent_ids"] == 0),
        ("All product SKUs resolve", q["unknown_product_skus"] == 0),
        ("Replacement cost calculable for every replacement", q["replacements_without_cost"] == 0),
        ("No negative AHT after legacy normalisation", q["normalized_negative_aht"] == 0),
        ("No Tier 2 agent in the Tier 1 ranking", q["tier2_in_tier1_ranking"] == 0),
        ("ticket_id is unique", q["duplicate_ticket_ids"] == 0),
    ]
    return q


# ---- Manual NLP review ---------------------------------------------------------
def make_review_sample(window: pd.DataFrame, tickets_topics: pd.DataFrame, candidate_ids, names=None, n: int = 40,
                       seed: int = 42, path: Path = REVIEW_FILE) -> Path:
    """Fixed-seed review sample drawn from latest-12-month Coaching Priority agents' tickets, as evenly
    balanced across the candidates as possible. Text is PII-masked. Nothing is auto-judged: a human fills
    the 'judged_relevant' column with Y or N."""
    from .text_analysis import mask_pii
    ids = sorted(candidate_ids)
    j = window[window["agent_id"].isin(ids)].merge(tickets_topics[["ticket_id", "topic_label"]], on="ticket_id")
    parts, left = [], n
    for k, aid in enumerate(ids):
        g = j[j["agent_id"] == aid]
        quota = min(len(g), left // (len(ids) - k))
        parts.append(g.sample(n=quota, random_state=seed))
        left -= quota
    s = pd.concat(parts).sort_values(["agent_id", "ticket_id"]) if parts else j.head(0)
    out = pd.DataFrame({
        "ticket_id": s["ticket_id"], "agent_id": s["agent_id"],
        "assigned_theme": s["topic_label"],
        "customer_message (masked)": s["customer_message"].map(lambda x: mask_pii(x, names).replace("\n", " ")),
        "agent_notes (masked)": s["agent_notes"].map(lambda x: mask_pii(x, names).replace("\n", " ")),
        "judged_relevant": "",
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def summarise_review(path: Path = REVIEW_FILE) -> dict | None:
    """Return real counts from a filled-in sheet, or None if nothing has been judged yet."""
    if not Path(path).exists():
        return None
    r = pd.read_csv(path, dtype=str, keep_default_na=False)
    j = r["judged_relevant"].str.strip().str.upper()
    yes, no = int((j == "Y").sum()), int((j == "N").sum())
    judged = yes + no
    if judged == 0:
        return {"sample_size": len(r), "judged": 0, "relevant": 0, "not_relevant": 0, "error_rate": None}
    return {"sample_size": len(r), "judged": judged, "relevant": yes, "not_relevant": no,
            "error_rate": no / judged * 100}


if __name__ == "__main__":
    from . import data, text_analysis as T
    if len(sys.argv) > 1 and sys.argv[1] == "make-sample":
        raw, t = data.load()
        names = T.name_tokens(raw["customers"])
        tt, _ = T.fit_topics(t, names)
        from . import scoring as S
        w = S.apply_window(t, "latest12")
        cands = S.agent_table(w).query("coaching_priority")["agent_id"].tolist()
        print("Wrote", make_review_sample(w, tt, cands, names))
    else:
        print(summarise_review())
