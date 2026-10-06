"""Load source CSVs and build one enriched ticket table. Joins use agent_id only."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from . import metrics as M

REQUIRED_COLUMNS = {
    "tickets": ["ticket_id", "created_at", "first_response_at", "resolved_at", "status", "channel",
                "customer_id", "order_id", "product_sku", "category", "priority", "assigned_team",
                "agent_id", "transfers", "csat_score", "refund_amount_inr", "refund_reason_code",
                "replacement_issued", "customer_message", "agent_notes", "source_system"],
    "agents": ["agent_id", "name", "site", "team", "shift", "tier", "from_date", "to_date"],
    "products": ["sku", "product_name", "family", "launch_date", "unit_cost_inr",
                 "retail_price_inr", "warranty_months"],
}
DEFAULT_DATA_DIR = Path(os.environ.get("VIREO_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))


def find_file(data_dir: Path, stem: str) -> Path | None:
    """Accept tickets.csv or the export-style '-tickets.csv'."""
    for name in (f"{stem}.csv", f"-{stem}.csv"):
        p = Path(data_dir) / name
        if p.exists():
            return p
    return None


def load_raw(data_dir: Path | str = DEFAULT_DATA_DIR) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    raw = {}
    for stem in ("tickets", "agents", "products"):
        p = find_file(data_dir, stem)
        if p is None:
            raise FileNotFoundError(f"{stem}.csv not found in {data_dir}. See README 'Expected data files'.")
        raw[stem] = pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""])
    # customers.csv is OPTIONAL and only used to mask customer names in text examples.
    p = find_file(data_dir, "customers")
    raw["customers"] = pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""]) if p else None
    return raw


def dedupe_agents(agents: pd.DataFrame) -> pd.DataFrame:
    """Roster may have several rows per agent_id; keep the most recent assignment."""
    a = agents.copy()
    a["from_date"] = pd.to_datetime(a["from_date"], errors="coerce")
    a = a.sort_values(["agent_id", "from_date"]).drop_duplicates("agent_id", keep="last")
    a["tier"] = pd.to_numeric(a["tier"], errors="coerce").astype("Int64")
    return a


def prepare(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    t = raw["tickets"].copy()
    for c in ("created_at", "first_response_at", "resolved_at"):
        t[c] = pd.to_datetime(t[c], errors="coerce")
    t["transfers"] = pd.to_numeric(t["transfers"], errors="coerce").fillna(0).astype(int)
    t["csat_raw"] = pd.to_numeric(t["csat_score"], errors="coerce")
    t["csat"] = M.clean_csat(t["csat_score"])

    # --- agents: join on agent_id ONLY (two agents share the name "Kavya Pandey")
    ag = dedupe_agents(raw["agents"])[["agent_id", "name", "site", "team", "shift", "tier"]]
    ag = ag.rename(columns={"name": "agent_name", "team": "agent_team"})
    t = t.merge(ag, on="agent_id", how="left")

    # --- products: unit cost for replacement planning
    pr = raw["products"][["sku", "unit_cost_inr"]].copy()
    pr["unit_cost_inr"] = pd.to_numeric(pr["unit_cost_inr"], errors="coerce")
    t = t.merge(pr, left_on="product_sku", right_on="sku", how="left").drop(columns="sku")

    # --- handle time (with legacy UTC->IST normalisation)
    t["raw_aht_min"] = M.minutes_between(t["first_response_at"], t["resolved_at"])
    t["resolved_at_ist"] = M.normalize_resolved_at(t["resolved_at"], t["source_system"])
    t["aht_min"] = M.aht_minutes(t["first_response_at"], t["resolved_at_ist"])

    # --- SLA, replacement, refund
    t["first_response_min"] = M.minutes_between(t["created_at"], t["first_response_at"])
    t["sla_breach"] = M.sla_breach(t["created_at"], t["first_response_at"], t["channel"])
    t["replacement_flag"] = t["replacement_issued"].str.upper().eq("Y")
    t["replacement_cost"] = M.replacement_cost(t["unit_cost_inr"], t["replacement_flag"]).values
    t["planned_replacement_cost"] = M.planned_replacement_cost(t["unit_cost_inr"]).values
    t["has_refund"] = pd.to_numeric(t["refund_amount_inr"], errors="coerce").fillna(0) > 0
    t["refund_and_replacement"] = t["has_refund"] & t["replacement_flag"]
    t["is_open"] = t["status"].isin(["open", "pending"])
    t["month"] = t["created_at"].dt.to_period("M").dt.to_timestamp()
    return t


def load(data_dir: Path | str = DEFAULT_DATA_DIR):
    raw = load_raw(data_dir)
    return raw, prepare(raw)
