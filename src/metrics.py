"""Pure metric functions. Definitions follow support-policy.pdf v3.2."""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---- Policy constants (support-policy.pdf) -------------------------------
SLA_MINUTES = {"chat": 15, "voice": 120, "social": 240, "email": 480}   # §3
SLA_CREDIT_INR = 350                                                    # §3
CONTACT_COST_INR = {"chat": 210, "email": 260, "voice": 520, "social": 240}  # §4
TRANSFER_COST_INR = 305                                                 # §4
AGENT_HOUR_INR = 165                                                    # §4
REPLACEMENT_LOGISTICS_INR = 340                                         # §5
LEGACY_SYSTEM = "legacy_fd"
LEGACY_UTC_TO_IST = pd.Timedelta(hours=5, minutes=30)                   # §9
POSITIVE_CSAT = (4, 5)                                                  # §8 / README


# ---- Row-level derivations -------------------------------------------------
def clean_csat(raw: pd.Series) -> pd.Series:
    """Valid CSAT = integer 1-5. Blank / invalid -> NaN (never zero)."""
    s = pd.to_numeric(raw, errors="coerce")
    return s.where(s.isin([1, 2, 3, 4, 5]))


def normalize_resolved_at(resolved_at: pd.Series, source_system: pd.Series) -> pd.Series:
    """Legacy resolved_at is UTC; add 5h30m to express it in IST."""
    out = pd.to_datetime(resolved_at).copy()
    mask = (source_system == LEGACY_SYSTEM) & out.notna()
    out[mask] = out[mask] + LEGACY_UTC_TO_IST
    return out


def minutes_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Minutes from start to end; NaN if either is missing (never 0)."""
    return (pd.to_datetime(end) - pd.to_datetime(start)).dt.total_seconds() / 60.0


def aht_minutes(first_response_at, resolved_at_ist) -> pd.Series:
    """Handle time = first_response_at -> resolved_at. Missing resolution -> NaN."""
    return minutes_between(first_response_at, resolved_at_ist)


def sla_breach(created_at, first_response_at, channel: pd.Series) -> pd.Series:
    """True when first human response is later than the channel target."""
    wait = minutes_between(created_at, first_response_at)
    target = channel.map(SLA_MINUTES)
    return (wait > target).where(wait.notna() & target.notna(), False).astype(bool)


def replacement_cost(unit_cost_inr, replacement_flag) -> pd.Series:
    """Planning cost = unit cost + Rs 340, only where a replacement was issued."""
    uc = pd.to_numeric(pd.Series(unit_cost_inr), errors="coerce")
    flag = pd.Series(replacement_flag).astype(bool).reset_index(drop=True)
    uc = uc.reset_index(drop=True)
    return (uc + REPLACEMENT_LOGISTICS_INR).where(flag)


def planned_replacement_cost(unit_cost_inr) -> pd.Series:
    """Planning cost of a replacement for any ticket (used for expected exposure)."""
    return pd.to_numeric(unit_cost_inr, errors="coerce") + REPLACEMENT_LOGISTICS_INR


# ---- Aggregations -----------------------------------------------------------
def csat_stats(csat: pd.Series) -> dict:
    """Blank CSAT excluded. CSAT % = share of rated tickets scoring 4 or 5."""
    rated = csat.dropna()
    n = len(rated)
    return {
        "rated": n,
        "csat_pct": float(rated.isin(POSITIVE_CSAT).mean() * 100) if n else np.nan,
        "avg_rating": float(rated.mean()) if n else np.nan,
    }


def summarise(g: pd.DataFrame) -> pd.Series:
    """Metric bundle for any slice of enriched tickets."""
    cs = csat_stats(g["csat"])
    n = len(g)
    aht = g["aht_min"].dropna()
    return pd.Series({
        "tickets": n,
        "rated": cs["rated"],
        "csat_pct": cs["csat_pct"],
        "avg_rating": cs["avg_rating"],
        "avg_aht": aht.mean() if len(aht) else np.nan,
        "median_aht": aht.median() if len(aht) else np.nan,
        "sla_breach_pct": g["sla_breach"].mean() * 100 if n else np.nan,
        "replacements": int(g["replacement_flag"].sum()),
        "replacement_rate": g["replacement_flag"].mean() * 100 if n else np.nan,
        "replacement_spend": float(g["replacement_cost"].sum()),
        "transfers": int(g["transfers"].sum()),
        "transfer_rate": (g["transfers"] > 0).mean() * 100 if n else np.nan,
    })
