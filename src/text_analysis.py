"""Lightweight local NLP: TF-IDF + NMF topics. Runs offline, 0 paid API calls.

NLP only EXPLAINS recurring themes. It never decides who is coached - business metrics do.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

N_TOPICS = 10
MIN_TOKENS = 3  # shorter texts (junk IVR transcripts, "help", "-") are left unclassified
EXTRA_STOP = {
    "hi", "helo", "hello", "dear", "sir", "ji", "team", "pls", "please", "kindly", "revert", "regards",
    "rgds", "thanks", "thank", "vireo", "ivr", "transcript", "anyone", "reply", "asap", "help", "need",
    "want", "wanted", "bought", "got", "customer", "cx", "re", "contact", "issue", "chk", "ticket",
    "closing", "closed", "resolved", "informed", "cust", "support", "awaiting", "response", "writing",
    "reference", "order", "product", "purchased", "ago", "days", "day", "week", "weeks", "month", "last",
    "tried", "already", "nothing", "changed", "different", "checked", "advise", "request", "earliest",
    "matter", "look", "looking", "bhai", "pareshan", "hai", "hu", "bahut", "mera", "kya", "ko", "ka",
    "expected", "expecting", "recieved", "received", "around", "pulse", "strata", "nexa", "orbit",
    "airlite", "arc", "earbuds", "buds", "mini", "watch", "speaker", "headphones", "unit", "new",
    "flipkart", "amazon", "vireo.in", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
    "oct", "nov", "dec", "xfer", "frontline", "sai", "kavya", "approved", "raised",
}
STOP = sorted(set(ENGLISH_STOP_WORDS) | EXTRA_STOP)

_RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_RE_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
_RE_ORDER = re.compile(r"\bVR\d{4,}\b", re.I)
_RE_CUST = re.compile(r"\bC\d{6}\b")
_RE_SIGNOFF = re.compile(
    r"(?i)\b(regards|rgds|thanks|thank you|thx|cheers|sincerely|revert)\b[,:\s]*\n?\s*"
    r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)")
_RE_AGENT_SIG = re.compile(r"(~|//)\s?[A-Za-z]+")


def name_tokens(customers: pd.DataFrame | None) -> set[str]:
    """Customer name parts, used only in memory to mask text. Never displayed or stored."""
    if customers is None or "name" not in customers:
        return set()
    toks = set()
    for n in customers["name"].dropna().unique():
        toks.update(p for p in str(n).split() if len(p) > 2)
    return toks


def mask_pii(text, names: set[str] | None = None) -> str:
    """Mask emails, phones, order/customer ids, sign-off names and known customer names."""
    if not isinstance(text, str):
        return ""
    s = _RE_EMAIL.sub("[email]", text)
    s = _RE_PHONE.sub("[phone]", s)
    s = _RE_ORDER.sub("[order]", s)
    s = _RE_CUST.sub("[customer]", s)
    s = _RE_SIGNOFF.sub(lambda m: f"{m.group(1)} [name]", s)
    if names:
        pat = r"\b(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")\b"
        s = re.sub(pat, "[name]", s, flags=re.I)
    return s


def clean_for_model(text, names: set[str] | None = None) -> str:
    """Mask PII (incl. customer names) FIRST, then normalise. The model never sees raw names."""
    s = mask_pii(text, names) if isinstance(text, str) else ""
    s = s.lower()
    s = re.sub(r"\[[a-z ]+\]", " ", s)          # masks / [ivr transcript]
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = _RE_AGENT_SIG.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_docs(tickets: pd.DataFrame, names: set[str] | None = None) -> pd.Series:
    """Masked, normalised model input. Names/emails/phones/ids are removed before vectorisation."""
    raw = (tickets["customer_message"].fillna("") + " . " + tickets["agent_notes"].fillna(""))
    return raw.map(lambda x: clean_for_model(x, names))


def fit_topics(tickets: pd.DataFrame, names: set[str] | None = None, n_topics: int = N_TOPICS,
               random_state: int = 42):
    """Return (tickets_with_topics, topic_table). Topic labels come from the data's top terms."""
    docs = build_docs(tickets, names)
    ok = docs.str.split().str.len().fillna(0) >= MIN_TOKENS
    vec = TfidfVectorizer(stop_words=STOP, ngram_range=(1, 2), min_df=15, max_df=0.4,
                          token_pattern=r"(?u)\b[a-z]{3,}\b", sublinear_tf=True)
    X = vec.fit_transform(docs[ok])
    nmf = NMF(n_components=n_topics, init="nndsvda", random_state=random_state, max_iter=400)
    W = nmf.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())

    rows = []
    for k, comp in enumerate(nmf.components_):
        top = terms[np.argsort(comp)[::-1][:6]]
        rows.append({"topic": k, "label": ", ".join(top[:3]), "top_terms": ", ".join(top)})
    topics = pd.DataFrame(rows)

    out = tickets[["ticket_id"]].copy()
    out["topic"] = -1
    out["topic_strength"] = 0.0
    idx = np.where(ok.values)[0]
    has_signal = W.sum(axis=1) > 0
    out.iloc[idx[has_signal], out.columns.get_loc("topic")] = W[has_signal].argmax(axis=1)
    out.iloc[idx[has_signal], out.columns.get_loc("topic_strength")] = W[has_signal].max(axis=1)
    lab = dict(zip(topics["topic"], topics["label"]))
    out["topic_label"] = out["topic"].map(lab).fillna("(unclassified / too short)")
    return out, topics


def agent_themes(tickets_topics: pd.DataFrame, window: pd.DataFrame, agent_id: str, top_n: int = 5) -> pd.DataFrame:
    """Share of an agent's tickets per theme vs. leave-one-out same-team Tier-1 peers (descriptive, not scoring)."""
    j = window.merge(tickets_topics[["ticket_id", "topic_label", "topic_strength"]], on="ticket_id", how="left")
    me = j[j["agent_id"] == agent_id]
    if me.empty:
        return pd.DataFrame()
    team = me["agent_team"].iloc[0]
    tm = j[(j["tier"] == 1) & (j["agent_team"] == team) & (j["agent_id"] != agent_id)]
    a = me.groupby("topic_label").agg(tickets=("ticket_id", "size"), replacements=("replacement_flag", "sum"))
    a["agent_share_pct"] = a["tickets"] / a["tickets"].sum() * 100
    b = tm.groupby("topic_label").size()
    a["peer_share_pct"] = (b / b.sum() * 100).reindex(a.index).fillna(0)
    a["replacement_rate_pct"] = a["replacements"] / a["tickets"] * 100
    return a.sort_values("tickets", ascending=False).head(top_n).reset_index()


def examples(tickets_topics: pd.DataFrame, window: pd.DataFrame, agent_id: str, names: set[str] | None = None,
             n: int = 4, max_chars: int = 220) -> pd.DataFrame:
    """Representative tickets (highest topic strength), PII-masked."""
    j = window.merge(tickets_topics[["ticket_id", "topic_label", "topic_strength"]], on="ticket_id", how="left")
    me = j[(j["agent_id"] == agent_id) & (j["topic_label"] != "(unclassified / too short)")]
    me = me.sort_values("topic_strength", ascending=False).drop_duplicates("topic_label").head(n)
    return pd.DataFrame({
        "date": me["created_at"].dt.strftime("%Y-%m-%d"),
        "channel": me["channel"],
        "theme": me["topic_label"],
        "replacement": np.where(me["replacement_flag"], "Y", "N"),
        "customer_message (masked)": me["customer_message"].map(lambda s: mask_pii(s, names).replace("\n", " ")[:max_chars]),
        "agent_note (masked)": me["agent_notes"].map(lambda s: mask_pii(s, names).replace("\n", " ")[:max_chars]),
    })
