from __future__ import annotations

from pathlib import Path
import os
import math
import re
import traceback
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

# Optional OpenAI dependency (only used for sector classification fallback)
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


# ============================================================
# Load thresholds table (robust column normalization)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
THRESHOLDS_XLSX = os.getenv("THRESHOLDS_XLSX", "Benchmark Thresholds-2X Global Benchmarks (1).xlsx")
THRESHOLDS_PATH = PROJECT_ROOT / THRESHOLDS_XLSX

THRESHOLDS_DF: Optional[pd.DataFrame] = None
THRESHOLDS_LOAD_ERROR: Optional[str] = None

try:
    THRESHOLDS_DF = pd.read_excel(THRESHOLDS_PATH)

    # Normalize column headers once (matches robust app behavior)
    THRESHOLDS_DF.columns = [
        c.strip().lower()
        for c in THRESHOLDS_DF.columns
        if isinstance(c, str)
    ]
    THRESHOLDS_DF = THRESHOLDS_DF.loc[:, [c for c in THRESHOLDS_DF.columns if c]]
except Exception as e:
    THRESHOLDS_DF = None
    THRESHOLDS_LOAD_ERROR = str(e)


# ============================================================
# Shared helpers
# ============================================================

def safe_float(val):
    if val is None:
        return None
    try:
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return None
            if s.endswith("%"):
                return float(s[:-1].strip()) / 100.0
            return float(s)
        return float(val)
    except Exception:
        return None


# ============================================================
# Matching + aliases (ported from app_criteria_chatbot.py)
# ============================================================

def normalize_for_match(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("&", " and ")
    s = re.sub(r"\b([a-z])\.\s*([a-z])\.\b", r"\1\2", s)  # collapses u.s. -> us
    s = re.sub(r"[\.\,\(\)\[\]\{\}\:\;\!\?\/\\\|]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_capital_city_aliases() -> dict:
    """
    Optional enhancement from the robust app: map capital city -> country name.
    If geonamescache isn't installed, this silently returns {} (no breakage).
    """
    try:
        import geonamescache  # type: ignore
    except Exception:
        return {}

    gc = geonamescache.GeonamesCache()
    countries = gc.get_countries()

    aliases: Dict[str, str] = {}
    for _, c in countries.items():
        cap = (c.get("capital") or "").strip()
        cname = (c.get("name") or "").strip()
        if cap and cname:
            aliases.setdefault(cap.lower(), cname)

    return aliases


CAPITAL_CITY_TO_COUNTRY = build_capital_city_aliases()


# ---- Exact COUNTRY_ALIASES from app_criteria_chatbot.py ----
COUNTRY_ALIASES = {
    "usa": "United States",
    "us": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "united states of america": "United States",

    "uk": "United Kingdom",
    "u.k.": "United Kingdom",

    "tanzania": "Tanzania",

    "dr congo": "Democratic Republic of the Congo",
    "drc": "Democratic Republic of the Congo",

    "bosnia": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",

    "egypt": "Egypt, Arab Rep.",
    "arab republic of egypt": "Egypt, Arab Rep.",

    "iran": "Iran, Islamic Rep.",
    "islamic republic of iran": "Iran, Islamic Rep.",

    "venezuela": "Venezuela, RB",
    "bolivarian republic of venezuela": "Venezuela, RB",

    "russia": "Russian Federation",
    "russian federation": "Russian Federation",

    "south korea": "Korea, Rep.",
    "republic of korea": "Korea, Rep.",
    "korea": "Korea, Rep.",

    "north korea": "Korea, Dem. People's Rep.",
    "dprk": "Korea, Dem. People's Rep.",

    "syria": "Syrian Arab Republic",
    "syrian arab republic": "Syrian Arab Republic",

    "laos": "Lao PDR",
    "lao people's democratic republic": "Lao PDR",

    "vietnam": "Viet Nam",

    "yemen": "Yemen, Rep.",
    "republic of yemen": "Yemen, Rep.",

    "congo": "Congo, Rep.",
    "republic of the congo": "Congo, Rep.",
    "congo republic": "Congo, Rep.",

    "ivory coast": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",

    "gambia": "Gambia, The",

    "bahamas": "Bahamas, The",

    "micronesia": "Micronesia, Fed. Sts.",
    "federated states of micronesia": "Micronesia, Fed. Sts.",

    "slovakia": "Slovak Republic",

    "czech republic": "Czechia",

    "brunei": "Brunei Darussalam",

    "myanmar": "Myanmar",
    "burma": "Myanmar",

    "cape verde": "Cabo Verde",

    "swaziland": "Eswatini",
}

# IMPORTANT: add the high-impact definite-article variants your users actually type.
# This is not in the original dict, but it prevents "the bahamas" misses.
COUNTRY_ALIASES.setdefault("the bahamas", "Bahamas, The")
COUNTRY_ALIASES.setdefault("the gambia", "Gambia, The")


# ---- Exact COUNTRY_ADJECTIVE_ALIASES from app_criteria_chatbot.py ----
COUNTRY_ADJECTIVE_ALIASES = {
    "nigerian": "nigeria",
    "kenyan": "kenya",
    "tanzanian": "tanzania",
    "bosnian": "bosnia and herzegovina",
    "salvadorian": "el salvador",
}


# Canonical industries list drawn from the sheet (exact pattern from app)
CANONICAL_INDUSTRIES: List[str] = []
if THRESHOLDS_DF is not None and "industry" in THRESHOLDS_DF.columns:
    CANONICAL_INDUSTRIES = sorted({
        str(i).strip()
        for i in THRESHOLDS_DF["industry"].dropna()
        if str(i).strip()
    })


def apply_sector_aliases(text: str) -> str:
    """
    Exact logic from app_criteria_chatbot.py: inject lightweight hints
    into the text so the model has better signals.
    """
    if not text:
        return text
    t = text.lower()
    hints = []

    if any(w in t for w in ["bottling", "food processing", "canner", "canning", "packaging plant"]):
        hints.append("manufacturing")

    if any(w in t for w in ["farm", "farmer", "agriculture", "agricultural", "agribusiness", "crop", "livestock"]):
        hints.append("agriculture")

    if any(w in t for w in ["winery", "wine maker", "vineyard", "brewery", "distillery", "beer", "spirits"]):
        hints.append("manufacturing")

    if any(w in t for w in ["fund", "investment fund", "private equity", "venture capital", "bank", "microfinance"]):
        hints.append("finance and insurance")

    if any(w in t for w in ["clinic", "hospital", "healthcare", "health care", "medical center"]):
        hints.append("health and social work")

    if any(w in t for w in ["school", "university", "college", "training center"]):
        hints.append("education")

    if any(w in t for w in ["factory", "plant", "production line", "assembly line"]):
        hints.append("manufacturing")

    if "light manufacturing" in t:
        hints.append("manufacturing")

    if any(w in t for w in ["bakery", "baking", "bread", "pastry"]):
        hints.append("manufacturing")
        hints.append("accommodation and food service activities")

    if any(w in t for w in ["telecom", "telecommunications", "mobile operator"]):
        hints.append("information and communication")

    if any(w in t for w in ["pottery", "ceramics", "ceramic"]):
        hints.append("manufacturing")

    if "widget" in t:
        hints.append("manufacturing")

    if not hints:
        return text

    hints_str = ", ".join(sorted(set(hints)))
    return f"{text}\n\n[Sector hints: {hints_str}]"


# ============================================================
# LLM sector classification (ported, but OPTIONAL)
# ============================================================

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")

client = None
if OpenAI is not None and _OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=_OPENAI_API_KEY)
    except Exception:
        client = None


def classify_sector_with_model(text: str) -> str:
    """
    Same contract as the robust app:
    return exactly one of CANONICAL_INDUSTRIES or 'None'.

    Difference vs the robust app:
    - If OPENAI_API_KEY is not set, we return 'None' rather than raising,
      so thresholds still work in offline/local environments.
    """
    if not text or not CANONICAL_INDUSTRIES or client is None:
        return "None"

    sectors_list = "\n".join(f"- {s}" for s in CANONICAL_INDUSTRIES)

    system_prompt = (
        "You are a classifier that maps company or investment descriptions to a single high-level sector, "
        "chosen from the list below. Always return exactly one of these labels, or 'None' if the text is not about "
        "an economic activity.\n\n"
        "Sectors:\n"
        f"{sectors_list}\n\n"
        "Return only the sector name or 'None'."
    )

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            max_tokens=10,
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()

        for s in CANONICAL_INDUSTRIES:
            if s.lower() == raw.lower():
                return s
        for s in CANONICAL_INDUSTRIES:
            if s.lower() in raw.lower():
                return s
        return "None"
    except Exception as e:
        print("Error in classify_sector_with_model:", e)
        traceback.print_exc()
        return "None"


# ============================================================
# Country + industry detection (robust app flow)
# ============================================================

def detect_country_and_industry_from_text(text: str):
    """
    Robust flow:
    - Country: adjective → exact match → alias → capital city
    - Industry: exact match → (hints + LLM) fallback
    """
    if not text or THRESHOLDS_DF is None:
        return None, None

    df = THRESHOLDS_DF
    t = normalize_for_match(text)

    # --- country detection ---
    if "country" in df.columns:
        raw_countries = [
            str(c).strip()
            for c in df["country"].dropna()
            if str(c).strip()
        ]
    else:
        raw_countries = []

    country_map = {c.lower(): c for c in raw_countries}
    country_key = None

    # 1) adjectival forms
    for adj, canonical_lower in COUNTRY_ADJECTIVE_ALIASES.items():
        if re.search(r"\b" + re.escape(adj) + r"\b", t):
            if canonical_lower in country_map:
                country_key = country_map[canonical_lower]
            else:
                country_key = canonical_lower.title()
            break

    # 2) exact country names from sheet (longest-first)
    if country_key is None:
        for c in sorted(raw_countries, key=len, reverse=True):
            if re.search(r"\b" + re.escape(c.lower()) + r"\b", t):
                country_key = c
                break

    # 3) alias hits (prefer exact spelling from sheet if present)
    if country_key is None:
        for alias, canonical in COUNTRY_ALIASES.items():
            if re.search(r"\b" + re.escape(alias) + r"\b", t):
                canonical_low = canonical.lower()
                match = next((c for c in raw_countries if c.lower() == canonical_low), None)
                country_key = match or canonical
                break

    # 4) capital city mapping fallback
    if country_key is None and CAPITAL_CITY_TO_COUNTRY:
        for token in re.findall(r"[a-zA-Z][a-zA-Z\-']+", text):
            tok = token.lower()
            if tok in CAPITAL_CITY_TO_COUNTRY:
                country_key = CAPITAL_CITY_TO_COUNTRY[tok]
                break

    # --- industry detection ---
    industry_key = None
    if "industry" in df.columns:
        raw_industries = [
            str(v).strip()
            for v in df["industry"].dropna()
            if str(v).strip()
        ]

        # exact match (longest-first)
        for ind in sorted(raw_industries, key=len, reverse=True):
            if re.search(r"\b" + re.escape(ind.lower()) + r"\b", t):
                industry_key = ind
                break

    # LLM fallback with sector hints
    if industry_key is None:
        aliased_text = apply_sector_aliases(text)
        sector = classify_sector_with_model(aliased_text)
        if sector != "None":
            industry_key = sector

    return country_key, industry_key


# ============================================================
# Public API used by rag.py/app.py
# ============================================================

def lookup_threshold(text: str):
    if THRESHOLDS_DF is None:
        return None

    df = THRESHOLDS_DF
    country_key, industry_key = detect_country_and_industry_from_text(text)

    if not country_key:
        return None

    df_country = df[
        df["country"].astype(str).str.strip().str.lower()
        == str(country_key).strip().lower()
    ]
    if df_country.empty:
        return None

    if industry_key and "industry" in df.columns:
        df_country_ind = df_country[
            df_country["industry"].astype(str).str.strip().str.lower()
            == str(industry_key).strip().lower()
        ]

        if not df_country_ind.empty:
            df_use = df_country_ind
        else:
            df_use = df_country
    else:
        df_use = df_country

    records = df_use.to_dict(orient="records")
    cleaned_records = []
    for rec in records:
        clean_rec = {}
        for k, v in rec.items():
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, float) and math.isnan(v):
                v = None
            clean_rec[k] = v
        cleaned_records.append(clean_rec)

    return cleaned_records, country_key, industry_key


# ============================================================
# Your existing summarization + formatting (unchanged)
# ============================================================

def summarise_thresholds(threshold_rows):
    summary = {
        "country": None,
        "industry": None,
        "board_or_investment_committee": None,
        "senior_leadership": None,
        "employees": None,
    }
    if not threshold_rows:
        return summary

    records = [r for r in threshold_rows if isinstance(r, dict)]

    country_val = next((r.get("country") for r in records if r.get("country")), None)
    industry_val = next((r.get("industry") for r in records if r.get("industry")), None)

    if country_val:
        summary["country"] = country_val
    if industry_val:
        summary["industry"] = industry_val

    uid_field = None
    for cand in ["unique id", "unique_id", "uid"]:
        if any(cand in r for r in records):
            uid_field = cand
            break

    #  extract numeric threshold.
    def _extract_numeric_threshold(rec):
        thr_raw = rec.get("threshold")
        thr = safe_float(thr_raw)
        if thr is None:
            return None
        if isinstance(thr, float) and math.isnan(thr):
            return None
        return thr

    #  pick by unique id.
    def _pick_by_unique_id(indicator_label: str, target_key: str):
        if not uid_field:
            return
        for rec in records:
            uid = str(rec.get(uid_field, "")).strip().lower()
            if uid == indicator_label:
                summary[target_key] = _extract_numeric_threshold(rec)
                return

    #  try canonical ids first (handles both old + new versions of the sheet).
    _pick_by_unique_id("board or investment committee", "board_or_investment_committee")
    _pick_by_unique_id("senior leadership", "senior_leadership")
    _pick_by_unique_id("employees", "employees")

    if any(summary[k] is not None for k in ["board_or_investment_committee", "senior_leadership", "employees"]):
        return summary

    # fallback if no UID column or ids differ.
    for rec in records:
        indicator = str(rec.get("indicator", "")).strip().lower()
        thr = _extract_numeric_threshold(rec)
        if not indicator or thr is None:
            continue
        if "board" in indicator or "investment committee" in indicator:
            summary["board_or_investment_committee"] = thr
        elif "senior" in indicator or "management" in indicator or "leadership" in indicator:
            summary["senior_leadership"] = thr
        elif "employee" in indicator or "workforce" in indicator or "staff" in indicator:
            summary["employees"] = thr

    return summary


def format_threshold_bullets(threshold_summary):
    if not threshold_summary:
        return ""

    def _fmt(p):
        if p is None:
            return None
        try:
            if p <= 1:
                return f"{int(round(p*100))}%"
            return f"{p}"
        except Exception:
            return str(p)

    lines = []
    c = threshold_summary.get("country")
    i = threshold_summary.get("industry")
    if c and i:
        lines.append(f"Country: {c}")
        lines.append(f"Industry: {i}")
    elif c:
        lines.append(f"Country: {c}")
    elif i:
        lines.append(f"Industry: {i}")

    bic = _fmt(threshold_summary.get("board_or_investment_committee"))
    sl = _fmt(threshold_summary.get("senior_leadership"))
    emp = _fmt(threshold_summary.get("employees"))

    if bic is not None:
        lines.append(f"Board / Investment Committee: {bic}")
    if sl is not None:
        lines.append(f"Senior Leadership: {sl}")
    if emp is not None:
        lines.append(f"Employees: {emp}")

    return "\n".join(lines).strip()