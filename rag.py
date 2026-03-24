import os
import json
import math
import re
import pickle
import traceback
from pathlib import Path, PureWindowsPath
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Manifest loader
# The manifest (resources_manifest_llm.json) contains metadata for every
# document in the library: title, publisher, year, quality tier, topic tags, etc.
# It is loaded once when the module is first imported and kept in memory for
# the lifetime of the app — loading it on every request would be too slow.
# ---------------------------------------------------------------------------

def _load_manifest_once() -> Dict[str, Dict[str, Any]]:
    """
    Reads data/resources_manifest_llm.json and builds a lookup dictionary
    that can be searched by either resource_id or filename. This means the
    rest of the code can find a document's metadata regardless of which
    identifier it has for the document.

    Called once at startup. The result is stored in the module-level
    MANIFEST variable and reused for every request.
    """
    project_root = Path(__file__).resolve().parent
    manifest_path = project_root / "data" / "resources_manifest_llm.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    combined: Dict[str, Dict[str, Any]] = {}
    for filename, meta in raw.items():
        rid = meta.get("resource_id")
        if rid:
            combined[rid] = meta
        combined[filename] = meta
    return combined


MANIFEST: Dict[str, Dict[str, Any]] = _load_manifest_once()


def _build_resource_catalogue(manifest: Dict[str, Dict[str, Any]]) -> str:
    """
    Builds a compact text catalogue of all documents in the library — one
    line per document with title, publisher, year, and topic tags.

    This catalogue is included in the prompt when the AI selects which
    sources to recommend to the user. Keeping it compact (one line per doc)
    means we can fit all 70+ documents into a single prompt without
    exceeding the token limit.

    Built once at startup alongside MANIFEST.
    """
    lines = []
    for rid, v in manifest.items():
        pub   = v.get("publisher") or ""
        year  = v.get("year") or ""
        title = v.get("title") or rid
        aud   = v.get("audience") or ""
        url   = v.get("url") or ""
        line  = f"{rid} | {pub} ({year}) | {title} | audience: {aud}"
        if url:
            line += f" | {url}"
        lines.append(line)
    return "\n".join(lines)


RESOURCE_CATALOGUE: str = _build_resource_catalogue(MANIFEST)


# ===========================================================================
# Threshold lookup (inlined from thresholds.py)
# Loads benchmarks.xlsx from the data/ subfolder and provides country/sector
# detection + threshold lookup used in Steps 7 and 9 of rag_query().
# ===========================================================================

_THRESH_PROJECT_ROOT = Path(__file__).resolve().parent
_THRESH_PATH = _THRESH_PROJECT_ROOT / "data" / "benchmarks.xlsx"

THRESHOLDS_DF: Optional[pd.DataFrame] = None

try:
    THRESHOLDS_DF = pd.read_excel(_THRESH_PATH)
    THRESHOLDS_DF.columns = [
        c.strip().lower()
        for c in THRESHOLDS_DF.columns
        if isinstance(c, str)
    ]
    THRESHOLDS_DF = THRESHOLDS_DF.loc[:, [c for c in THRESHOLDS_DF.columns if c]]
    print(f"[thresholds] Loaded OK: {len(THRESHOLDS_DF)} rows from {_THRESH_PATH}")
except Exception as _e:
    THRESHOLDS_DF = None
    print(f"[thresholds] WARNING: could not load {_THRESH_PATH}: {_e}")


def _thresh_safe_float(val):
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


def _normalize_for_match(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("\xa0", " ")  # non-breaking spaces from Excel
    s = s.replace("&", " and ")
    s = re.sub(r"\b([a-z])\.\s*([a-z])\.\b", r"\1\2", s)
    s = re.sub(r"[\.\,\(\)\[\]\{\}\:\;\!\?\/\\\|]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_capital_city_aliases() -> Dict[str, str]:
    try:
        import geonamescache  # type: ignore
        gc = geonamescache.GeonamesCache()
        aliases: Dict[str, str] = {}
        for _, c in gc.get_countries().items():
            cap   = (c.get("capital") or "").strip()
            cname = (c.get("name")    or "").strip()
            if cap and cname:
                aliases.setdefault(cap.lower(), cname)
        return aliases
    except Exception:
        return {}


_CAPITAL_CITY_TO_COUNTRY: Dict[str, str] = _build_capital_city_aliases()

_COUNTRY_ALIASES: Dict[str, str] = {
    "usa": "United States", "us": "United States",
    "u.s.": "United States", "u.s.a.": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom", "u.k.": "United Kingdom",
    "tanzania": "Tanzania",
    "dr congo": "Democratic Republic of the Congo",
    "drc": "Democratic Republic of the Congo",
    "bosnia": "Bosnia and Herzegovina",
    "egypt": "Egypt, Arab Rep.",
    "arab republic of egypt": "Egypt, Arab Rep.",
    "iran": "Iran, Islamic Rep.",
    "venezuela": "Venezuela, RB",
    "russia": "Russian Federation",
    "south korea": "Korea, Rep.", "republic of korea": "Korea, Rep.", "korea": "Korea, Rep.",
    "north korea": "Korea, Dem. People's Rep.", "dprk": "Korea, Dem. People's Rep.",
    "syria": "Syrian Arab Republic",
    "laos": "Lao PDR",
    "vietnam": "Viet Nam",
    "yemen": "Yemen, Rep.", "republic of yemen": "Yemen, Rep.",
    "congo": "Congo, Rep.", "republic of the congo": "Congo, Rep.",
    "ivory coast": "\u00c9te d'Ivoire", "cote d'ivoire": "C\u00f4te d'Ivoire",
    "gambia": "Gambia, The", "the gambia": "Gambia, The",
    "bahamas": "Bahamas, The", "the bahamas": "Bahamas, The",
    "micronesia": "Micronesia, Fed. Sts.",
    "slovakia": "Slovak Republic",
    "czech republic": "Czechia",
    "brunei": "Brunei Darussalam",
    "myanmar": "Myanmar", "burma": "Myanmar",
    "cape verde": "Cabo Verde",
    "swaziland": "Eswatini",

    # UAE variants
    "uae": "United Arab Emirates",
    "u.a.e.": "United Arab Emirates",
    "united arab emerites": "United Arab Emirates",  # common misspelling
    "united arab emeriates": "United Arab Emirates",  # common misspelling
    "dubai": "United Arab Emirates",
    "abu dhabi": "United Arab Emirates",
    "sharjah": "United Arab Emirates",

    # Other common aliases missing from original list
    "tanzania": "Tanzania",
    "türkiye": "Turkey",
    "turkey": "Turkey",
    "taiwan": "Taiwan",
    "palestine": "Palestine",
    "west bank": "Palestine",
    "hong kong": "Hong Kong",
    "macau": "Macao SAR",
    "macao": "Macao SAR",
    "republic of congo": "Republic of the Congo",
    "trinidad": "Trinidad and Tobago",
    "tobago": "Trinidad and Tobago",
    "saint lucia": "St. Lucia",
    "st lucia": "St. Lucia",
    "saint kitts": "St. Kitts and Nevis",
    "st kitts": "St. Kitts and Nevis",
    "saint vincent": "St. Vincent and the Grenadines",
    "st vincent": "St. Vincent and the Grenadines",
    "north korea": "Democratic People's Republic of Korea",
    "south korea": "Republic of Korea",
    "korea": "Republic of Korea",
    "bolivia": "Bolivia",
    "venezuela": "Bolivarian Republic\xa0of\xa0Venezuela",
    "iran": "Islamic Republic\xa0of\xa0Iran",
    "micronesia": "Federated\xa0States of\xa0Micronesia",
    "sao tome": "São Tomé\xa0and Príncipe",
    "sao tome and principe": "São Tomé\xa0and Príncipe",
}

_COUNTRY_ADJECTIVE_ALIASES: Dict[str, str] = {
    "nigerian": "nigeria",
    "kenyan": "kenya",
    "tanzanian": "tanzania",
    "bosnian": "bosnia and herzegovina",
    "salvadorian": "el salvador",
    "ghanaian": "ghana",
    "ugandan": "uganda",
    "rwandan": "rwanda",
    "ethiopian": "ethiopia",
    "zambian": "zambia",
    "zimbabwean": "zimbabwe",
    "bangladeshi": "bangladesh",
    "pakistani": "pakistan",
    "indonesian": "indonesia",
    "vietnamese": "viet nam",
    "peruvian": "peru",
    "colombian": "colombia",
    "mexican": "mexico",
    "brazilian": "brazil",
    "egyptian": "egypt, arab rep.",
    "moroccan": "morocco",
}

_CANONICAL_INDUSTRIES: List[str] = []
if THRESHOLDS_DF is not None and "industry" in THRESHOLDS_DF.columns:
    _CANONICAL_INDUSTRIES = sorted({
        str(i).strip()
        for i in THRESHOLDS_DF["industry"].dropna()
        if str(i).strip()
    })


def _classify_sector(text: str) -> str:
    """LLM sector classifier -- handles any company type via general knowledge."""
    if not text or not _CANONICAL_INDUSTRIES:
        return "None"
    try:
        from openai import OpenAI as _OAI
        _client = _OAI()
        _model  = os.getenv("OPENAI_CHAT_MODEL", os.getenv("OPENAI_ROUTER_MODEL", "gpt-4o-mini"))
        sectors_list = "\n".join(f"- {s}" for s in _CANONICAL_INDUSTRIES)
        system_prompt = (
            "You are an expert sector classifier for a gender-lens investing tool.\n\n"
            "Read the user's question and identify the company's PRIMARY economic activity.\n"
            "Map it to exactly ONE sector from the list below.\n\n"
            "Rules:\n"
            "1. Use your general knowledge -- do not require an exact keyword match.\n"
            "   Examples: 'pizza company' -> accommodation and food service activities;\n"
            "   'solar installer' -> electricity, gas, steam and air conditioning supply;\n"
            "   'online marketplace' -> information and communication.\n"
            "2. If a sector name is mentioned directly, use that.\n"
            "3. If no company/business is described, return 'None'.\n"
            "4. Return ONLY the exact sector label or 'None'. No explanation.\n\n"
            f"Sectors:\n{sectors_list}"
        )
        resp = _client.chat.completions.create(
            model=_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": text},
            ],
            max_tokens=20,
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        for s in _CANONICAL_INDUSTRIES:
            if s.lower() == raw.lower():
                return s
        for s in _CANONICAL_INDUSTRIES:
            if s.lower() in raw.lower():
                return s
        return "None"
    except Exception as e:
        print(f"[thresholds] classify_sector error: {e}")
        return "None"


def detect_country_and_industry_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Detect (country, industry) from free text. Returns (None, None) if no match."""
    if not text or THRESHOLDS_DF is None:
        return None, None

    t  = _normalize_for_match(text)
    df = THRESHOLDS_DF

    raw_countries = (
        [str(c).strip() for c in df["country"].dropna() if str(c).strip()]
        if "country" in df.columns else []
    )
    country_map  = {c.lower(): c for c in raw_countries}
    country_key: Optional[str] = None

    # 1) adjectival demonyms
    for adj, canon_lower in _COUNTRY_ADJECTIVE_ALIASES.items():
        if re.search(r"\b" + re.escape(adj) + r"\b", t):
            country_key = country_map.get(canon_lower, canon_lower.title())
            break

    # 2) exact name from sheet (longest first)
    if country_key is None:
        for c in sorted(raw_countries, key=len, reverse=True):
            if re.search(r"\b" + re.escape(c.lower()) + r"\b", t):
                country_key = c
                break

    # 3) alias scan
    if country_key is None:
        for alias, canonical in _COUNTRY_ALIASES.items():
            if re.search(r"\b" + re.escape(alias) + r"\b", t):
                canon_low = canonical.lower()
                country_key = country_map.get(canon_low, canonical)
                break

    # 4) capital city fallback
    if country_key is None:
        for token in re.findall(r"[a-zA-Z][a-zA-Z\-']+", text):
            if token.lower() in _CAPITAL_CITY_TO_COUNTRY:
                country_key = _CAPITAL_CITY_TO_COUNTRY[token.lower()]
                break

    # industry: exact match first
    industry_key: Optional[str] = None
    if "industry" in df.columns:
        raw_inds = [str(v).strip() for v in df["industry"].dropna() if str(v).strip()]
        for ind in sorted(raw_inds, key=len, reverse=True):
            if re.search(r"\b" + re.escape(ind.lower()) + r"\b", t):
                industry_key = ind
                break

    # industry: LLM fallback
    if industry_key is None:
        sector = _classify_sector(text)
        if sector != "None":
            industry_key = sector

    return country_key, industry_key


def lookup_threshold(text: str):
    """Return (records, country_key, industry_key) or None."""
    if THRESHOLDS_DF is None:
        print("[thresholds] lookup_threshold: THRESHOLDS_DF is None")
        return None

    country_key, industry_key = detect_country_and_industry_from_text(text)
    print(f"[thresholds] detected country={country_key!r}  industry={industry_key!r}")

    if not country_key:
        return None

    df = THRESHOLDS_DF
    df_country = df[df["country"].astype(str).str.strip().str.lower()
                    == str(country_key).strip().lower()]
    if df_country.empty:
        return None

    if industry_key and "industry" in df.columns:
        df_ind = df_country[df_country["industry"].astype(str).str.strip().str.lower()
                            == str(industry_key).strip().lower()]
        df_use = df_ind if not df_ind.empty else df_country
    else:
        df_use = df_country

    records = []
    for rec in df_use.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if isinstance(v, float) and math.isnan(v):
                v = None
            elif hasattr(v, "item"):
                v = v.item()
            clean[k] = v
        records.append(clean)
    return records, country_key, industry_key


def summarise_thresholds(threshold_rows) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "country": None, "industry": None,
        "board_or_investment_committee": None,
        "senior_leadership": None, "employees": None,
    }
    if not threshold_rows:
        return summary
    records = [r for r in threshold_rows if isinstance(r, dict)]
    summary["country"]  = next((r.get("country")  for r in records if r.get("country")),  None)
    summary["industry"] = next((r.get("industry") for r in records if r.get("industry")), None)

    uid_field = next((f for f in ["unique id", "unique_id", "uid"]
                      if any(f in r for r in records)), None)

    def _thr(rec):
        v = _thresh_safe_float(rec.get("threshold"))
        return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v

    def _pick(label, key):
        if not uid_field:
            return
        for rec in records:
            if str(rec.get(uid_field, "")).strip().lower() == label:
                summary[key] = _thr(rec)
                return

    _pick("board or investment committee", "board_or_investment_committee")
    _pick("senior leadership",             "senior_leadership")
    _pick("employees",                     "employees")

    if any(summary[k] is not None for k in
           ["board_or_investment_committee", "senior_leadership", "employees"]):
        return summary

    for rec in records:
        ind = str(rec.get("indicator", "")).strip().lower()
        v   = _thr(rec)
        if not ind or v is None:
            continue
        if "board" in ind or "investment committee" in ind:
            summary["board_or_investment_committee"] = v
        elif "senior" in ind or "management" in ind or "leadership" in ind:
            summary["senior_leadership"] = v
        elif "employee" in ind or "workforce" in ind or "staff" in ind:
            summary["employees"] = v
    return summary


def format_threshold_bullets(threshold_summary: dict) -> str:
    """Clean header + bulleted thresholds, matching app.py output style."""
    if not threshold_summary:
        return ""
    country  = (threshold_summary.get("country")  or "").strip()
    industry = (threshold_summary.get("industry") or "").strip()

    def _fmt(val):
        f = _thresh_safe_float(val)
        if f is None:
            return None
        pct = f * 100 if f <= 1.5 else f
        return f"{int(round(pct))}%" if abs(pct - round(pct)) < 1e-9 else f"{pct:.1f}%"

    bullets = []
    bic = _fmt(threshold_summary.get("board_or_investment_committee"))
    sl  = _fmt(threshold_summary.get("senior_leadership"))
    emp = _fmt(threshold_summary.get("employees"))
    if bic: bullets.append(f"- Board / Investment Committee: {bic}")
    if sl:  bullets.append(f"- Senior leadership: {sl}")
    if emp: bullets.append(f"- Employees: {emp}")

    if not bullets:
        return ""
    hc = country  if country  else "the selected country"
    hi = industry.lower() if industry else "the selected sector"
    return f"For the {hi} sector in {hc}, the 2X benchmark thresholds are:\n" + "\n".join(bullets)



def load_manifest(manifest_path: Path) -> dict:
    """Legacy helper kept for any callers that pass an explicit path."""
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_index(index_path: Path) -> Dict[str, Any]:
    with index_path.open("rb") as f:
        return pickle.load(f)


def embed_query(query: str) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI()
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    resp = client.embeddings.create(model=model, input=[query])
    return np.array(resp.data[0].embedding, dtype=np.float32)


def _basename_any(path_str: str) -> str:
    """Handle Windows paths even when running on Linux (Render, etc.)."""
    if "\\" in path_str:
        return PureWindowsPath(path_str).name
    return Path(path_str).name


def _ensure_normalized_vectors(index: Dict[str, Any]) -> np.ndarray:
    """Cache a normalized vectors matrix inside the loaded index dict."""
    if index.get("vectors_norm") is not None:
        return index["vectors_norm"]

    vectors = index["vectors"]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    index["vectors_norm"] = vectors / norms
    return index["vectors_norm"]


def cosine_sims(matrix_norm: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Cosine similarity via normalized dot product."""
    v = vec / (np.linalg.norm(vec) + 1e-9)
    return matrix_norm @ v


def _is_2x_primary(meta_entry: Dict[str, Any]) -> bool:
    """
    Returns True if a document is an official 2X Global primary source.
    These documents get a 20% similarity boost (see _diverse_top_k).

    Detection logic (checked in order, first match wins):
    1. Explicit flag: if the manifest entry has is_2x_primary: true, trust it.
    2. Publisher field: if publisher contains "2x global".
    3. Resource ID patterns: catches documents like the 2X Criteria Reference
       Guide even if the publisher field was entered incorrectly.
    4. Title signals: "2x criteria", "reference guide + 2x", etc.

    The multi-layer approach means a mislabelled document in the manifest
    still gets correctly identified as a primary source.
    """
    # 1. Explicit flag (set during manifest audit — most reliable)
    if meta_entry.get("is_2x_primary"):
        return True

    rid       = (meta_entry.get("resource_id") or "").lower()
    title     = (meta_entry.get("title")       or "").lower()
    publisher = (meta_entry.get("publisher")   or "").lower()

    # 2. Publisher field is correct
    if "2x global" in publisher:
        return True

    # 3. Resource-id patterns (catches mis-attributed docs like the criteria guide)
    if rid.startswith("2x_"):
        return True
    if "2x_global" in rid or "2x-global" in rid:
        return True
    if "2x_criteria" in rid or "2x-criteria" in rid:
        return True
    # Catches: 2025_OECD_2025-2X-Criteria-Reference-Guide-vF-1
    if re.search(r"2x.criteria", rid):
        return True

    # 4. Title signals
    if "2x criteria" in title:
        return True
    if "reference guide" in title and "2x" in title:
        return True
    if "principles for responsible exits" in title and "2x" in rid:
        return True

    return False


def _diverse_top_k(
    sims: np.ndarray,
    meta: List[Dict[str, Any]],
    k: int,
    institution_type: Optional[str] = None,
    pool_multiplier: int = 6,
    max_per_resource: int = 2,
    route: Optional[str] = None,
    question: Optional[str] = None,
) -> List[int]:
    """
    Selects the best k document chunks to use when answering a question.

    "Best" is not purely the closest semantic match — we also apply a set of
    boosts and penalties so that the most authoritative and relevant documents
    are preferred. The adjustments are cumulative and applied in this order:

      A. +20%  Official 2X primary sources (2X Criteria Reference Guide, etc.)
      B. +15%  tier_1 quality documents (the go-to reference for their topic)
      C. -20%  tier_3 documents (background only — cited less prominently)
      D. -50%  Documents aimed at a different audience type than the user
               (e.g. a "bank" document when the user is a fund manager)
      E. +18%  Documents whose topic tags match keywords in the question
               (only active on the "resources" route)

    We work on a copy of the similarity scores so the original array is never
    modified — important because the coverage check later needs the originals.

    The function also enforces diversity: no more than 2 chunks from the same
    document, so the answer draws on multiple sources rather than one long doc.
    """
    # Work on a copy so the original similarity scores are not modified.
    # The coverage check later in the pipeline uses the original scores to
    # decide how confident we are — if we modified them here, it would
    # see inflated scores and think coverage is better than it really is.
    sims = sims.copy()

    # Multipliers applied to similarity scores before ranking.
    # >1.0 = boost (document scores higher), <1.0 = penalty (scores lower).
    boost_2x      = 1.20   # Official 2X sources: always prioritised
    boost_tier1   = 1.15   # tier_1 documents: authoritative references
    penalty_tier3 = 0.80   # tier_3 documents: background only, ranked lower
    penalty_aud   = 0.50   # Wrong audience type: strongly deprioritised
    boost_tag     = 1.18   # Topic tag match: rewards topically relevant docs

    user_type_norm = institution_type.strip().lower() if institution_type else None

    # Keyword-to-tag mapping for the "resources" route.
    # When a user asks for templates, checklists, or practitioner tools,
    # we look for these keywords in their question and boost documents
    # whose topic tags match. This helps surface the right toolkit or
    # template even if the document text itself doesn't match well.
    # For example: "LP reporting language" → boost documents tagged
    # "investor_reporting" and "fund_structure".
    _TAG_SIGNALS = {
        "lp report":         ["investor_reporting", "fund_structure"],
        r"\bic memo\b":      ["investor_reporting", "fund_structure"],
        r"\bic\b.*memo":     ["investor_reporting", "fund_structure"],
        "reporting language":["investor_reporting"],
        "draft language":    ["investor_reporting"],
        "sample language":   ["investor_reporting"],
        "limited partner":   ["investor_reporting", "fund_structure"],
        r"\blpac\b":         ["investor_reporting", "fund_structure"],
        "investor report":   ["investor_reporting"],
        "esg report":        ["investor_reporting", "monitoring_reporting"],
        "due diligence":     ["due_diligence"],
        "diligence":         ["due_diligence"],
        r"ddq\b":            ["due_diligence", "fund_structure"],
        "blended finance":   ["blended_finance", "transaction_structuring"],
        "concessional":      ["blended_finance", "transaction_structuring"],
        r"first.loss":       ["blended_finance", "transaction_structuring"],
        "gender bond":       ["gender_bonds", "transaction_structuring"],
        r"gss bond":         ["gender_bonds", "transaction_structuring"],
        "bond framework":    ["gender_bonds", "transaction_structuring"],
        "covenant":          ["transaction_structuring"],
        "loan agreement":    ["transaction_structuring"],
        "fund structure":    ["fund_structure"],
        r"\bcarry\b":        ["fund_structure"],
        r"gp.{0,3}lp":      ["fund_structure", "investor_reporting"],
        r"\bbenchmark":      ["benchmarks"],
        r"\bthreshold":      ["benchmarks", "certification_assessment"],
        r"certif\w*":        ["certification_assessment"],
        "2x certif":         ["certification_assessment"],
        "evidence":          ["evidence_collection"],
        "monitoring":        ["monitoring_reporting"],
        r"\bkpi\b":          ["monitoring_reporting"],
        "supply chain":      ["supply_chain_procurement"],
        "supplier":          ["supply_chain_procurement"],
        "workforce":         ["workforce_practices"],
        "pay gap":           ["workforce_practices"],
        "parental leave":    ["workforce_practices"],
        r"women.led":        ["entrepreneurship_ownership"],
        r"entrepreneur\w*":  ["entrepreneurship_ownership"],
        r"\bsme\b":          ["entrepreneurship_ownership"],
        "technical assist":  ["technical_assistance"],
        "capacity build":    ["technical_assistance"],
        r"\breturns\b":      ["returns_data"],
        "financial perform": ["returns_data"],
        "business case":     ["returns_data"],
        r"\bconsumer\b":     ["consumer_markets"],
        "women as customer": ["consumer_markets"],
    }

    implied_tags: set = set()
    if route == "resources" and question:
        q_lower = question.lower()
        for pattern, tags in _TAG_SIGNALS.items():
            if re.search(pattern, q_lower):
                implied_tags.update(tags)

    for i, m in enumerate(meta):
        rid_key = m.get("resource_id") or _basename_any(m.get("local_path", ""))
        manifest_entry = MANIFEST.get(rid_key, {})

        # A. Boost Official 2X Sources
        if _is_2x_primary(m):
            sims[i] *= boost_2x

        # B. Quality tier boost / penalty
        tier = manifest_entry.get("quality_tier", "")
        if tier == "tier_1":
            sims[i] *= boost_tier1
        elif tier == "tier_3":
            sims[i] *= penalty_tier3

        # C. Audience filtering
        if user_type_norm:
            doc_audience = m.get("audience")
            if doc_audience:
                if isinstance(doc_audience, str):
                    audiences = {s.strip().lower() for s in doc_audience.split(",")}
                elif isinstance(doc_audience, list):
                    audiences = {str(s).strip().lower() for s in doc_audience}
                else:
                    audiences = set()

                if audiences:
                    is_general = any(x in audiences for x in ["general", "all", "everyone", "public"])
                    if not is_general and (user_type_norm not in audiences):
                        sims[i] *= penalty_aud

        # D. Topic tag match boost (resources route only)
        if implied_tags and route == "resources":
            doc_tags_raw = manifest_entry.get("topic_tags") or []
            if isinstance(doc_tags_raw, str):
                doc_tags = {t.strip() for t in doc_tags_raw.split(";")}
            elif isinstance(doc_tags_raw, list):
                doc_tags = {str(t).strip() for t in doc_tags_raw}
            else:
                doc_tags = set()
            if doc_tags & implied_tags:
                sims[i] *= boost_tag


    # Diversity selection: consider a larger candidate pool (pool_size chunks),
    # then pick the top k while enforcing that no single document contributes
    # more than max_per_resource chunks. This ensures the answer draws on
    # several sources rather than repeating text from one long document.
    pool_size = max(k * pool_multiplier, 30)
    cand_idxs = np.argsort(-sims)[:pool_size].tolist()

    picked: List[int] = []
    counts: Dict[str, int] = {}

    for i in cand_idxs:
        rid = meta[i].get("resource_id") or _basename_any(meta[i].get("local_path", ""))
        counts.setdefault(rid, 0)
        if counts[rid] >= max_per_resource:
            continue
        picked.append(i)
        counts[rid] += 1
        if len(picked) >= k:
            break

    # If the diversity constraint left us with fewer than k chunks
    # (e.g. because there are very few documents on the topic), relax it
    # and fill the remaining slots from the candidate pool.
    if len(picked) < k:
        for i in cand_idxs:
            if i in picked:
                continue
            picked.append(i)
            if len(picked) >= k:
                break

    return picked


ROUTE_LABELS = [
    "guidance",
    "comparison",
    "resources",
    "find_resource",
    "benchmarks",
    "qualification",
    "evidence",
    "implementation",
]


def _route_question_llm(
    question: str,
    response_mode: Optional[str] = None,
    institution_type: Optional[str] = None,
) -> str:
    """
    LLM-based routing. Returns one of ROUTE_LABELS.
    Falls back to deterministic routing on any error.
    """
    if response_mode:
        return response_mode.strip().lower()

    q = (question or "").strip()
    if not q:
        return "guidance"

    use_router = str(os.getenv("RAG_USE_LLM_ROUTER", "0")).strip() == "1"
    if not use_router:
        return _route_question(question, response_mode=None)

    try:
        from openai import OpenAI
        client = OpenAI()
        model = os.getenv("OPENAI_ROUTER_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))

        system = (
            "You are a routing classifier for a 2X Global member chatbot.\n"
            "Classify the user's question into exactly one route label from this list:\n"
            f"{ROUTE_LABELS}\n\n"
            "Return ONLY valid JSON with keys: route, confidence, rationale.\n"
            "- route: one of the labels\n"
            "- confidence: high|medium|low\n"
            "- rationale: <= 20 words\n\n"
            "Routing guidance:\n"
            "- benchmarks: user asks for benchmark/threshold values by country/sector — no user data provided\n"
            "- qualification: user provides their own percentages/data OR uses possessives (we/our/they/their/investee) to ask if they meet criteria\n"
            "- implementation: 'can X qualify' or 'what does X need' with NO user data — conceptual, not situational\n"
            "- evidence: what documentation/proof to collect for a claim\n"
            "- find_resource: user wants a reading recommendation — best guide, toolkit, or document to learn from\n"
            "- resources: explicit request for templates, boilerplate, sample language, draft, checklists, ic memo\n"
            "- comparison: compare A vs B, trade-offs, difference, differ\n"
            "- guidance: definitions, interpretation, overviews, explainers, 'what does X mean', general help\n"
        )

        user = {
            "question": q,
            "institution_type": institution_type or None,
        }

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            temperature=0.0,
        )

        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        route = str(data.get("route") or "").strip().lower()

        if route in ROUTE_LABELS:
            return route

        return _route_question(question, response_mode=None)

    except Exception:
        return _route_question(question, response_mode=None)


def _route_question(question: str, response_mode: Optional[str] = None) -> str:
    """
    Lightweight, deterministic routing.

    Priority order (highest to lowest):
      1. comparison   — explicit A-vs-B signals
      2. resources    — explicit deliverable/draft request
      3. qualification (early exit) — user provides percentages + possessives
      4. benchmarks   — pure threshold lookup (no user data)
      5. guidance     — definitions, explainers, "what does X mean"
      6. qualification (late) — qual keywords + possessives but no percentages
      7. implementation — "can X qualify" with no data, or how-to
      8. evidence     — documentation/proof requests
      9. implementation — how-to catch-all
     10. guidance     — default
    """
    if response_mode:
        return response_mode.strip().lower()

    q = (question or "").strip().lower()

    # ── 0. Research framing — always guidance even if "compared to" appears ───
    if re.search(r"\bwhat (does|do) the (research|evidence|data|literature|studies)\b", q):
        return "guidance"

    # ── 1. Comparison ─────────────────────────────────────────────────────────
    if any(tok in q for tok in [
        "vs ", "versus",
        "difference", "differences", "differ",
        "pros and cons", "tradeoff", "trade-offs",
    ]) or re.search(r"\bcompar", q):   # catches compare/compared/comparing
        return "comparison"

    # ── 2a. Find resource — reading recommendation (must run BEFORE resources) ─
    # Explicit resource-finding phrases. Checked first so "find me a resource"
    # and "recommend a guide" don't fall into the resources deliverable route.
    if any(tok in q for tok in [
        "find me a resource", "find a resource", "find me resources",
        "recommend a", "best resource", "good resource", "useful resource",
        "helpful resource", "any resources", "are there resources",
        "what should i read", "what to read", "point me to",
        "where can i learn", "suggest a", "good guide", "good toolkit",
        "useful guide", "useful toolkit", "any guides", "any toolkits",
        "reading list", "further reading",
    ]) or re.search(
        r"\b(recommend|suggest)\b.{0,30}\b(resource|guide|toolkit|document|reading)\b", q
    ):
        return "find_resource"

    # ── 2b. Resources — explicit deliverable request ──────────────────────────
    # Use word boundary for "draft" / "terms of reference"; avoid "tor" substring
    # Exclude "policy" / "procedure" / "outline" — too generic (fires on "ESG policy in place")
    # "give me a" removed — too broad, was catching "give me a good resource"
    if any(tok in q for tok in [
        "template", "sample language", "boilerplate", "checklist",
        "model clause", "sample covenant", "ic memo",
        "write me", "can you draft", "can you write",
    ]) or re.search(r"\b(draft|terms of reference)\b", q):
        return "resources"

    # ── 3. Qualification (early) — user has provided data points ─────────────
    # Must run BEFORE benchmarks to catch "do they meet the threshold?" with percentages
    has_percentages = bool(re.search(r"\d+\s*%", q))
    has_possessive  = bool(re.search(
        r"\b(we|our|they|their|investee|portfolio company|my fund|the fund)\b", q
    ))

    if has_percentages and has_possessive:
        return "qualification"

    # ── 4. Benchmarks — pure lookup, no user data ─────────────────────────────
    if any(tok in q for tok in [
        "benchmark", "benchmarks", "threshold", "thresholds",
        "what is the threshold", "board threshold", "ic threshold",
        "investment committee threshold", "senior leadership threshold",
        "employees threshold", "country benchmark", "sector benchmark",
    ]):
        return "benchmarks"

    # ── 5. Guidance — definitions, explainers, "what does X mean" ────────────
    # Exclude "what does it need / what do we need" — those are operational
    # Also exclude compound questions like "what's the role of X AND how do you Y"
    # — those have an implementation tail that should dominate
    if re.search(r"\band how (do|does|would|can|should)\b", q):
        return "implementation"

    if re.search(
        r"\b(what is|what does|what do|define|definition|meaning of|explain|overview|introduction)\b", q
    ) and not re.search(
        r"\b(what does it need|what do (i|we|they) need|what criteria|what does a \w+ need)\b", q
    ):
        return "guidance"

    if any(tok in q for tok in [
        "substantive influence", "interpretation of",
        "new to this", "i'm new", "im new",
    ]):
        return "guidance"

    # ── 6a. "Does/would that count toward 2X?" — qualification without qual keyword ──
    if re.search(r"\b(does|would|will|can)\b.{0,30}\bcount (toward|towards|as|for)\b", q):
        return "qualification"

    # ── 6b. Qualification (late) — qual keywords + possessives, no percentages ─
    is_qual = bool(re.search(
        r"\b(qualif(y|ies|ication)?|aligned|alignment|eligible|eligibility|meet|meets)\b", q
    ))

    if is_qual and has_possessive:
        return "qualification"

    # ── 7. "Can X qualify" with no user data → guidance (conceptual question) ──
    if is_qual:
        return "guidance"

    # ── 8. Evidence ───────────────────────────────────────────────────────────
    if any(tok in q for tok in [
        "evidence", "documentation", "audit trail",
        "what proof", "substantiat",
        "what should we record", "what do we need to collect",
        "what data", "data should we", "data to collect",
        "should we be collecting", "need to collect",
    ]) or re.search(r"\b(audit|verify|document)\b", q):
        return "evidence"

    # ── 9. Implementation ─────────────────────────────────────────────────────
    if re.search(
        r"\b(how do i|how to|steps to|process|implement|integrate|apply|conduct|"
        r"set up|build|design|develop|roll out|operationalize|screen|"
        r"due diligence|dd|monitor|report|what does it need|what criteria|"
        r"what does a|what do i need)\b", q
    ):
        return "implementation"

    # ── 10. Default ───────────────────────────────────────────────────────────
    return "guidance"


def score_threshold_intent(question: str) -> int:
    """Heuristic: how likely the user is asking for benchmark threshold values."""
    if not question:
        return 0

    q = question.strip().lower()
    score = 0

    if any(x in q for x in ["threshold", "thresholds", "benchmark", "benchmarks", "baseline", "cutoff"]):
        score += 3
    if any(x in q for x in ["what %", "what percent", "what percentage", "percentage", "minimum", "at least", "target"]):
        score += 2
    if any(x in q for x in ["board", "investment committee", " ic", "ic ", "senior leadership", "senior management", "employees", "workforce", "staff"]):
        score += 2
    if any(x in q for x in ["what counts", "counts", "count toward", "include", "included", "interpret", "how do we calculate", "round"]):
        score -= 4
    if any(x in q for x in ["define", "definition", "explain", "in plain english", "overview", "introduction"]):
        score -= 1

    return score


def _estimate_k(question: str, route: str) -> int:
    """
    Dynamically size the retrieval budget (k) based on question complexity
    and route type.  Returns an integer between RAG_K_MIN and RAG_K_MAX.

    Simple definitional questions get fewer chunks (less noise, faster).
    Multi-topic comparative or implementation questions get more (broader coverage).
    """
    k_min     = int(os.getenv("RAG_K_MIN",     "3"))
    k_default = int(os.getenv("RAG_K_DEFAULT",  "6"))
    k_max     = int(os.getenv("RAG_K_MAX",     "12"))

    q   = (question or "").strip().lower()
    wc  = len(q.split())
    k   = k_default

    # ── Signals that push k DOWN (simple / focused question) ──
    definition_markers = [
        "what is", "define", "definition", "meaning of", "explain",
        "overview", "introduction", "how does",
    ]
    is_simple_definition = any(m in q for m in definition_markers) and wc < 20
    if is_simple_definition:
        return k_min

    # ── Signals that push k UP (complex / multi-topic question) ──
    complexity_score = 0

    # Multi-part questions
    if q.count("?") >= 2 or (" and " in q and wc > 15):
        complexity_score += 2

    # Comparative questions always need broader context
    if route == "comparison" or any(m in q for m in ["compare", "vs ", "versus", "difference between", "trade-off"]):
        complexity_score += 2

    # Multi-geography or multi-sector
    if re.search(r"\b(multiple|across|various|different)\b.*(countr|sector|region|geograph)", q):
        complexity_score += 2

    # Asks about multiple criteria or pillars
    criteria_mentions = sum(1 for t in [
        "leadership", "employment", "entrepreneurship", "supply chain",
        "products", "services", "portfolio", "esg", "governance"
    ] if t in q)
    if criteria_mentions >= 3:
        complexity_score += 2
    elif criteria_mentions == 2:
        complexity_score += 1

    # Implementation and evidence routes tend to need more examples/context
    if route in ("implementation", "evidence", "resources"):
        complexity_score += 1

    # Long questions are usually more specific/complex
    if wc > 40:
        complexity_score += 1
    if wc > 70:
        complexity_score += 1

    # Apply complexity score → k
    if complexity_score >= 5:
        k = k_max
    elif complexity_score >= 3:
        k = min(k_default + 4, k_max)
    elif complexity_score >= 1:
        k = min(k_default + 2, k_max)

    return max(k_min, min(k, k_max))





def _deterministic_alignment_assessment(
    threshold_summary: Optional[Dict[str, Any]] = None,
    provided_percentages: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Deterministic alignment pre-computation.

    KEY RULE: This function NEVER returns conclusion='aligned'.
    In chat, ESG and Governance can never be confirmed from text.
    Valid conclusions: 'not aligned' (hard blocker) or 'needs info' (all other cases).
    All percentage comparisons are done here in Python — never by the LLM.
    """
    result: Dict[str, Any] = {
        "conclusion":           "needs info",
        "structural_blockers":  [],
        "missing_requirements": [
            "ESG minimum compliance (must be verified separately).",
            "Governance & Accountability minimums (must be verified separately).",
        ],
        "direct_status":     "unknown",
        "esg_status":        "unconfirmed",   # ALWAYS unconfirmed in chat
        "governance_status": "unconfirmed",   # ALWAYS unconfirmed in chat
        "comparisons":       {},              # pre-computed per-indicator comparisons
        "conclusion_reason": "",
    }

    if not threshold_summary or not provided_percentages:
        result["conclusion_reason"] = (
            "Insufficient data — thresholds or user percentages missing."
        )
        return result

    # ── Per-indicator comparison (done in Python, never by the LLM) ──────────
    indicator_map = {
        "board_or_investment_committee": ("Board / Investment Committee", "board"),
        "senior_leadership":             ("Senior Leadership",            "senior_leadership"),
        "employees":                     ("Employees",                    "employees"),
    }

    any_checked = False
    any_met     = False

    for summary_key, (label, pct_key) in indicator_map.items():
        threshold_raw = threshold_summary.get(summary_key)
        if threshold_raw is None:
            threshold_raw = threshold_summary.get(pct_key)
        provided_raw = provided_percentages.get(pct_key)
        # also accept "investment_committee" key for board slot
        if provided_raw is None and pct_key == "board":
            provided_raw = provided_percentages.get("investment_committee")

        if threshold_raw is None or provided_raw is None:
            continue

        any_checked = True

        # Normalise to percentage points (0-100 scale)
        t = float(threshold_raw) * 100 if float(threshold_raw) <= 1.5 else float(threshold_raw)
        p = float(provided_raw)  * 100 if float(provided_raw)  <= 1.5 else float(provided_raw)

        meets = p >= t
        if meets:
            any_met = True

        result["comparisons"][pct_key] = {
            "label":     label,
            "threshold": round(t, 1),
            "provided":  round(p, 1),
            "meets":     meets,
            "display":   (
                f"{p:.1f}% {'≥' if meets else '<'} {t:.1f}% threshold "
                f"({'✓ meets' if meets else '✗ below threshold'})"
            ),
        }

    if not any_checked:
        result["conclusion_reason"] = (
            "No indicators could be compared — thresholds or percentages missing."
        )
        return result

    if any_met:
        result["direct_status"]    = "met"
        result["conclusion"]       = "needs info"
        result["conclusion_reason"] = (
            "At least one direct criterion benchmark is met. "
            "Cannot conclude 'aligned' until ESG and Governance are confirmed separately."
        )
    else:
        result["direct_status"]    = "not met"
        result["conclusion"]       = "not aligned"
        result["conclusion_reason"] = (
            "No direct criterion benchmark is met based on the provided percentages."
        )
        result["structural_blockers"].append(
            "Provided percentages fall below the applicable threshold for all compared indicators."
        )
        result["missing_requirements"] = [
            "Direct criterion: at least one benchmark (board/IC, senior leadership, or employees) must be met.",
        ]

    return result


def _extract_percentages_with_context(question: str) -> Dict[str, float]:
    """
    Extract percentage values from a question and map them to the correct
    indicator based on surrounding context words.

    BUG FIX over original: the old code took only the *first* percentage and
    guessed a single indicator.  This version scans all percentage mentions and
    assigns each to the closest preceding indicator keyword, so a question like
    "our board is 30% and senior leadership is 45%" correctly captures both.
    """
    q = question.lower()
    provided: Dict[str, float] = {}

    # Find all (position, value) pairs for percentages in the text
    pct_matches = [(m.start(), float(m.group(1))) for m in re.finditer(r"(\d{1,3})\s*%", q)]
    if not pct_matches:
        return provided

    # Indicator keywords and their canonical key
    indicator_patterns = [
        ("investment_committee", [r"investment\s+committee", r"\bic\b"]),
        ("board",                [r"\bboard\b"]),
        ("senior_leadership",    [r"senior\s+leadership", r"senior\s+management"]),
        ("employees",            [r"\bemployee", r"\bworkforce\b", r"\bstaff\b"]),
    ]

    for pct_pos, pct_val in pct_matches:
        # Look at the text *before* this percentage (up to 60 chars back)
        window = q[max(0, pct_pos - 60): pct_pos]
        matched_key = None
        matched_pos = -1

        for key, patterns in indicator_patterns:
            for pat in patterns:
                m = None
                for candidate in re.finditer(pat, window):
                    m = candidate  # take the last (closest) match
                if m and m.start() > matched_pos:
                    matched_key = key
                    matched_pos = m.start()

        if matched_key and matched_key not in provided:
            provided[matched_key] = pct_val

    return provided


def _question_specificity(question: str) -> Dict[str, Any]:
    """Heuristic classifier for how *specific / niche* a question is.

    High specificity → coverage thresholds are raised → more likely to return
    'general' rather than 'ok', preventing overconfident answers on thin corpus hits.

    Three signal types:
      1. Niche topic terms (crypto, AI, fragile states, etc.)
      2. Specific asset class or institution type (infrastructure debt, pension fund, etc.)
      3. Multi-constraint geography + asset class combinations
    """
    q = (question or "").strip().lower()

    niche_terms = [
        # Technology / alternative assets
        "cryptocurrency", "crypto", "mining", "blockchain",
        "ai", "artificial intelligence", "foundation model",
        "climate tech", "cleantech", "deep tech",
        "carbon credit", "nature-based", "nature based",
        # Context / fragility
        "fragile state", "fragile states", "conflict-affected", "fcv", "fcs",
        "post-conflict", "humanitarian",
        # Stage
        "startup", "start-up", "early-stage", "seed stage", "series a", "venture",
        # Specific asset classes / structures (likely thin in corpus)
        "infrastructure debt", "infrastructure fund", "real assets",
        "private credit", "direct lending", "mezzanine",
        "sovereign wealth", "pension fund", "insurance fund",
        "family office",
        # Specific geographies likely thin in corpus
        "sub-saharan", "sub saharan", "southeast asia", "south-east asia",
        "latin america", "central asia", "mena", "middle east",
        "pacific islands", "small island",
    ]

    constraint_markers = [" in ", " for ", " within ", " across ", " among "]

    niche_hits     = sum(1 for t in niche_terms if t in q)
    word_count     = len(q.split())
    has_constraint = any(m in q for m in constraint_markers) and word_count >= 10

    # Multi-constraint: geography + asset class / instrument → always specific
    geo_words   = {"africa", "asia", "latin", "mena", "pacific", "europe", "caribbean"}
    asset_words = {"debt", "equity", "fund", "bond", "loan", "facility", "vehicle",
                   "infrastructure", "credit", "pension", "insurance"}
    multi_constraint = (
        any(w in q for w in geo_words)
        and any(w in q for w in asset_words)
        and word_count >= 10
    )

    is_high = (
        niche_hits >= 1        # any niche term is enough — these are known corpus gaps
        or multi_constraint    # geo + asset class combo even without niche terms
    )

    return {
        "niche_hits":       float(niche_hits),
        "high_specificity": bool(is_high),
        "multi_constraint": bool(multi_constraint),
    }


def _coverage_flags(sims_sorted: List[float], question: str) -> Tuple[str, Dict[str, float]]:
    """Decide whether coverage is strong enough to attempt a *specific* answer."""
    min_max_sim   = float(os.getenv("RAG_MIN_MAX_SIM",        "0.30"))
    min_mean_top  = float(os.getenv("RAG_MIN_MEAN_TOP",       "0.22"))
    strong_max_sim  = float(os.getenv("RAG_MIN_MAX_SIM_STRONG",  "0.36"))
    strong_mean_top = float(os.getenv("RAG_MIN_MEAN_TOP_STRONG", "0.28"))

    top_n    = min(5, len(sims_sorted))
    top_sims = sims_sorted[:top_n] if top_n else []

    max_sim  = float(top_sims[0])                          if top_sims else 0.0
    mean_top = float(sum(top_sims) / len(top_sims))        if top_sims else 0.0

    spec     = _question_specificity(question)
    high_spec = spec["high_specificity"]

    stats = {
        "max_sim":          max_sim,
        "mean_top":         mean_top,
        "min_max_sim":      min_max_sim,
        "min_mean_top":     min_mean_top,
        "strong_max_sim":   strong_max_sim,
        "strong_mean_top":  strong_mean_top,
        "high_specificity": float(1.0 if high_spec else 0.0),
        "niche_hits":       float(spec["niche_hits"]),
        "multi_constraint": float(1.0 if spec.get("multi_constraint") else 0.0),
    }

    if max_sim < min_max_sim or mean_top < min_mean_top:
        return "thin", stats

    if high_spec and (max_sim < strong_max_sim or mean_top < strong_mean_top):
        return "general", stats

    if max_sim < strong_max_sim or mean_top < strong_mean_top:
        return "general", stats

    return "ok", stats


def format_citations(
    chunks: List[Dict[str, Any]],
    manifest: dict,
    question: Optional[str] = None,
    route: Optional[str] = None,
) -> str:
    """
    Takes a list of retrieved document chunks and returns a formatted list of
    citation strings for display in the chat UI.

    Each citation looks like:
        Publisher (Year) — Title
        -> One sentence describing why this document is relevant

    Deduplicates so the same document is never listed twice, even if multiple
    chunks from it were retrieved. Capped at RAG_MAX_SOURCES (default: 3).
    """
    q          = (question or "").strip().lower()
    route_norm = (route    or "").strip().lower()

    is_2x_question = ("2x" in q) or ("2-x" in q) or ("2 x" in q)
    is_comparison  = (route_norm == "comparison")

    if is_2x_question and not is_comparison:
        chunks_2x = [c for c in chunks if _is_2x_primary(
            MANIFEST.get(c.get("resource_id") or "", {})
            or MANIFEST.get(_basename_any(c.get("local_path", "")), {})
            or {}
        )]
        if chunks_2x:
            chunks = chunks_2x

    max_sources = int(os.getenv("RAG_MAX_SOURCES", "3"))

    order: List[str] = []
    pages_by_key: Dict[str, set] = {}
    # Track all aliases (rid AND filename) that resolve to the same canonical key
    # so the same document retrieved under different identifiers is never listed twice.
    alias_to_key: Dict[str, str] = {}

    def _canonical_key(c: Dict[str, Any]) -> str:
        rid      = (c.get("resource_id") or "").strip()
        filename = _basename_any(c.get("local_path", ""))
        for alias in [rid, filename]:
            if alias and alias in alias_to_key:
                return alias_to_key[alias]
        key = rid or filename or "(unknown)"
        if rid:
            alias_to_key[rid] = key
        if filename:
            alias_to_key[filename] = key
        return key

    for c in chunks:
        key = _canonical_key(c)
        if key not in pages_by_key:
            pages_by_key[key] = set()
            order.append(key)
        page = c.get("page_start")
        if page is not None:
            try:
                pages_by_key[key].add(int(page))
            except Exception:
                pages_by_key[key].add(page)

    lines: List[str] = []
    for key in order[:max_sources]:
        meta  = MANIFEST.get(key, {})
        title     = meta.get("title")     or meta.get("filename") or key
        publisher = meta.get("publisher") or ""
        year      = meta.get("year")
        url       = meta.get("url")       or ""

        if publisher and year:
            left = f"{publisher} ({year})"
        elif publisher:
            left = publisher
        elif year:
            left = f"({year})"
        else:
            left = ""

        label = f"{left} – {title}" if left else str(title)

        pages = sorted(pages_by_key.get(key, set()), key=lambda x: (isinstance(x, str), x))
        if pages:
            max_pages  = 5
            shown      = pages[:max_pages]
            truncated  = len(pages) > max_pages
            page_str   = ("p. " if len(shown) == 1 else "pp. ") + ", ".join(str(p) for p in shown)
            if truncated:
                page_str += "…"
            cite = f"{label} ({page_str})"
        else:
            cite = label

        if url:
            cite += f" — {url}"

        lines.append(f"- {cite}")

    return "\n".join(lines)


def build_context(chunks: List[Dict[str, Any]]) -> str:
    blocks = []
    for c in chunks:
        file_label = _basename_any(c.get("local_path", ""))
        blocks.append(
            f'[SOURCE: {file_label} | page {c.get("page_start")} | chunk {c.get("chunk_id")} | resource_id {c.get("resource_id")} ]\n'
            f'{c.get("text", "")}'
        )
    return "\n\n".join(blocks)


def _strip_confidence_line(text: str) -> str:
    """
    Remove the 'Confidence level: X' first line from LLM answer text.
    The traffic light color is surfaced via meta.traffic_light; the text
    line is redundant and clutters the answer display.
    Handles optional trailing newlines after the line.
    """
    stripped = text.lstrip()
    m = re.match(r"confidence\s*level\s*:\s*(high|medium|low)[^\n]*\n*", stripped, re.IGNORECASE)
    if m:
        return stripped[m.end():].lstrip("\n")
    return text


def _thin_coverage_answer(
    question: str,
    top_chunks: List[Dict[str, Any]],
    sources_block: str,
    route: str,
) -> Dict[str, Any]:
    """Non-LLM fallback when retrieval coverage is too thin."""
    hint_lines = []
    if route == "comparison":
        hint_lines.append("Try naming the two items you want compared (frameworks, standards, tools) and the decision context.")
    elif route == "resources":
        hint_lines.append("Tell me what deliverable you want (policy, IC memo language, covenants, checklist) and the instrument (equity/debt/fund).")
    elif route == "benchmarks":
        hint_lines.append("Share Country, Sector (or 'Overall'), and whether you mean Board/IC, Senior Leadership, or Employees.")
    elif route == "qualification":
        hint_lines.append("Tell me the criterion/pillar, instrument (equity/debt/fund), geography, and the investee's current percentages.")
    elif route == "evidence":
        hint_lines.append("Tell me what claim you want to substantiate, what documents/data you have, and what decision you're making (IC, certification, reporting).")
    else:
        hint_lines.append("If you share the institution type (bank, fund, DFI) and the task, I can try a more targeted search.")

    answer = (
        "I don't have strong coverage in the current resource library to answer that confidently. "
        "The retrieved excerpts only weakly match your question.\n\n"
        "What you can do next:\n"
        f"- Rephrase with more specific terms (framework name, sector, instrument, geography).\n"
        f"- {hint_lines[0]}\n"
    )

    return {
        "answer": answer.strip(),
        "sources": sources_block.split("\n") if sources_block else [],
        "retrieved": _structured_retrieved(top_chunks),
        "meta": {"route": route, "coverage": "thin", "traffic_light": "red"},
    }


def _curate_sources(
    question: str,
    answer: str,
    route: str,
    institution_type: Optional[str],
    top_chunks: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """
    Uses the AI to select the 2-3 most valuable documents for the user to
    read, based on their question and the answer just generated.

    Unlike the retrieval step (which finds chunks that semantically match
    the question), this step looks at the full 70-document catalogue and
    picks the best reading list — prioritising credibility, recency, and
    documents that add value beyond what the answer already covered.

    Returns a list of dicts with keys: resource_id, title, publisher, year,
    url, reason — or None if the call fails (the caller then falls back to
    the retrieval-based sources from format_citations).
    """
    try:
        from openai import OpenAI
        client = OpenAI()
        model  = os.getenv("OPENAI_ROUTER_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))

        # Hint: tell the LLM which docs were retrieved so it can go beyond them
        retrieved_ids = []
        seen_ids: set = set()
        for c in top_chunks:
            rid = (c.get("resource_id") or "").strip()
            fn  = _basename_any(c.get("local_path", ""))
            key = rid or fn
            if key and key not in seen_ids:
                seen_ids.add(key)
                if rid:
                    retrieved_ids.append(rid)

        retrieved_hint = (
            f"\nDocuments already retrieved for the answer (you may include or go beyond them):\n"
            + "\n".join(f"  - {r}" for r in retrieved_ids[:8])
        ) if retrieved_ids else ""

        audience_hint = f"\nUser institution type: {institution_type}" if institution_type else ""

        system = (
            "You are a research librarian for a 71-document gender lens investing resource library.\n"
            "Your job: given a practitioner's question and the answer they just received, "
            "select the 3–5 documents from the CATALOGUE below that would be most valuable "
            "for them to read.\n\n"
            "Selection criteria (in order of priority):\n"
            "1. Most directly addresses the specific topic, instrument, or institution type in the question\n"
            "2. Published by a credible, recognised institution\n"
            "3. Recent and actionable (prefer post-2020 unless an older doc is a definitive reference)\n"
            "4. Matches the user's apparent role or institution type\n"
            "5. Adds value beyond what the answer already covered — reading list, not just echo\n\n"
            "Return ONLY a valid JSON array, no other text, no markdown fences.\n"
            'Each element: {"resource_id": "<exact id from CATALOGUE>", '
            '"reason": "<one sentence, max 15 words, why this doc is valuable>"}\n\n'
            f"CATALOGUE:\n{RESOURCE_CATALOGUE}"
        )

        # Summarise the answer to keep tokens down (cap at ~600 chars)
        answer_summary = answer[:600] + ("…" if len(answer) > 600 else "")

        user = (
            f"Question: {question}\n\n"
            f"Answer produced:\n{answer_summary}\n"
            f"{retrieved_hint}"
            f"{audience_hint}\n\n"
            "Select the 2–3 most valuable documents for this practitioner to read next."
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.0,
        )

        raw = (resp.choices[0].message.content or "").strip()
        # Strip markdown fences if the model adds them despite instruction
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        selections = json.loads(raw)

        curated: List[Dict[str, Any]] = []
        seen: set = set()
        for item in selections:
            rid = (item.get("resource_id") or "").strip()
            if not rid or rid not in MANIFEST or rid in seen:
                continue   # hallucinated or duplicate — silently drop
            seen.add(rid)
            meta = MANIFEST[rid]
            curated.append({
                "resource_id": rid,
                "title":       meta.get("title")     or rid,
                "publisher":   meta.get("publisher") or "",
                "year":        meta.get("year"),
                "url":         meta.get("url")        or "",
                "reason":      (item.get("reason") or "").strip(),
            })

        return curated[:5] if len(curated) >= 2 else None

    except Exception:
        return None   # silent fallback — never break the main response


def _structured_retrieved(top_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build structured retrieved list using the module-level MANIFEST cache.
    Deduplicates by canonical document identity (resource_id preferred, else filename)
    so the same source never appears twice under different chunk identifiers.
    """
    structured = []
    seen: set = set()
    for c in top_chunks:
        filename = _basename_any(c.get("local_path", ""))
        rid      = (c.get("resource_id") or "").strip()
        canon    = rid or filename or "(unknown)"
        if canon in seen:
            continue
        seen.add(canon)
        if rid:
            seen.add(rid)
        if filename:
            seen.add(filename)
        meta = MANIFEST.get(rid, {}) or MANIFEST.get(filename, {})
        structured.append(
            {
                "resource_id": rid or None,
                "title":     meta.get("title")     or c.get("title") or filename,
                "publisher": meta.get("publisher") or "",
                "year":      meta.get("year"),
                "url":       meta.get("url")       or "",
                "file":      filename,
                "page":      c.get("page_start"),
                "chunk_id":  c.get("chunk_id"),
            }
        )
    return structured


def _build_answer_brief(
    question:          str,
    route:             str,
    top_chunks:        List[Dict[str, Any]],
    coverage:          str,
    stats:             Dict[str, Any],
    threshold_hit:     Optional[Any],
    threshold_context: str,
    provided_pcts:     Dict[str, float],
    expertise:         str,
    user_context:      Dict[str, str],
    history:           Optional[List[Dict[str, str]]],
) -> Dict[str, Any]:
    """
    Build a fully pre-computed answer brief before the LLM is called.

    Contains every fact the LLM needs to narrate:
      - route, coverage signal, retrieval stats
      - threshold values (deterministic from Excel)
      - per-indicator comparisons (Python math, not LLM arithmetic)
      - alignment conclusion (deterministic — never 'aligned' in chat)
      - what is confirmed vs what is still required
      - source chunks + citation block

    The LLM receives this as a FACT BLOCK it must explain, not a set of
    rules it must follow.  It cannot contradict pre-computed values.
    """
    brief: Dict[str, Any] = {
        "question":          question,
        "route":             route,
        "coverage":          coverage,
        "retrieval_stats":   stats,
        "expertise":         expertise,
        "user_context":      user_context,
        "history":           history or [],
        "top_chunks":        top_chunks,
        "threshold_context": threshold_context,
        "threshold_summary": {},
        "provided_pcts":     provided_pcts,
        "alignment":         None,
        "sources_block":     format_citations(
                                 top_chunks, MANIFEST,
                                 question=question, route=route
                             ),
    }

    # ── Threshold summary ─────────────────────────────────────────────────────
    if threshold_hit:
        try:
            threshold_rows, country_key, industry_key = threshold_hit
            summary = summarise_thresholds(threshold_rows)
            brief["threshold_summary"] = {
                "country":                      summary.get("country")  or country_key,
                "industry":                     summary.get("industry") or industry_key or "Overall",
                "board_or_investment_committee": summary.get("board_or_investment_committee"),
                "senior_leadership":             summary.get("senior_leadership"),
                "employees":                     summary.get("employees"),
            }
        except Exception:
            pass

    # ── Alignment assessment (qualification route only) ───────────────────────
    if route == "qualification":
        brief["alignment"] = _deterministic_alignment_assessment(
            threshold_summary=brief["threshold_summary"],
            provided_percentages=provided_pcts,
        )

    return brief


def _narrate_brief(brief: Dict[str, Any]) -> Dict[str, Any]:
    """
    The LLM's sole job: explain the pre-computed brief clearly.

    System prompt is ~200 tokens of formatting/tone guidance.
    The LLM receives pre-computed facts and must not contradict them.
    Coverage routing is structural — 'thin' never reaches the LLM at all.
    'general' coverage restricts the LLM to principles, no invented specifics.
    """
    from openai import OpenAI
    client = OpenAI()
    model  = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    route       = brief["route"]
    coverage    = brief["coverage"]
    expertise   = brief["expertise"]
    question    = brief["question"]
    top_chunks  = brief["top_chunks"]
    alignment   = brief["alignment"]
    threshold_summary = brief["threshold_summary"]
    provided_pcts     = brief["provided_pcts"]
    user_context      = brief["user_context"]
    history           = brief["history"]
    stats             = brief["retrieval_stats"]

    # ── Tone ──────────────────────────────────────────────────────────────────
    if expertise == "newbie":
        tone = (
            "The person is new to gender lens investing. Write warmly and clearly. "
            "Spell out acronyms (GLI, DFI, 2X Criteria) when first used. "
            "Use plain language and real-world analogies where helpful. "
            "Avoid jargon unless you immediately explain it."
        )
    elif expertise == "expert":
        tone = (
            "The person is a seasoned practitioner. Be concise and precise. "
            "Skip basic definitions. Lead with thresholds, requirements, and any important nuances. "
            "If guidance varies by context or the source is ambiguous, say so explicitly."
        )
    else:
        tone = (
            "Write for a practitioner audience — someone who knows the space but wants "
            "clear, actionable information without unnecessary hedging or padding."
        )

    # ── Entity-type framing ─────────────────────────────────────────────────────────────
    # Tells the LLM whose shoes to write from so it doesn't default to
    # investor framing for company/enterprise questions.
    _inst = (user_context.get("institution_type") or "").strip().lower()
    if _inst in ("enterprise", "company"):
        entity_framing = (
            "IMPORTANT: The user is an enterprise or company (not an investor, fund, or bank). "
            "Frame all answers from the perspective of a company integrating gender into its "
            "own operations, workforce, products, and supply chain. "
            "Do NOT use investor language: do not mention 'deal origination', 'portfolio management', "
            "'investees', 'due diligence on investments', 'responsible exits', or 'LP reporting'. "
            "Focus instead on: workforce practices, leadership diversity, product/service design, "
            "community engagement, supplier diversity, and internal governance."
        )
    elif _inst in ("fund manager", "fund"):
        entity_framing = (
            "The user is a fund manager. Frame answers from the perspective of a fund "
            "applying gender lens to investment selection, due diligence, portfolio monitoring, "
            "and LP reporting."
        )
    elif _inst in ("bank",):
        entity_framing = (
            "The user is a bank or financial institution. Frame answers around lending, "
            "client screening, financial products, and institutional gender policies."
        )
    elif _inst in ("dfi",):
        entity_framing = (
            "The user is a DFI. Frame answers around development mandates, concessional finance, "
            "blended structures, and portfolio-level gender integration."
        )
    else:
        entity_framing = ""

    if entity_framing:
        tone = tone + "\n\n" + entity_framing

    # ── Coverage signal ───────────────────────────────────────────────────────
    n   = len(top_chunks)
    ms  = stats.get("max_sim")
    mt  = stats.get("mean_top")
    sim = f" (best match: {ms:.2f}, mean top-5: {mt:.2f})" if ms and mt else ""

    if coverage == "ok":
        coverage_instruction = (
            f"{n} excerpts retrieved{sim}. Coverage is good — answer specifically "
            "using the excerpts. Ground claims in the source material."
        )
    else:
        coverage_instruction = (
            f"{n} excerpts retrieved{sim}. Coverage is moderate — stick to general "
            "principles supported by the excerpts. Do NOT state specific thresholds, "
            "percentages, or named policies unless they appear word-for-word in the excerpts. "
            "Where you're drawing on general principles rather than direct evidence, say so briefly."
        )

    # ── Route writing brief ───────────────────────────────────────────────────
    # These are editorial directions, not rigid templates.
    # The LLM should write natural prose that accomplishes these goals —
    # not fill in blanks or reproduce these headings literally.

    # Detect guidance sub-type: conceptual explainer vs "what does X need / what criteria apply"
    q_lower = question.strip().lower()
    is_criteria_question = bool(re.search(
        r"\b(what (does|do) (it|a|an|they|this) need|what criteria|what (does|do) \w+ need"
        r"|what (is|are) required|what (must|should) \w+ (have|do|show|demonstrate)"
        r"|how (does|do|can|would) \w+ qualif)\b",
        q_lower,
    ))

    if route == "guidance" and is_criteria_question:
        writing_brief = (
            "This is a 'what does X need / what criteria apply' question — not a live qualification check, "
            "but an explanation of what the framework requires.\n\n"
            "Write a clear, practical explanation that covers:\n"
            "- Which 2X criterion or criteria are relevant and why\n"
            "- What the direct requirements are (the specific indicator(s) that must be met)\n"
            "- Any conditions, caveats, or common complications specific to this context\n"
            "- What ESG and Governance requirements always apply regardless of criterion\n"
            "- A concrete next step the reader can take\n\n"
            "Write in flowing prose with short paragraphs. Use a bullet list only where "
            "you're enumerating distinct items (e.g. a checklist of requirements). "
            "Don't force the answer into a rigid structure — let the complexity of the "
            "question dictate the length and shape."
        )
    elif route == "guidance":
        writing_brief = (
            "Answer the question directly and clearly. Lead with the most useful information — "
            "don't build up to it. One or two sentences of direct answer, then explain the "
            "key points that give it context or nuance. Keep it concise; don't pad. "
            "If there's a practical implication or next step worth flagging, add it at the end."
        )
    elif route == "comparison":
        writing_brief = (
            "The person wants to understand the practical difference between two things "
            "and, ideally, know which one fits their situation.\n\n"
            "Lead with a one-sentence bottom line that captures the essential distinction. "
            "Then explain the most important difference in plain terms, followed by any "
            "meaningful trade-offs or caveats.\n\n"
            "If the excerpts support a 'when to use which' framing, close with that — "
            "but write it as natural prose, not as a formulaic 'If your goal is X, choose A / "
            "If your goal is Y, choose B' list. That pattern feels scripted. Instead, explain "
            "the contexts in which each is the better fit, in plain language.\n\n"
            "Write in prose throughout. Use a bullet list only if you're comparing three or "
            "more distinct attributes side-by-side where a table-like layout genuinely helps."
        )
    elif route == "qualification":
        writing_brief = (
            "The person wants to know where they stand against the 2X Criteria.\n\n"
            "The FACT BLOCK above contains pre-computed results — reflect them accurately. "
            "Your job is to explain what those results mean in plain language, not to restate "
            "the raw labels. Specifically:\n"
            "- Open with a clear plain-language verdict based on the conclusion in the FACT BLOCK\n"
            "- Explain what is and isn't confirmed, in human terms (not internal field names)\n"
            "- If comparisons are in the FACT BLOCK, narrate them naturally "
            "  (e.g. 'Your board representation of 37.5% exceeds the 30% threshold')\n"
            "- Be clear that ESG and Governance cannot be confirmed from this conversation "
            "  and explain briefly what that means practically\n"
            "- Close with one or two concrete next steps\n\n"
            "Do NOT reproduce internal labels like 'direct_status', 'esg_status', "
            "'UNCONFIRMED', 'NEEDS INFO', or 'structural_blockers' in your answer. "
            "Write as if you're explaining the situation to the person, not reading back a database record."
        )
    elif route == "evidence":
        writing_brief = (
            "The person wants to know what documentation or proof to gather.\n\n"
            "Lead with a direct answer about what evidence works best. Then give a "
            "practical breakdown: what counts as strong evidence, what's acceptable as an "
            "alternative, and what to watch out for. If there's anything specific to record "
            "for an IC memo or audit trail, include that.\n\n"
            "Be concrete — name the type of document, data source, or record, not just "
            "abstract categories."
        )
    elif route == "resources":
        writing_brief = (
            "The person has asked for a draft, template, or sample language.\n\n"
            "Produce the actual deliverable — don't describe what you're going to write, "
            "just write it. If BENCHMARK THRESHOLDS are provided above, use those exact "
            "figures; do not invent percentages.\n\n"
            "After the draft, add a brief 'Notes for adaptation' section (2–4 bullets) "
            "covering what the user should verify or adjust before using it. "
            "Keep the notes practical, not generic."
        )
    elif route == "implementation":
        writing_brief = (
            "The person wants to know how to do something operationally.\n\n"
            "Give a direct, practical answer. If there are distinct steps, use a numbered "
            "list. If it's more of a set of considerations, use short paragraphs or bullets. "
            "Ground recommendations in the source excerpts where possible. "
            "Flag any important caveats or decision points."
        )
    else:
        writing_brief = (
            "Answer the question directly and helpfully using the source excerpts. "
            "Be concrete and practical. Lead with the most useful information."
        )

    # ── Build fact block for qualification route ──────────────────────────────
    # Separates WHAT THE LLM KNOWS (data) from HOW TO USE IT (rules).
    # Internal field names (esg_status, direct_status, etc.) are kept here, not
    # reproduced in the answer — the writing brief instructs the LLM to narrate them.
    fact_block = ""
    if route == "qualification" and alignment:
        comparisons_text = ""
        for key, comp in (alignment.get("comparisons") or {}).items():
            comparisons_text += f"  - {comp['label']}: {comp['display']}\n"

        conclusion = alignment["conclusion"].upper()
        reason     = alignment["conclusion_reason"]
        direct_st  = alignment["direct_status"]
        blockers   = alignment.get("structural_blockers") or []
        missing    = alignment.get("missing_requirements") or []

        fact_block = (
            "━━━ PRE-COMPUTED RESULTS (do not recalculate — narrate these in plain language) ━━━\n"
            f"Conclusion: {conclusion}\n"
            f"Reason: {reason}\n"
            f"Direct criterion status: {direct_st}\n"
            "ESG compliance: cannot be confirmed from this conversation — always unconfirmed\n"
            "Governance & Accountability: cannot be confirmed from this conversation — always unconfirmed\n"
        )
        if comparisons_text:
            fact_block += f"Indicator comparisons (Python-computed — do not recalculate):\n{comparisons_text}"
        if blockers:
            fact_block += f"Hard blockers: {'; '.join(blockers)}\n"
        if missing:
            fact_block += "Still required to conclude aligned:\n"
            for req in missing:
                fact_block += f"  - {req}\n"
        fact_block += (
            "\nCRITICAL RULES:\n"
            "- Reflect the conclusion above exactly — do not upgrade it to 'aligned'.\n"
            "- Do not recalculate or second-guess the indicator comparisons.\n"
            "- Do not infer ESG or Governance status from any excerpt — they are always unconfirmed.\n"
            "- Do NOT reproduce internal field names in your answer text.\n\n"
        )

    elif route == "qualification":
        fact_block = (
            "━━━ PRE-COMPUTED RESULTS ━━━\n"
            "Conclusion: NEEDS INFO\n"
            "Reason: No threshold data or user percentages available to make a comparison.\n"
            "ESG compliance: cannot be confirmed from this conversation — always unconfirmed\n"
            "Governance & Accountability: cannot be confirmed from this conversation — always unconfirmed\n"
            "\nCRITICAL RULES:\n"
            "- Do not conclude 'aligned'. Do not infer ESG or Governance.\n"
            "- Do NOT reproduce internal field names in your answer text.\n\n"
        )

    # ── Threshold block ───────────────────────────────────────────────────────
    thresh_block = ""
    if threshold_summary:
        def _pct(v):
            if v is None: return "N/A"
            f = float(v) * 100 if float(v) <= 1.5 else float(v)
            return f"{f:.0f}%"
        thresh_block = (
            "━━━ BENCHMARK THRESHOLDS (from benchmarks.xlsx — do not recalculate) ━━━\n"
            f"  Country:           {threshold_summary.get('country', 'unknown')}\n"
            f"  Industry:          {threshold_summary.get('industry', 'Overall')}\n"
            f"  Board / IC:        {_pct(threshold_summary.get('board_or_investment_committee'))}\n"
            f"  Senior Leadership: {_pct(threshold_summary.get('senior_leadership'))}\n"
            f"  Employees:         {_pct(threshold_summary.get('employees'))}\n\n"
        )

    # ── System prompt ─────────────────────────────────────────────────────────
    system = (
        "You are the 2X Global gender lens investing assistant. "
        "You help fund managers, banks, DFIs, and enterprises navigate the 2X Criteria.\n\n"
        f"{tone}\n\n"
        "FORMAT RULES (always):\n"
        "- Your FIRST LINE must be exactly one of: "
        "'Confidence level: High', 'Confidence level: Medium', or 'Confidence level: Low'. "
        "Nothing before it.\n"
        "- After that, write naturally. Do not reproduce template headings like "
        "'Direct answer:', 'Key points:', 'Bottom line:' — these are prompts for you, not headers.\n"
        "- Do not use filenames, publishers, or page numbers in your answer text.\n"
        "- Do not write a Sources section — that is appended separately.\n"
        "- Do not reproduce internal field names (direct_status, esg_status, NEEDS INFO, "
        "UNCONFIRMED, structural_blockers) in your answer.\n"
        "- After your answer, on a new line output exactly: "
        "SOURCES_JSON: followed by a JSON array of the 2-3 most valuable documents "
        "for this practitioner to read. "
        "Each element: {resource_id: <exact id>, reason: <max 12 words>}. "
        "Use only resource_ids from the CATALOGUE below. No markdown fences.\n\n"
        f"CATALOGUE:\n{RESOURCE_CATALOGUE}\n\n"
        f"RETRIEVAL: {coverage_instruction}\n"
    )

    # ── User message ──────────────────────────────────────────────────────────
    uc_lines = [f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in user_context.items() if v]
    uc_str   = "\n".join(uc_lines)
    context  = build_context(top_chunks)

    user_msg = (
        f"Question: {question}\n\n"
        + (fact_block if fact_block else "")
        + (thresh_block if thresh_block else "")
        + (brief.get("threshold_context", "") + "\n\n"
           if brief.get("threshold_context") and not thresh_block else "")
        + (f"USER CONTEXT:\n{uc_str}\n\n" if uc_str else "")
        + f"SOURCE EXCERPTS:\n{context}\n\n"
        + "Answer using ONLY the FACT BLOCK (if present) and SOURCE EXCERPTS.\n\n"
        + f"WRITING BRIEF:\n{writing_brief}\n"
    )

    # ── History messages ──────────────────────────────────────────────────────
    history_msgs = []
    for m in (history or [])[-6:]:
        role    = (m.get("role")    or "").strip().lower()
        content = (m.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            history_msgs.append({"role": role, "content": content})

    resp = client.chat.completions.create(
        model=model,
        messages=(
            [{"role": "system", "content": system}]
            + history_msgs
            + [{"role": "user", "content": user_msg}]
        ),
        temperature=float(os.getenv("RAG_TEMPERATURE", "0.2")),
    )

    answer_text = resp.choices[0].message.content or ""

    # ── Traffic light — extract from raw text, then strip the line ────────────
    traffic_light = "gray"
    m = re.search(r"confidence\s*level:\s*(high|medium|low)", answer_text, re.IGNORECASE)
    if m:
        lvl = m.group(1).lower()
        traffic_light = {"high": "green", "medium": "yellow", "low": "red"}.get(lvl, "gray")
    answer_text = _strip_confidence_line(answer_text)

    # Benchmarks answers come from the Excel lookup, not the RAG corpus,
    # so no document sources are relevant or shown.
    if route == "benchmarks":
        return {
            "answer":    answer_text,
            "sources":   [],
            "retrieved": [],
            "curated":   [],
            "meta": {
                "route":          route,
                "coverage":       coverage,
                "traffic_light":  traffic_light,
                "expertise_used": expertise,
            },
        }

    # ── Parse SOURCES_JSON from answer — no second API call needed ───────────
    # _narrate_brief now asks the model to append SOURCES_JSON: [...] after its
    # answer. We split it out here, resolve IDs against MANIFEST, and render.
    sources_list: List[str] = []
    curated: List[Dict[str, Any]] = []

    if "SOURCES_JSON:" in answer_text:
        _parts = answer_text.split("SOURCES_JSON:", 1)
        answer_text = _parts[0].strip()
        try:
            _json_str = re.sub(
                r"^```(?:json)?\s*|\s*```$", "", _parts[1].strip(),
                flags=re.MULTILINE
            ).strip()
            for _item in json.loads(_json_str):
                _rid = (_item.get("resource_id") or "").strip()
                if not _rid or _rid not in MANIFEST:
                    continue
                _me = MANIFEST[_rid]
                _pub    = _me.get("publisher") or ""
                _year   = _me.get("year")
                _title  = _me.get("title") or _rid
                _url    = _me.get("url") or ""
                _reason = (_item.get("reason") or "").strip()
                _left   = f"{_pub} ({_year})" if _pub and _year else (_pub or (f"({_year})" if _year else ""))
                _label  = f"{_left} – {_title}" if _left else _title
                if _url:    _label += f" — {_url}"
                if _reason: _label += f"\n  → {_reason}"
                sources_list.append(f"- {_label}")
                curated.append(_me)
        except Exception:
            pass  # Fall through to retrieval-based sources

    # Fallback if model omitted SOURCES_JSON (benchmarks never shows sources)
    if not sources_list and route != "benchmarks":
        sources_list = brief["sources_block"].split("\n") if brief["sources_block"] else []

    return {
        "answer":    answer_text,
        "sources":   sources_list,
        "retrieved": [] if curated else _structured_retrieved(top_chunks),
        "curated":   curated,
        "meta": {
            "route":          route,
            "coverage":       coverage,
            "traffic_light":  traffic_light,
            "expertise_used": expertise,
        },
    }

# ---------------------------------------------------------------------------
# Stub kept for import compatibility — new code paths go through _narrate_brief
# ---------------------------------------------------------------------------

def _generate_llm_answer(
    question: str,
    top_chunks: List[Dict[str, Any]],
    sources_block: str,
    route: str,
    coverage_mode: str,
    expertise: str = "general",
    user_context: Dict[str, str] = {},
    history: List = [],
    threshold_context: str = "",
    alignment_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    from openai import OpenAI
    client = OpenAI()
    model   = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    context = build_context(top_chunks)

    if expertise == "newbie":
        tone_instruction = (
            "TONE: Educational, encouraging, and clear. \n"
            "Assume the user is new to Gender Lens Investing. \n"
            "Define any acronyms (like GIIN, DFI, 2X) on first use. \n"
            "Use analogies where helpful. Avoid jargon unless explained."
        )
    elif expertise == "expert":
        tone_instruction = (
            "TONE: Professional, concise, and technical. \n"
            "Assume the user is a seasoned Fund Manager. \n"
            "Skip all basic definitions. Prioritize numeric thresholds and specific legal/governance requirements. \n"
            "If the source is ambiguous, state 'Practice varies' and cite the specific guidance."
        )
    else:
        tone_instruction = (
            "TONE: Professional and practical. \n"
            "Write for a practitioner audience. Be concrete."
        )

    confidence_instruction = (
        "You must start your response (after any title) with a line: "
        "'Confidence level: High', 'Confidence level: Medium', or 'Confidence level: Low'."
    )

    alignment_guardrail = ""
    if route == "qualification":
        alignment_guardrail = (
            "ALIGNMENT RULES:\n"
            "- Do NOT conclude '2X aligned' unless ESG and Governance minimums are confirmed.\n"
            "- If information is missing, say 'needs info' rather than concluding alignment.\n"
            "- Meeting one benchmark alone does NOT equal full 2X alignment.\n\n"
        )

    if route == "guidance":
        template_hint = (
            "Direct answer: <1–2 sentences>\n\n"
            "Confidence level: <High/Medium/Low>\n\n"
            "Key points / Practical implications:\n"
            "- <bullet>\n"
            "- <bullet>\n\n"
            "If you want, I can tailor this to your context (instrument, sector, geography).\n"
        )
    elif route == "comparison":
        template_hint = (
            "Bottom line: <one sentence>\n\n"
            "Confidence level: <High/Medium/Low>\n\n"
            "Comparison (plain language):\n"
            "- Most important difference: <...>\n"
            "- Trade-offs: <...>\n"
            "- Data / evidence required: <...>\n"
            "- Implementation burden: <...>\n"
            "- Best fit contexts: <...>\n\n"
            "Recommendation:\n"
            "- If your goal is <X>, choose A.\n"
            "- If your goal is <Y>, choose B.\n"
        )
    elif route == "qualification":
        template_hint = (
            "Alignment conclusion (deterministic): <aligned / not aligned / needs info>\n\n"
            "Confidence level: <High/Medium/Low>\n\n"
            "Reasoning:\n"
            "- Direct benchmark comparison\n"
            "- ESG status: Not confirmed — user must verify against 2X ESG minimum requirements\n"
            "- Governance status: Not confirmed — user must verify against 2X Governance & Accountability minimums\n\n"
            "Next steps:\n"
            "- <bullet>\n"
        )
    elif route == "evidence":
        template_hint = (
            "Direct answer: <one sentence>\n\n"
            "Confidence level: <High/Medium/Low>\n\n"
            "Strong evidence (preferred):\n"
            "- <bullet>\n\n"
            "Acceptable alternatives (if data limited):\n"
            "- <bullet>\n\n"
            "Red flags / weak evidence:\n"
            "- <bullet>\n\n"
            "What to record (for IC / audit trail):\n"
            "- <bullet>\n"
        )
    elif route == "resources":
        template_hint = (
            "Deliverable: <what you will produce>\n\n"
            "Confidence level: <High/Medium/Low>\n\n"
            "Draft / Template:\n"
            "- <section heading>: <content>\n"
            "- <section heading>: <content>\n\n"
            "Notes for adaptation:\n"
            "- <bullet>\n"
        )
    else:
        template_hint = (
            "Direct answer: <one sentence>\n\n"
            "Confidence level: <High/Medium/Low>\n\n"
            "Key points / Steps:\n"
            "- <bullet>\n"
            "- <bullet>\n\n"
        )

    alignment_block = ""
    if route == "qualification" and alignment_result:
        alignment_block = (
            "DETERMINISTIC ALIGNMENT ASSESSMENT:\n"
            f"- Conclusion: {alignment_result.get('conclusion')}\n"
            f"- Direct status: {alignment_result.get('direct_status')}\n"
            f"- Structural blockers: {alignment_result.get('structural_blockers')}\n"
            f"- Missing requirements: {alignment_result.get('missing_requirements')}\n\n"
            "You MUST reflect this conclusion and may NOT contradict it.\n\n"
        )

    system = alignment_guardrail + alignment_block + (
        "You are the 2X Global member resource assistant for gender lens investing.\n"
        "Use the provided SOURCE excerpts as your evidence base.\n"
        f"{tone_instruction}\n"
        "If the excerpts do not contain enough information to answer confidently, say so plainly.\n"
        f"{confidence_instruction}\n"
        "Safety/behavior constraints:\n"
        "- Do NOT use outside knowledge beyond what is in the excerpts.\n"
        "- Do NOT mention documents, filenames, publishers, or page numbers in the answer text.\n"
        "- Do NOT write a Sources section in the answer text.\n"
    )

    if threshold_context:
        system += (
            "\nIf the user prompt includes a BENCHMARK THRESHOLDS block, you MUST do two things:\n"
            "1) Use those benchmark values when relevant.\n"
            "2) Add a short section titled 'Benchmark check' (2–5 bullets) explaining how the benchmark relates to the answer.\n"
            "Never invent benchmark numbers.\n"
        )

    if coverage_mode == "general":
        system += "\nIMPORTANT: The excerpts are only a moderate match. Provide GENERAL PRINCIPLES only. Do not invent specific thresholds."

    uc_lines = [f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in user_context.items() if v]
    user_context_str = "\n".join(uc_lines)

    history_msgs = []
    if history and isinstance(history, list):
        for m in history[-6:]:
            role    = (m.get("role")    or "").strip().lower()
            content = (m.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                history_msgs.append({"role": role, "content": content})

    user_msg = (
        f"Question: {question}\n\n"
        + (threshold_context + "\n\n" if threshold_context else "")
        + (f"USER CONTEXT:\n{user_context_str}\n\n" if user_context_str else "")
        + f"SOURCE EXCERPTS:\n{context}\n\n"
        "Write the answer using ONLY the excerpts.\n"
        f"Use this template:\n{template_hint}\n"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=(
            [{"role": "system", "content": system}]
            + history_msgs
            + [{"role": "user", "content": user_msg}]
        ),
        temperature=float(os.getenv("RAG_TEMPERATURE", "0.2")),
    )

    answer_text = resp.choices[0].message.content or ""

    traffic_light = "gray"
    match = re.search(r"confidence\s*level:\s*(high|medium|low)", answer_text, re.IGNORECASE)
    if match:
        level = match.group(1).lower()
        if level == "high":   traffic_light = "green"
        elif level == "medium": traffic_light = "yellow"
        elif level == "low":    traffic_light = "red"
    answer_text = _strip_confidence_line(answer_text)

    return {
        "answer": answer_text,
        "sources": sources_block.split("\n") if sources_block else [],
        "retrieved": _structured_retrieved(top_chunks),
        "meta": {
            "route":          route,
            "coverage":       coverage_mode,
            "traffic_light":  traffic_light,
            "expertise_used": expertise,
        },
    }


def answer_question(
    question: str,
    top_chunks: List[Dict[str, Any]],
    route: str = "implementation",
    expertise: str = "general",
    audience: Optional[str] = None,
    institution_type: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    asset_class: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    coverage: str = "ok",
    threshold_context: str = "",
    alignment_result: Optional[Dict[str, Any]] = None,
    retrieval_stats: Optional[Dict[str, float]] = None,
    threshold_hit: Optional[Any] = None,
    provided_pcts: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:

    # ── Structural thin-coverage gate (no LLM at all) ─────────────────────────
    if coverage == "thin":
        sources_block = format_citations(top_chunks, MANIFEST, question=question, route=route)
        return _thin_coverage_answer(question, top_chunks, sources_block, route)

    # ── Build the pre-computed brief ──────────────────────────────────────────
    brief = _build_answer_brief(
        question=question,
        route=route,
        top_chunks=top_chunks,
        coverage=coverage,
        stats=retrieval_stats or {},
        threshold_hit=threshold_hit,
        threshold_context=threshold_context,
        provided_pcts=provided_pcts or {},
        expertise=expertise,
        user_context={
            "audience":         audience,
            "institution_type": institution_type,
            "jurisdiction":     jurisdiction,
            "asset_class":      asset_class,
        },
        history=history,
    )

    # ── LLM narrates the brief ────────────────────────────────────────────────
    return _narrate_brief(brief)


# ---------------------------------------------------------------------------
# find_resource route
# Skips RAG retrieval entirely. One gpt-4o-mini call against the full
# RESOURCE_CATALOGUE to pick and narrate the 2 best documents.
# ---------------------------------------------------------------------------

def _find_resource_llm(
    question: str,
    institution_type: Optional[str],
) -> Dict[str, Any]:
    """Select and narrate the top 2 resources. Returns a full result dict."""
    model = os.getenv("OPENAI_ROUTER_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    audience_hint = f"\nUser institution type: {institution_type}" if institution_type else ""

    system = (
        "You are a research librarian for a gender lens investing resource library.\n"
        "A practitioner wants a reading recommendation.\n\n"
        "Your task:\n"
        "1. Select the 2 documents from the CATALOGUE below that best answer their question.\n"
        "2. For each, write 2-3 sentences: what the document is, who published it, and "
        "specifically why it is the right fit for this question.\n"
        "3. End with one concrete action the user should take when reading the top pick \n"  "   (e.g. go straight to chapter X, use the checklist on page Y, apply the framework to Z).\n\n"
        "Rules:\n"
        "- Prioritise relevance to the specific question over general quality.\n"
        "- Prefer tier_1 documents and post-2020 publications unless an older one is uniquely relevant.\n"
        "- Match the user's institution type when possible.\n"
        "- Be direct — no hedging, no 'it depends', no bullet lists.\n"
        "- Do NOT mention document IDs or filenames.\n"
        "- Write in flowing prose.\n\n"
        "After your prose answer, on a new line output:\n"
        "SOURCES_JSON: [{resource_id: <id>, reason: <max 12 words>}]\n\n"
        f"CATALOGUE:\n{RESOURCE_CATALOGUE}"
    )

    try:
        from openai import OpenAI as _OAI2
        _fr_client = _OAI2()
        resp = _fr_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Question: {question}{audience_hint}"},
            ],
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[find_resource] LLM call failed: {e}")
        return {
            "answer":    "I wasn't able to search the resource library right now. Please try again.",
            "sources":   [],
            "retrieved": [],
            "meta":      {"route": "find_resource", "traffic_light": "red"},
        }

    # Split prose from SOURCES_JSON
    answer_text = raw
    sources_list: List[str] = []

    if "SOURCES_JSON:" in raw:
        parts = raw.split("SOURCES_JSON:", 1)
        answer_text = parts[0].strip()
        try:
            import re as _re
            json_str = _re.sub(
                r"^```(?:json)?\s*|\s*```$", "", parts[1].strip(), flags=_re.MULTILINE
            ).strip()
            # Normalise keys: resource_id and reason may be unquoted
            json_str = _re.sub(r'(\w+)(?=\s*:)', r'"\1"', json_str)
            for item in json.loads(json_str):
                rid = (item.get("resource_id") or "").strip()
                if not rid or rid not in MANIFEST:
                    continue
                me = MANIFEST[rid]
                pub   = me.get("publisher") or ""
                year  = me.get("year")
                title = me.get("title") or rid
                url   = me.get("url") or ""
                reason = (item.get("reason") or "").strip()
                left  = f"{pub} ({year})" if pub and year else (pub or str(year))
                label = f"{left} \u2013 {title}" if left else title
                if url:    label += f" \u2014 {url}"
                if reason: label += f"\n  \u2192 {reason}"
                sources_list.append(f"- {label}")
        except Exception:
            pass

    return {
        "answer":    answer_text,
        "sources":   sources_list,
        "retrieved": [],
        "meta":      {"route": "find_resource", "traffic_light": "green"},
    }


def rag_query(
    question: str,
    index: Dict[str, Any],
    k: int = 8,
    response_mode: Optional[str] = None,
    expertise: str = "general",
    audience: Optional[str] = None,
    institution_type: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    asset_class: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    The main function that app.py calls to answer a user question.

    Full pipeline (10 steps):
      1. Route   — classify the question (benchmarks / guidance / qualification
                   / implementation / resources / etc.) to choose the right
                   answer strategy
      2. k       — decide how many document chunks to retrieve based on
                   question complexity
      3. Embed   — convert the question to a vector (once, reused for both
                   retrieval passes)
      4. Retrieve — find the k most relevant chunks using cosine similarity
                    plus the boost/penalty logic in _diverse_top_k
      5. Coverage — check whether the retrieved chunks are actually relevant
                    enough to attempt a specific answer
      6. Expand  — if coverage is thin, broaden the search
      7. Threshold — if the question asks about 2X benchmarks, look up the
                     exact numbers from the Excel spreadsheet
      8. Percentages — extract any user-provided percentages from the question
                       for use in eligibility comparisons
      9. Benchmarks — if this is a pure threshold lookup, return the answer
                      immediately without calling the main LLM
     10. Narrate — send the question, retrieved chunks, and context to the LLM
                   to generate the final answer and source list

    Returns a dict with: answer (str), sources (list), retrieved (list), meta (dict).
    """
    meta            = index["meta"]
    vectors_norm    = _ensure_normalized_vectors(index)
    pool_multiplier = int(os.getenv("RAG_POOL_MULTIPLIER", "6"))

    # Step 1: Classify the question into a route.
    # The route determines which answer strategy to use — e.g. "benchmarks"
    # triggers a spreadsheet lookup, "qualification" triggers a threshold
    # comparison, "resources" boosts practitioner tool documents, etc.
    route = _route_question_llm(
        question,
        response_mode=response_mode,
        institution_type=institution_type,
    )

    # Step 1.5: find_resource — skip retrieval, call LLM on full catalogue.
    if route == "find_resource":
        return _find_resource_llm(
            question=question,
            institution_type=institution_type,
        )

    # Step 2: Decide how many document chunks to retrieve.
    # Complex or multi-part questions retrieve more chunks for broader context.
    # Simple factual questions retrieve fewer to stay focused.
    caller_default = 8
    if k != caller_default:
        k_max = int(os.getenv("RAG_K_MAX", "12"))
        k = max(1, min(k, k_max))
    else:
        k = _estimate_k(question, route)

    # Step 3: Convert the question to a vector (embedding).
    # Done once here and reused for both retrieval passes so we only pay
    # the API cost once.
    qv   = embed_query(question)
    sims = cosine_sims(vectors_norm, qv)

    # Step 4: Find the most relevant document chunks using cosine similarity.
    # Applies boosts and penalties for source quality, audience, and topic tags.
    picked_idxs = _diverse_top_k(
        sims=sims,
        meta=meta,
        k=k,
        institution_type=institution_type,
        pool_multiplier=pool_multiplier,
        max_per_resource=int(os.getenv("RAG_MAX_PER_RESOURCE", "2")),
        route=route,
        question=question,
    )
    top_chunks = [meta[i] for i in picked_idxs]

    # Step 5: Check whether the retrieved chunks are actually relevant.
    # If the best match scores are too low, we flag thin coverage which
    # triggers a broader search in step 6 and a more cautious answer.
    pool_size   = max(k * pool_multiplier, 30)
    cand_idxs   = np.argsort(-sims)[:pool_size].tolist()
    sims_sorted = sorted([float(sims[i]) for i in cand_idxs], reverse=True)
    coverage, stats = _coverage_flags(sims_sorted, question)

    # Step 6: If coverage is thin, broaden the search.
    # Retrieves more chunks and re-ranks to try to find better matches.
    # If coverage is still thin after expansion, the answer will carry a
    # yellow or red confidence indicator.
    adaptive_enabled = str(os.getenv("RAG_ADAPTIVE_EXPANSION", "1")).strip() == "1"
    k_max_env        = int(os.getenv("RAG_K_MAX", "12"))

    if adaptive_enabled and coverage in ("general", "thin") and k < k_max_env:
        k_expanded  = k_max_env
        picked_idxs = _diverse_top_k(
            sims=sims,
            meta=meta,
            k=k_expanded,
            institution_type=institution_type,
            pool_multiplier=pool_multiplier,
            max_per_resource=int(os.getenv("RAG_MAX_PER_RESOURCE", "2")),
            route=route,
            question=question,
        )
        top_chunks = [meta[i] for i in picked_idxs]

        pool_size_exp   = max(k_expanded * pool_multiplier, 30)
        cand_idxs_exp   = np.argsort(-sims)[:pool_size_exp].tolist()
        sims_sorted_exp = sorted([float(sims[i]) for i in cand_idxs_exp], reverse=True)
        coverage_exp, stats_exp = _coverage_flags(sims_sorted_exp, question)

        if coverage == "thin" and coverage_exp in ("general", "ok"):
            coverage, stats = coverage_exp, stats_exp
        elif coverage == "general" and coverage_exp == "ok":
            coverage, stats = coverage_exp, stats_exp

        stats["adaptive_expansion"] = True
        stats["k_initial"]          = k
        stats["k_expanded"]         = k_expanded
    else:
        stats["adaptive_expansion"] = False

    # Step 7: Look up 2X benchmark thresholds if the question needs them.
    # Reads from the Excel spreadsheet (data/Benchmark Thresholds-2X...).
    # Returns None if no threshold data is needed for this question.
    threshold_context = ""
    threshold_hit     = None

    try:
        t_score = score_threshold_intent(question)

        should_try = (t_score >= 3) or (route in {"qualification", "benchmarks", "resources"})
        if should_try:
            threshold_hit = lookup_threshold(question)
            if threshold_hit:
                threshold_rows, country_key, industry_key = threshold_hit
                summary = summarise_thresholds(threshold_rows)
                bullets = format_threshold_bullets(summary)
                threshold_context = (
                    "BENCHMARK THRESHOLDS (deterministic lookup from benchmarks.xlsx):\n"
                    f"- Detected country: {country_key}\n"
                    f"- Detected sector: {industry_key or 'Overall / not detected'}\n"
                    f"{bullets}"
                )
            elif t_score >= 3:
                threshold_context = (
                    "BENCHMARK THRESHOLDS: Could not detect country/sector from question.\n"
                    "Ask the user for Country and Sector (or 'Overall') if benchmarks matter here.\n"
                )
    except Exception:
        threshold_context = ""

    # Step 8: Extract any percentages from the user's question.
    # e.g. "we have 35% women in leadership" — used to compare against
    # the thresholds in step 9 or in the qualification route.
    provided_pcts = _extract_percentages_with_context(question)

    # Step 9: Pure threshold lookup -- no qualification/alignment intent.
    # Returns a formatted string directly; no LLM call needed.
    if route == "benchmarks" and not re.search(
        r"\b(qualif(y|ies|ication)?|aligned|alignment|eligible|eligibility)\b",
        question.lower(),
    ):

        # No threshold hit: context-aware clarification request
        if not threshold_hit:
            try:
                _det_country, _det_industry = detect_country_and_industry_from_text(question)
            except Exception:
                _det_country = _det_industry = None

            if _det_country and _det_industry:
                _msg = (
                    f"I captured **{_det_industry}** in **{_det_country}**, "
                    "but couldn't match that combination to the thresholds database. "
                    "Try using the exact sector label from "
                    "https://2xchallenge.org/2xcriteria, or paste it here."
                )
            elif _det_country:
                _msg = (
                    f"I captured the country (**{_det_country}**), but not the sector. "
                    "What sector should I use "
                    "(e.g., agriculture, manufacturing, finance and insurance)?"
                )
            elif _det_industry:
                _msg = (
                    f"I captured the sector (**{_det_industry}**), but not the country. "
                    "Which country should I use?"
                )
            else:
                _msg = (
                    "I couldn't identify a country or sector from your question. "
                    "Please tell me the country and sector "
                    "(or say \u2018Overall\u2019 for the sector) and I'll look up the thresholds."
                )
            return {
                "answer":    _msg,
                "sources":   [],
                "retrieved": [],
                "meta":      {"route": "benchmarks", "traffic_light": "red"},
            }

        # Threshold hit: render directly -- no LLM needed
        threshold_rows, country_key, industry_key = threshold_hit
        summary = summarise_thresholds(threshold_rows)
        answer  = format_threshold_bullets(summary)
        if not answer:
            answer = "No benchmark values found for that country/sector combination."
        else:
            answer += "\n\nShare the investee\u2019s current percentages and I\u2019ll compare them against these thresholds."
        return {
            "answer":    answer,
            "sources":   [],
            "retrieved": [],
            "meta":      {"route": "benchmarks", "traffic_light": "green"},
        }

    # Step 10: Generate the final answer.
    # Builds a "brief" (structured context block with retrieved chunks,
    # threshold data, and writing instructions) then passes it to the LLM
    # to produce the natural-language answer and source recommendations.
    result = answer_question(
        question=question,
        top_chunks=top_chunks,
        route=route,
        expertise=expertise,
        audience=audience,
        institution_type=institution_type,
        jurisdiction=jurisdiction,
        asset_class=asset_class,
        history=history,
        coverage=coverage,
        threshold_context=threshold_context,
        retrieval_stats=stats,
        threshold_hit=threshold_hit,
        provided_pcts=provided_pcts,
    )

    result.setdefault("meta", {})
    result["meta"].update({"retrieval_stats": stats, "k": k})
    return result
