#!/usr/bin/env python3
"""
update_library.py  —  2X Resource Library Update Tool
======================================================
Run this script whenever new PDFs have been added to the data/Pdfs/ folder.
It handles the full update process: extracting text from the PDFs, building
the search index, and updating the document library (manifest).

Usage:
    python update_library.py

No command-line arguments needed — just run it and follow the on-screen menu.

Menu options:
    1   Full update Part A  — process new PDFs and extract their metadata.
                              Writes metadata_review.csv for you to check.
    1b  Full update Part B  — run after reviewing metadata_review.csv.
                              Saves the metadata and renames the PDFs.
    2   Re-enrich           — update quality tiers and topic tags for all docs.
                              Writes enrichment_review.csv for you to check.
    3   Apply edits         — apply any changes you made to enrichment_review.csv.
    4   Exit

Everything is logged to update_library.log so you can email it if something goes wrong.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import pickle
import re
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths  (all relative to wherever this script lives)
# ---------------------------------------------------------------------------

ROOT          = Path(__file__).resolve().parent
DATA_DIR      = ROOT / "data"
PDF_DIR       = DATA_DIR / "Pdfs"
CHUNKS_PATH   = DATA_DIR / "chunks.jsonl"
INDEX_PATH    = DATA_DIR / "index.pkl"
MANIFEST_PATH = DATA_DIR / "resources_manifest_llm.json"
ENRICH_CSV         = ROOT / "enrichment_review.csv"
METADATA_REVIEW_CSV = ROOT / "metadata_review.csv"
LOG_PATH      = ROOT / "update_library.log"

# ---------------------------------------------------------------------------
# Logging — both to screen and to a log file colleagues can email
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def _banner(text: str) -> None:
    log.info("")
    log.info("─" * 60)
    log.info(f"  {text}")
    log.info("─" * 60)


def _ok(msg: str) -> None:
    log.info(f"  ✓  {msg}")


def _warn(msg: str) -> None:
    log.warning(f"  ⚠  {msg}")


def _err(msg: str) -> None:
    log.error(f"  ✗  {msg}")


def _abort(msg: str) -> None:
    _err(msg)
    log.error("")
    log.error("  The update was stopped.  No files were changed.")
    log.error(f"  If you need help, email the file: {LOG_PATH}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

def _check_env() -> None:
    """
    Checks that the OPENAI_API_KEY environment variable is set before we
    start any work. If it is missing, we stop immediately with a plain-English
    message rather than failing halfway through and leaving files in a bad state.

    The API key lives in the .env file in the root folder.
    """
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        _abort(
            "OPENAI_API_KEY is not set.\n"
            "  Open the .env file in the Resource Bot folder and make sure the line\n"
            "  OPENAI_API_KEY=sk-...  is present and correct."
        )


def _check_paths() -> None:
    if not PDF_DIR.exists():
        _abort(
            f"PDF folder not found: {PDF_DIR}\n"
            "  Make sure the 'Pdfs' folder exists inside the 'data' folder."
        )
    pdfs = list(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        _abort(
            f"No PDF files found in {PDF_DIR}\n"
            "  Drop the new PDFs into data/Pdfs/ and run this again."
        )
    _ok(f"Found {len(pdfs)} PDF(s) in {PDF_DIR}")


# ---------------------------------------------------------------------------
# Step 1 — Rename PDFs to clean filenames
# ---------------------------------------------------------------------------

_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _slugify(s: str) -> str:
    s = (s or "").strip().replace("–", "-").replace("—", "-")
    s = re.sub(r"[^\w\s\-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip().replace(" ", "-")
    return re.sub(r"-{2,}", "-", s)


def _safe_stem(title: str, publisher: str, year: Any) -> Optional[str]:
    parts = []
    if isinstance(year, int):
        parts.append(str(year))
    if publisher:
        parts.append(publisher)
    if title:
        parts.append(title)
    if not parts:
        return None
    stem = _slugify("_".join(parts))[:180].rstrip("-_")
    if stem.upper() in _WINDOWS_RESERVED:
        stem += "_file"
    return stem or None


def step_rename_pdfs(manifest: dict) -> int:
    """
    Renames PDFs in data/Pdfs/ to a clean, consistent format:
        year_Publisher_Title.pdf
        e.g. 2023_IFC_Gender-Smart-Investing-Guide.pdf

    Only renames files that are already in the manifest (i.e. whose metadata
    has been extracted). New PDFs that have not yet been through Part A are
    left untouched — they get renamed in Part B after their metadata is known.

    Returns the number of files that were actually renamed.
    """
    _banner("Step 1 of 4 — Renaming PDFs")

    # Build a map: current filename (normalised) → manifest entry
    norm_to_meta: Dict[str, Dict] = {}
    for rid, meta in manifest.items():
        # Index by each known filename
        for fn in (meta.get("filenames") or []):
            norm_to_meta[fn.strip().lower()] = meta
        # Also index by resource_id stem (catches newly added docs where
        # filenames list may not yet be populated)
        norm_to_meta[(rid + ".pdf").lower()] = meta

    renamed = 0
    used: set = set()

    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        meta = norm_to_meta.get(pdf.name.strip().lower())
        if not meta:
            continue  # not in manifest yet — will be picked up by manifest build

        stem = _safe_stem(
            meta.get("title", ""),
            meta.get("publisher", ""),
            meta.get("year"),
        )
        if not stem:
            continue

        target = PDF_DIR / (stem + ".pdf")

        # avoid collisions
        base, i = stem, 1
        while target.exists() and target != pdf or target.name.lower() in used:
            target = PDF_DIR / f"{base}_{i}.pdf"
            i += 1

        if target == pdf:
            continue  # already has the right name

        try:
            shutil.move(str(pdf), str(target))
            used.add(target.name.lower())
            log.info(f"    {pdf.name}  →  {target.name}")
            renamed += 1
        except Exception as exc:
            _warn(f"Could not rename {pdf.name}: {exc}")

    _ok(f"{renamed} file(s) renamed, rest unchanged.")
    return renamed


# ---------------------------------------------------------------------------
# Step 2 — Chunk new PDFs and add to chunks.jsonl
# ---------------------------------------------------------------------------

def _extract_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    from pypdf import PdfReader
    try:
        reader = PdfReader(str(pdf_path), strict=False)
        # Some PDFs are password-protected. We try an empty password first.
        # If that doesn't work, we skip the file and warn the user.
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                _warn(f"Encrypted PDF skipped: {pdf_path.name}")
                return []
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = " ".join((page.extract_text() or "").split())
            pages.append((i, text))
        return pages
    except Exception as exc:
        _warn(f"Could not read {pdf_path.name}: {exc}")
        return []


def _chunk_text(text: str, chunk_chars: int = 1400, overlap: int = 200) -> List[str]:
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, end)
    return chunks


def step_chunk_new_pdfs() -> List[Dict]:
    """
    Reads new PDFs and splits their text into small overlapping chunks
    (roughly 1,400 characters each with 200-character overlap). These chunks
    are what gets searched when a user asks a question.

    Only processes PDFs that are not already in chunks.jsonl — so running
    this multiple times is safe and will not duplicate existing documents.

    Returns a list of chunk dictionaries for the newly processed PDFs.
    Each dict contains the chunk text, which document it came from, and
    which page it appeared on.
    """
    _banner("Step 2 of 4 — Reading and chunking new PDFs")

    # Find which resource_ids are already chunked
    existing_rids: set = set()
    if CHUNKS_PATH.exists():
        with CHUNKS_PATH.open(encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    rid = obj.get("resource_id", "")
                    if rid:
                        existing_rids.add(rid)
                except Exception:
                    pass

    new_pdfs = [
        p for p in sorted(PDF_DIR.glob("*.pdf"))
        if p.stem not in existing_rids
    ]

    if not new_pdfs:
        _ok("No new PDFs to chunk — all already in the index.")
        return []

    log.info(f"  Processing {len(new_pdfs)} new PDF(s)...")
    new_chunks: List[Dict] = []

    for pdf in new_pdfs:
        pages = _extract_pages(pdf)
        if not pages:
            _warn(f"No text extracted from {pdf.name} — skipping.")
            continue

        rid = pdf.stem
        n_chunks = 0
        for page_num, page_text in pages:
            if len(page_text.strip()) < 40:
                continue
            for j, ctext in enumerate(_chunk_text(page_text), start=1):
                new_chunks.append({
                    "chunk_id":   f"{rid}__p{page_num:04d}__c{j:03d}",
                    "resource_id": rid,
                    "title":      pdf.stem.replace("_", " "),
                    "local_path": str(pdf),
                    "page_start": page_num,
                    "page_end":   page_num,
                    "text":       ctext,
                })
                n_chunks += 1

        log.info(f"    {pdf.name}: {n_chunks} chunks")

    # Append new chunks to existing jsonl
    if new_chunks:
        with CHUNKS_PATH.open("a", encoding="utf-8") as f:
            for c in new_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        _ok(f"{len(new_chunks)} new chunks added to {CHUNKS_PATH.name}")
    else:
        _ok("No new chunks produced.")

    return new_chunks


# ---------------------------------------------------------------------------
# Step 3 — Embed new chunks and rebuild index
# ---------------------------------------------------------------------------

def step_embed_and_index(new_chunks: List[Dict]) -> None:
    """
    Converts each new text chunk into a vector (a list of numbers that
    represents its meaning) using OpenAI's embedding model. These vectors
    are then merged into the existing search index (index.pkl) so the
    assistant can find relevant chunks when answering questions.

    Only embeds the new chunks passed in — the existing index is loaded
    and the new vectors are appended to it. The result is saved atomically
    (written to a temp file first, then renamed) to prevent corruption if
    something goes wrong mid-write.
    """
    _banner("Step 3 of 4 — Building search index")

    if not new_chunks:
        _ok("No new chunks to embed — index unchanged.")
        return

    try:
        import numpy as np
        from openai import OpenAI
    except ImportError as exc:
        _abort(f"Missing required package: {exc}\nRun: pip install -r requirements.txt")

    client = OpenAI()
    model  = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    # Embed new chunks in batches
    texts    = [c["text"] for c in new_chunks]
    vectors  = []
    batch_sz = 64

    log.info(f"  Embedding {len(texts)} chunks in batches of {batch_sz}...")
    for i in range(0, len(texts), batch_sz):
        batch = texts[i : i + batch_sz]
        try:
            resp = client.embeddings.create(model=model, input=batch)
            vectors.extend([
                np.array(d.embedding, dtype=np.float32) for d in resp.data
            ])
            log.info(f"    Batch {i // batch_sz + 1}/{(len(texts) - 1) // batch_sz + 1} done")
        except Exception as exc:
            _abort(
                f"OpenAI embedding request failed: {exc}\n"
                "  Check your internet connection and API key, then try again."
            )

    new_vecs = np.vstack(vectors)

    # Load existing index and merge
    if INDEX_PATH.exists():
        try:
            with INDEX_PATH.open("rb") as f:
                existing = pickle.load(f)
            old_vecs = existing["vectors"]
            old_meta = existing["meta"]
            merged_vecs = np.vstack([old_vecs, new_vecs])
            merged_meta = old_meta + new_chunks
            _ok(f"Merged with existing index ({len(old_meta)} + {len(new_chunks)} = {len(merged_meta)} chunks)")
        except Exception as exc:
            _warn(f"Could not load existing index ({exc}) — rebuilding from scratch.")
            merged_vecs = new_vecs
            merged_meta = new_chunks
    else:
        merged_vecs = new_vecs
        merged_meta = new_chunks
        _ok("Creating new index from scratch.")

    # Atomic write
    tmp = INDEX_PATH.with_suffix(".pkl.tmp")
    try:
        with tmp.open("wb") as f:
            pickle.dump({"vectors": merged_vecs, "meta": merged_meta}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()
        tmp.replace(INDEX_PATH)
        _ok(f"Index saved: {INDEX_PATH.name}  ({INDEX_PATH.stat().st_size // 1024} KB)")
    except Exception as exc:
        _abort(
            f"Could not save index file: {exc}\n"
            "  Make sure the data/ folder is not open in another program."
        )


# ---------------------------------------------------------------------------
# Step 4 — Build / update manifest for new resources
# ---------------------------------------------------------------------------

_MIN_YEAR, _MAX_YEAR = 1990, 2027


def _clamp_year(y: Any) -> Optional[int]:
    try:
        y = int(y)
        return y if _MIN_YEAR <= y <= _MAX_YEAR else None
    except Exception:
        return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _collect_evidence(rids: List[str]) -> Dict[str, Dict]:
    """Read chunks.jsonl and collect early-page text for the given resource_ids."""
    evidence: Dict[str, Dict] = {
        rid: {"early_text": "", "title_hints": set(), "filenames": set()}
        for rid in rids
    }
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rid = obj.get("resource_id", "")
            if rid not in evidence:
                continue
            fn = (obj.get("local_path") or "").replace("\\", "/").split("/")[-1]
            if fn:
                evidence[rid]["filenames"].add(fn)
            t = (obj.get("title") or "").strip()
            if t:
                evidence[rid]["title_hints"].add(t)
            page = obj.get("page_start", 999)
            try:
                if int(page) <= 3:
                    evidence[rid]["early_text"] += "\n" + (obj.get("text") or "")
            except Exception:
                pass
    for info in evidence.values():
        info["early_text"] = info["early_text"][:14000]
    return evidence


def _call_manifest_llm(client: Any, rid: str, info: Dict) -> Dict:
    fn = sorted(info["filenames"])[0] if info["filenames"] else rid
    title_hint = Counter(info["title_hints"]).most_common(1)[0][0] if info["title_hints"] else "(none)"

    system = (
        "You extract citation metadata for investor-grade references.\n"
        "Return only what is supported by evidence in the text.\n"
        "If unsure, return null. Do not guess. Return JSON only."
    )
    user = (
        f"FILENAME: {fn}\n"
        f"TITLE HINT: {title_hint}\n\n"
        f"EARLY PAGES TEXT:\n{info['early_text']}\n\n"
        "Return a JSON object with exactly these keys:\n"
        '{"title": string|null, "publisher": string|null, "year": integer|null, '
        '"url": string|null, "confidence": "high"|"medium"|"low", '
        '"evidence": {"title_snippet": string|null, "publisher_snippet": string|null, '
        '"year_snippet": string|null, "url_snippet": string|null}, "notes": string|null}'
    )

    resp = client.chat.completions.create(
        model=os.getenv("MANIFEST_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    data = json.loads(resp.choices[0].message.content)

    # Normalise
    data.setdefault("title", None)
    data.setdefault("publisher", None)
    data.setdefault("year", None)
    data.setdefault("url", None)
    data.setdefault("confidence", "low")
    data.setdefault("notes", None)
    ev = data.setdefault("evidence", {})
    ev.setdefault("title_snippet", None)
    ev.setdefault("publisher_snippet", None)
    ev.setdefault("year_snippet", None)
    ev.setdefault("url_snippet", None)

    data["year"] = _clamp_year(data.get("year"))
    for field in ("title", "publisher", "url"):
        if isinstance(data.get(field), str):
            data[field] = _norm(data[field]) or None
    if data.get("confidence") not in ("high", "medium", "low"):
        data["confidence"] = "low"

    return data


def step_build_manifest_csv(new_chunks: List[Dict]) -> int:
    """
    Part A of the manifest update process.

    For each new PDF, calls the AI to extract the document title, publisher,
    year, and URL from the first few pages of the document. The results are
    written to metadata_review.csv so a human can verify them before they
    are saved permanently.

    The manifest itself is NOT updated yet — that happens in Part B
    (step_apply_metadata_review) after you have reviewed and corrected the CSV.

    Returns the number of new resources found (0 means all PDFs are already
    in the library and nothing needs to be done).
    """
    _banner("Step 3 of 4 — Extracting metadata for new resources")

    existing_manifest = {}
    if MANIFEST_PATH.exists():
        try:
            existing_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            _warn(f"Could not read existing manifest ({exc}) — will rebuild.")

    new_rids = sorted({
        c["resource_id"] for c in new_chunks
        if c.get("resource_id") and c["resource_id"] not in existing_manifest
    })

    if not new_rids:
        _ok("No new resources found — manifest already up to date.")
        return 0

    log.info(f"  Extracting metadata for {len(new_rids)} new resource(s)...")

    try:
        from openai import OpenAI
        client = OpenAI()
    except ImportError as exc:
        _abort(f"Missing required package: {exc}")

    evidence = _collect_evidence(new_rids)
    rows = []

    for i, rid in enumerate(new_rids, 1):
        info = evidence[rid]
        log.info(f"  [{i}/{len(new_rids)}] {rid[:60]}")
        try:
            row = _call_manifest_llm(client, rid, info)
        except Exception as exc:
            _warn(f"  LLM call failed for {rid}: {exc}")
            row = {
                "title": None, "publisher": None, "year": None, "url": None,
                "confidence": "low",
                "evidence": {"title_snippet": None, "publisher_snippet": None,
                             "year_snippet": None, "url_snippet": None},
                "notes": f"LLM failed — please fill in manually: {exc}",
            }

        rows.append({
            "resource_id":       rid,
            "original_filename": sorted(info["filenames"])[0] if info["filenames"] else rid,
            # ── Fields to review / edit ──────────────────────────────────────
            "title":             row.get("title") or "",
            "publisher":         row.get("publisher") or "",
            "year":              row.get("year") or "",
            "url":               row.get("url") or "",
            "audience":          "general",
            # ── Evidence snippets (read-only context for reviewer) ───────────
            "title_snippet":     (row.get("evidence") or {}).get("title_snippet") or "",
            "publisher_snippet": (row.get("evidence") or {}).get("publisher_snippet") or "",
            "year_snippet":      (row.get("evidence") or {}).get("year_snippet") or "",
            # ── Status flags ─────────────────────────────────────────────────
            "confidence":        row.get("confidence") or "low",
            "needs_review":      (
                not row.get("title") or not row.get("publisher") or
                not row.get("year") or row.get("confidence") == "low"
            ),
            "notes":             row.get("notes") or "",
        })
        time.sleep(0.25)

    # Write review CSV
    fieldnames = list(rows[0].keys())
    with METADATA_REVIEW_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    needs = sum(1 for r in rows if r["needs_review"])
    _ok(f"Metadata extracted for {len(rows)} resource(s).")
    if needs:
        _warn(f"{needs} resource(s) flagged for review (low confidence or missing fields).")

    return len(rows)


def step_apply_metadata_review() -> None:
    """
    Part B of the manifest update process.

    Reads back the metadata_review.csv that you have reviewed and corrected,
    and saves the new document entries into resources_manifest_llm.json.
    This is the permanent record the assistant uses to cite sources.

    Entries already in the manifest are never overwritten — only genuinely
    new documents (from your latest batch of PDFs) are added.
    """
    _banner("Step 4 of 4 — Saving reviewed metadata to manifest")

    if not METADATA_REVIEW_CSV.exists():
        _abort(
            "metadata_review.csv not found at " + str(METADATA_REVIEW_CSV) + "\n"
            "  Run option 1 again to regenerate it."
        )

    existing_manifest = {}
    if MANIFEST_PATH.exists():
        try:
            existing_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            _warn(f"Could not read existing manifest ({exc}) — will create fresh.")

    new_entries = {}
    skipped = []

    with METADATA_REVIEW_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = (row.get("resource_id") or "").strip()
            if not rid:
                continue
            if rid in existing_manifest:
                skipped.append(rid)
                continue  # already in manifest — don't overwrite

            title = (row.get("title") or "").strip()
            if not title:
                _warn(f"  {rid}: title is blank — using filename as fallback.")
                title = (row.get("original_filename") or rid)

            year_raw = (row.get("year") or "").strip()
            year = _clamp_year(year_raw) if year_raw else None

            new_entries[rid] = {
                "title":        title,
                "publisher":    (row.get("publisher") or "").strip() or None,
                "year":         year,
                "url":          (row.get("url") or "").strip() or None,
                "confidence":   row.get("confidence") or "low",
                "evidence": {
                    "title_snippet":     row.get("title_snippet") or None,
                    "publisher_snippet": row.get("publisher_snippet") or None,
                    "year_snippet":      row.get("year_snippet") or None,
                    "url_snippet":       None,
                },
                "notes":        row.get("notes") or None,
                "resource_id":  rid,
                "filenames":    [row.get("original_filename")] if row.get("original_filename") else [],
                "needs_review": not title or not (row.get("publisher") or "").strip() or not year,
                "audience":     (row.get("audience") or "general").strip(),
            }

    existing_manifest.update(new_entries)

    # Atomic write
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(existing_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()
    tmp.replace(MANIFEST_PATH)

    _ok(f"Added {len(new_entries)} new resource(s) to manifest.")
    if skipped:
        log.info(f"  Skipped {len(skipped)} already-existing entries (not overwritten).")


# ---------------------------------------------------------------------------
# Option 2 — Re-enrich existing resources (tiers / tags / use_case)
# ---------------------------------------------------------------------------

TOPIC_TAGS = [
    "due_diligence", "monitoring_reporting", "investor_reporting", "blended_finance",
    "benchmarks", "evidence_collection", "fund_structure", "transaction_structuring",
    "technical_assistance", "gender_bonds", "returns_data", "policy_context",
    "certification_assessment", "supply_chain_procurement", "consumer_markets",
    "workforce_practices", "entrepreneurship_ownership",
]

DEPTH_TYPES = [
    "framework_reference", "how_to_guide", "case_study", "data_report",
    "policy_brief", "toolkit", "standards_guidance",
]

_ENRICH_SYSTEM = f"""You are a research librarian specialising in gender lens investing (GLI).

Given metadata and early-page text from a document, return a JSON object with EXACTLY these fields:

{{
  "use_case": "<1-2 sentences: what is this document best used for?>",
  "depth_type": "<one of: {', '.join(DEPTH_TYPES)}>",
  "topic_tags": ["<2-6 tags from vocabulary below>"],
  "quality_tier": "<tier_1 | tier_2 | tier_3>",
  "best_for": "<one sentence: Best for [audience] trying to [task]>"
}}

QUALITY TIER:
  tier_1 — Go-to document for its topic. Comprehensive, credible, widely used by practitioners.
           Can be tier_1 within its niche (e.g. ILPA DDQ is tier_1 for LP due diligence).
           Expect ~30% of documents.
  tier_2 — Useful supplement. Solid but narrower or less comprehensive than tier_1.
           Expect ~55% of documents.
  tier_3 — Background only. Short overviews, dated reports, thin one-pagers.
           Expect ~15% of documents.

TOPIC TAGS (pick 2-6 that genuinely apply):
{chr(10).join(f'  {t}' for t in TOPIC_TAGS)}

IMPORTANT: Return ONLY valid JSON. Use the full tier range — do not default everything to tier_2.
gender_bonds = ONLY labelled bond instruments (GSS bonds, gender-themed fixed income). NOT general gender finance.
"""


def step_enrich(force_all: bool = False) -> None:
    """
    Adds enrichment metadata to manifest entries: quality tier, topic tags,
    a use-case description, and a "best for" sentence.

    Quality tiers:
        tier_1  Most authoritative / go-to reference for its topic (~30% of docs)
        tier_2  Useful supplement — solid but narrower (~55% of docs)
        tier_3  Background only — short overviews, dated or thin content (~15% of docs)

    Topic tags help the assistant surface the right documents for each question.
    For example, a question about LP reporting will boost documents tagged
    "investor_reporting".

    If force_all is True, re-enriches everything including already-enriched docs.
    Otherwise only enriches documents that have not been enriched yet.
    Progress is saved after every document so it is safe to interrupt and restart.
    """
    _banner("Re-enriching resources (tiers / tags / use_case)")

    if not MANIFEST_PATH.exists():
        _abort("No manifest found. Run option 1 first.")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    to_enrich = [
        rid for rid, v in manifest.items()
        if force_all or "enriched_at" not in v
    ]

    if not to_enrich:
        _ok("All resources already enriched. Use 'force re-enrich all' to redo them.")
        return

    log.info(f"  Enriching {len(to_enrich)} resource(s)...")

    try:
        from openai import OpenAI
        import pypdf
        client = OpenAI()
    except ImportError as exc:
        _abort(f"Missing required package: {exc}")

    success = failed = 0

    for i, rid in enumerate(to_enrich, 1):
        meta = manifest[rid]
        log.info(f"  [{i}/{len(to_enrich)}] {rid[:60]}")

        # Extract PDF text
        fns = meta.get("filenames") or []
        pdf_text = ""
        for fn in fns:
            pdf_path = PDF_DIR / fn
            if pdf_path.exists():
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(str(pdf_path))
                    chunks = []
                    for page in reader.pages[:5]:
                        t = page.extract_text() or ""
                        if t.strip():
                            chunks.append(t.strip()[:1500])
                    pdf_text = "\n\n".join(chunks)
                    break
                except Exception:
                    pass

        user_msg = (
            f"Title: {meta.get('title')}\n"
            f"Publisher: {meta.get('publisher')}\n"
            f"Year: {meta.get('year')}\n\n"
            f"Text sample:\n{pdf_text[:5000] or '(no text available)'}\n\n"
            "Return your assessment as a JSON object."
        )

        try:
            resp = client.chat.completions.create(
                model=os.getenv("MANIFEST_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": _ENRICH_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
            )
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "",
                         resp.choices[0].message.content.strip(), flags=re.MULTILINE)
            enrichment = json.loads(raw)

            # Validate
            if enrichment.get("depth_type") not in DEPTH_TYPES:
                enrichment.pop("depth_type", None)
            tags = [t for t in (enrichment.get("topic_tags") or []) if t in TOPIC_TAGS]
            if tags:
                enrichment["topic_tags"] = tags
            if enrichment.get("quality_tier") not in ("tier_1", "tier_2", "tier_3"):
                enrichment.pop("quality_tier", None)

            enrichment["enriched_at"] = datetime.now(timezone.utc).isoformat()
            manifest[rid].update(enrichment)

            tier = enrichment.get("quality_tier", "?")
            tags_str = ", ".join(enrichment.get("topic_tags") or [])
            log.info(f"      tier={tier}  tags=[{tags_str}]")
            success += 1

        except Exception as exc:
            _warn(f"  Enrichment failed for {rid}: {exc}")
            failed += 1

        # Save after every doc so progress is preserved
        MANIFEST_PATH.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        time.sleep(1.0)

    _ok(f"Enrichment complete: {success} succeeded, {failed} failed.")

    # Write review CSV
    _write_enrich_csv(manifest)
    _ok(f"Review CSV written: {ENRICH_CSV.name}")
    log.info("")
    log.info("  Next step: open enrichment_review.csv, review tier/tag assignments,")
    log.info("  edit any that look wrong, then run option 3 to apply your changes.")


def _write_enrich_csv(manifest: dict) -> None:
    rows = []
    for rid, v in manifest.items():
        rows.append({
            "resource_id":   rid,
            "title":         v.get("title", ""),
            "publisher":     v.get("publisher", ""),
            "year":          v.get("year", ""),
            "quality_tier":  v.get("quality_tier", ""),
            "depth_type":    v.get("depth_type", ""),
            "topic_tags":    "; ".join(v.get("topic_tags") or []),
            "use_case":      v.get("use_case", ""),
            "best_for":      v.get("best_for", ""),
            "enriched_at":   v.get("enriched_at", ""),
            "enrichment_status": "enriched" if "enriched_at" in v else "missing",
        })
    with ENRICH_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Option 3 — Apply edits from enrichment_review.csv
# ---------------------------------------------------------------------------

def step_apply_review() -> None:
    """
    Reads back the enrichment_review.csv that you have edited and applies
    your changes to the manifest. Only the fields you actually changed are
    updated — everything else is left as-is.

    Valid values for quality_tier:  tier_1 / tier_2 / tier_3
    Topic tags must be from the approved vocabulary (see TOPIC_TAGS list above).
    Invalid values are silently skipped so a typo cannot corrupt the manifest.
    """
    _banner("Applying enrichment review edits")

    if not ENRICH_CSV.exists():
        _abort(f"enrichment_review.csv not found at {ENRICH_CSV}")
    if not MANIFEST_PATH.exists():
        _abort("No manifest found. Run option 1 first.")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    editable = {"quality_tier", "depth_type", "topic_tags", "use_case", "best_for"}
    changes = 0

    with ENRICH_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = (row.get("resource_id") or "").strip()
            if rid not in manifest:
                _warn(f"Unknown resource_id in CSV: {rid} — skipping")
                continue

            for field in editable:
                val = (row.get(field) or "").strip()
                if not val:
                    continue
                if field == "topic_tags":
                    tags = [t.strip() for t in val.split(";") if t.strip() in TOPIC_TAGS]
                    if tags != (manifest[rid].get("topic_tags") or []):
                        manifest[rid]["topic_tags"] = tags
                        changes += 1
                elif field == "quality_tier":
                    if val in ("tier_1", "tier_2", "tier_3") and val != manifest[rid].get("quality_tier"):
                        manifest[rid]["quality_tier"] = val
                        changes += 1
                elif field == "depth_type":
                    if val in DEPTH_TYPES and val != manifest[rid].get("depth_type"):
                        manifest[rid]["depth_type"] = val
                        changes += 1
                else:
                    if val != manifest[rid].get(field):
                        manifest[rid][field] = val
                        changes += 1

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _ok(f"Applied {changes} field update(s) to manifest.")
    log.info("  The app will use the updated manifest on its next restart.")


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def _menu() -> str:
    print()
    print("=" * 60)
    print("  2X Resource Library — Update Tool")
    print("=" * 60)
    print()
    print("  What would you like to do?")
    print()
    print("  1.  Full update — Part A  (new PDFs have been added to data/Pdfs/)")
    print("       Chunks → embeds → extracts metadata → writes metadata_review.csv")
    print()
    print("  1b. Full update — Part B  (after reviewing metadata_review.csv)")
    print("       Saves reviewed metadata → renames PDFs")
    print()
    print("  2.  Re-enrich existing resources")
    print("       Updates quality tiers, topic tags, and descriptions")
    print("       Produces enrichment_review.csv for you to check")
    print()
    print("  3.  Apply edits from enrichment_review.csv")
    print("       Use after editing the CSV from option 2")
    print()
    print("  4.  Exit")
    print()
    choice = input("  Enter choice (1 / 1b / 2 / 3 / 4): ").strip().lower()
    print()
    return choice


def main() -> None:
    # Load the .env file so OPENAI_API_KEY is available.
    # If python-dotenv is not installed we skip this quietly — the key
    # may still be set as a system environment variable.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    log.info(f"update_library.py started  {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    choice = _menu()

    if choice == "1":
        _check_env()
        _check_paths()

        # Load manifest for renaming step (may be empty on first run)
        manifest = {}
        if MANIFEST_PATH.exists():
            try:
                manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass

        new_chunks = step_chunk_new_pdfs()
        step_embed_and_index(new_chunks)
        n_new = step_build_manifest_csv(new_chunks)

        if n_new == 0:
            log.info("")
            log.info("  No new PDFs were found — nothing changed.")
            log.info("  To add resources, drop PDFs into data/Pdfs/ and run again.")
        else:
            log.info("")
            log.info("  ─" * 30)
            log.info(f"  Metadata extracted for {n_new} new resource(s).")
            log.info("")
            log.info("  ACTION REQUIRED before continuing:")
            log.info(f"  1. Open  metadata_review.csv  in Excel or Google Sheets.")
            log.info("  2. Check the title, publisher, year, and url columns.")
            log.info("     The 'title_snippet', 'publisher_snippet', 'year_snippet' columns")
            log.info("     show what the AI found in the document — use these to verify.")
            log.info("  3. Fix anything that looks wrong.  'needs_review = True' rows")
            log.info("     are the ones most likely to need attention.")
            log.info("  4. Save and close the file.")
            log.info("  5. Run this script again and choose option 1b to continue.")
            log.info("  ─" * 30)

    elif choice in ("1b", "1 b"):
        _check_env()

        if not METADATA_REVIEW_CSV.exists():
            _abort(
                "metadata_review.csv not found.\n"
                "  Run option 1 first to generate it."
            )

        step_apply_metadata_review()

        # Rename PDFs now that manifest has the correct titles
        manifest_updated = {}
        if MANIFEST_PATH.exists():
            try:
                manifest_updated = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        step_rename_pdfs(manifest_updated)

        log.info("")
        log.info("  ✓  Metadata saved and PDFs renamed.")
        log.info("  Recommended next steps:")
        log.info("    1. Run option 2 to assign quality tiers and topic tags")
        log.info("       to the new resources (improves search accuracy).")
        log.info("    2. Restart the app to pick up the changes.")

    elif choice == "2":
        _check_env()
        force = input("  Re-enrich ALL resources (including already-enriched ones)? [y/N]: ").strip().lower() == "y"
        step_enrich(force_all=force)

    elif choice == "3":
        step_apply_review()

    elif choice == "4":
        log.info("Exiting.")
        sys.exit(0)

    else:
        log.info("Invalid choice. Please run the script again and enter 1, 2, 3, or 4.")
        sys.exit(1)

    log.info("")
    log.info(f"  Log saved to: {LOG_PATH}")
    log.info("")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("\n  Cancelled by user.")
        sys.exit(0)
    except Exception as exc:
        _err(f"Unexpected error: {exc}")
        log.error(traceback.format_exc())
        log.error(f"  Full error saved to: {LOG_PATH}")
        log.error("  Please email that file if you need help.")
        sys.exit(1)
