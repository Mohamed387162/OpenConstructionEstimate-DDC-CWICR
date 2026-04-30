"""
Text pipeline — translate language-bound columns from source to target.

Strategy:
  1. SHA1-dedup all unique strings across the 11 text columns. Typical
     ratio: 55K work items × 11 cols → ~15-20K truly unique strings.
  2. Frequency-rank. Top-3000 by occurrence go to Claude (high quality
     with construction glossary as in-context vocabulary). The tail
     goes to Google Translate (cheap, lower quality, acceptable for
     rare terms).
  3. Cross-validation pass (optional): GPT re-translates Claude's output
     and disagreements are flagged for manual review.
  4. Persist per language pair in glossary/translations/<src>_<tgt>.json
     so reruns are free.
  5. Re-map the dataframe by hash join — translation is never repeated.

Designed to be a no-op when source language == target language, so the
orchestrator can call it unconditionally.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd


HERE = Path(__file__).resolve().parent
GLOSSARY_DIR = HERE / "glossary"
TRANSLATIONS_DIR = GLOSSARY_DIR / "translations"
SEED_GLOSSARY = GLOSSARY_DIR / "construction_glossary.json"

# Columns translated. Order doesn't matter for correctness, but matters
# for pretty-printing in case of partial-translation reports.
TEXT_COLS: tuple[str, ...] = (
    "rate_original_name",
    "rate_final_name",
    "rate_unit",
    "work_composition_text",
    "resource_name",
    "resource_unit",
    "department_name",
    "section_name",
    "subsection_name",
    "collection_name",
    "mass_name",
    "mass_unit",
    "labor_class",
    "labor_title",
    "operator_class",
    "machine_class3_name",
    "machine_class2_name",
    "abstract_resource_tech_group",
    "service_category",
    "service_type",
    "parameter_service_name",
    "parameter_service_unit",
    # Taxonomy enums — tiny vocabularies (3-7 unique values each) that
    # were leaking source-language labels (e.g. "Abteilung", "Abschnitt",
    # "Ressource") into every derived track. Translating them costs
    # cents and removes the most visible UI/Catalog gaps.
    "department_type",
    "section_type",
    "row_type",
    "category_type",
    # Labor breakdown — domain-critical join surface for resource
    # aggregation. 43 unique German values like "Lehrlingsgrad".
    "personnel_operator_grade",
    # CSI MasterFormat titles. The numeric `masterformat_section_code`
    # remains the cross-track join key; the textual descriptions are
    # localised so non-English readers see meaningful classifications.
    "masterformat_section_title",
    "masterformat_division",
    # Mixed unit phrases ("100 Stück, Stück") visible in catalog/SIMPLE
    # outputs. Translator already handles unit phrases via `mass_unit`.
    "price_abstract_resource_unit",
    # Composite price-abstract descriptors — `•`-separated lists and
    # `unit=tech_group` maps that aggregate the resource catalog. These
    # were leaking the source-language descriptions ("Geotextilien",
    # "Polyamid-Erosionsschutzmatte, Dicke") into every derived track.
    "price_abstract_resource_common_start",
    "price_abstract_resource_variable_parts",
    "price_abstract_resource_est_price_range",
    "price_abstract_resource_est_price_all_values",
    "price_abstract_resource_group_per_unit",
    "price_abstract_resource_common_start_per_unit",
    "price_abstract_resource_variable_parts_per_unit",
    "price_abstract_resource_est_price_range_per_unit",
    "price_abstract_resource_est_price_all_values_per_unit",
    # Region of origin for the prices ("Berlin, Germany") — single value
    # per track, but it should localise to the target language ("Berlin,
    # Germania" for Romanian, "ベルリン、ドイツ" for Japanese, etc.).
    "price_region",
)

# Number of high-frequency strings sent to the LLM.
# 81K unique strings × 11 languages × ~0.5s per string = days of wall time
# even at moderate concurrency. Cap LLM translation at the top-N most
# frequent strings (covers the vast majority of *occurrences*); the rare
# long tail of unique-once rate descriptions is left in the source
# language. The downstream embedding step still indexes them, just in
# the source language for that subset of text.
# Override with env var LLM_TOP_N to widen.
LLM_TOP_N = int(os.getenv("LLM_TOP_N", "10000"))

# Batch size for LLM requests.
LLM_BATCH = int(os.getenv("LLM_BATCH", "25"))

# Concurrent in-flight LLM batches. gpt-4o-mini tier-1 rate limits are
# ~500 RPM; concurrency 8 keeps us comfortably below.
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "8"))

# Batch size for Google Translate fallback.
GOOGLE_BATCH = 50


# ---------------------------------------------------------------------------
# Step 1 — extract unique strings
# ---------------------------------------------------------------------------

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def extract_unique_strings(df: pd.DataFrame) -> dict[str, str]:
    """Return {sha1_hex: original_string} for every non-null value across
    TEXT_COLS that survives in this dataframe."""
    out: dict[str, str] = {}
    for col in TEXT_COLS:
        if col not in df.columns:
            continue
        for v in df[col].dropna().unique():
            if not isinstance(v, str) or not v.strip():
                continue
            out[_hash(v)] = v
    return out


def frequency_rank(df: pd.DataFrame) -> dict[str, int]:
    """Return {sha1_hex: occurrence_count} aggregated across all TEXT_COLS."""
    counts: Counter[str] = Counter()
    for col in TEXT_COLS:
        if col not in df.columns:
            continue
        col_counts = df[col].dropna().value_counts()
        for v, n in col_counts.items():
            if isinstance(v, str) and v.strip():
                counts[_hash(v)] += int(n)
    return dict(counts)


# ---------------------------------------------------------------------------
# Step 2 — translation backends
# ---------------------------------------------------------------------------

@dataclass
class TranslationCache:
    """Persistent cache for one src→tgt pair. Idempotent across runs."""
    src_lang: str
    tgt_lang: str
    map: dict[str, str] = field(default_factory=dict)
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)
        return TRANSLATIONS_DIR / f"{self.src_lang}_{self.tgt_lang}.json"

    @classmethod
    def load(cls, src: str, tgt: str) -> "TranslationCache":
        cache = cls(src, tgt)
        if cache.path.exists():
            data = json.loads(cache.path.read_text(encoding="utf-8"))
            cache.map = data.get("map", {})
            cache.meta = data.get("meta", {})
        return cache

    def save(self) -> None:
        # Atomic write: build the JSON in a sibling temp file, then rename.
        # A mid-write crash (e.g. disk full) used to truncate the cache to
        # zero bytes and force a re-translation of every cached entry.
        payload = json.dumps(
            {"map": self.map, "meta": self.meta},
            ensure_ascii=False, indent=2,
        )
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.path)

    def has(self, h: str) -> bool:
        return h in self.map

    def get(self, h: str) -> str | None:
        return self.map.get(h)

    def set(self, h: str, value: str) -> None:
        self.map[h] = value


def _llm_prompt(
    texts: list[str], src_lang: str, tgt_lang: str, glossary: dict | None,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for any LLM backend."""
    glossary_block = ""
    if glossary:
        relevant = []
        for src_term, tgt_term in glossary.items():
            if any(src_term.lower() in t.lower() for t in texts):
                relevant.append(f"  - {src_term} -> {tgt_term}")
                if len(relevant) >= 30:
                    break
        if relevant:
            glossary_block = (
                "\nGlossary (must follow):\n" + "\n".join(relevant) + "\n"
            )

    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    system = (
        "You translate construction-industry technical strings from "
        f"{src_lang} to {tgt_lang}. Preserve all numbers, units, codes, "
        "abbreviations, and dimensions exactly. Do not paraphrase. "
        "Respond as a numbered list, one translation per input line, "
        "no preamble, no commentary." + glossary_block
    )
    return system, numbered


def _parse_numbered_list(raw: str, n_expected: int, fallback: list[str]) -> list[str]:
    parsed: list[str] = [""] * n_expected
    for ln in raw.splitlines():
        body = ln.lstrip()
        if not body:
            continue
        for sep in (".", ")", " -", ":"):
            if sep in body[:6]:
                head, _, rest = body.partition(sep)
                if head.strip().isdigit():
                    idx = int(head.strip()) - 1
                    if 0 <= idx < n_expected:
                        parsed[idx] = rest.strip()
                        break
    for i, p in enumerate(parsed):
        if not p:
            parsed[i] = fallback[i]
    return parsed


def _claude_translate_batch(
    texts: list[str], src_lang: str, tgt_lang: str, glossary: dict | None,
) -> list[str]:
    """One Claude call. Requires ANTHROPIC_API_KEY."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    sys_p, user_p = _llm_prompt(texts, src_lang, tgt_lang, glossary)
    msg = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=4096,
        system=sys_p,
        messages=[{"role": "user", "content": user_p}],
    )
    return _parse_numbered_list(msg.content[0].text, len(texts), texts)


def _openai_translate_batch(
    texts: list[str], src_lang: str, tgt_lang: str, glossary: dict | None,
) -> list[str]:
    """One OpenAI call. Requires OPENAI_API_KEY."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    sys_p, user_p = _llm_prompt(texts, src_lang, tgt_lang, glossary)
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user_p},
        ],
        max_tokens=4096,
        temperature=0.0,
    )
    return _parse_numbered_list(
        resp.choices[0].message.content or "", len(texts), texts,
    )


def _llm_translate_batch(
    texts: list[str], src_lang: str, tgt_lang: str, glossary: dict | None,
) -> list[str]:
    """Pick whichever LLM backend is configured, preferring Claude."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return _claude_translate_batch(texts, src_lang, tgt_lang, glossary)
    if os.getenv("OPENAI_API_KEY"):
        return _openai_translate_batch(texts, src_lang, tgt_lang, glossary)
    raise RuntimeError("Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set")


async def _openai_translate_batch_async(
    client, texts: list[str], src_lang: str, tgt_lang: str, glossary: dict | None,
) -> list[str]:
    sys_p, user_p = _llm_prompt(texts, src_lang, tgt_lang, glossary)
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": user_p},
                ],
                max_tokens=4096,
                temperature=0.0,
                timeout=60,
            )
            return _parse_numbered_list(
                resp.choices[0].message.content or "", len(texts), texts,
            )
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(2 ** attempt)
    return texts


def _google_translate_batch(
    texts: list[str], src_lang: str, tgt_lang: str,
) -> list[str]:
    """Google Translate via deep_translator."""
    try:
        from deep_translator import GoogleTranslator
    except ImportError as e:
        raise RuntimeError(
            "deep_translator not installed. `pip install deep-translator`."
        ) from e
    gt = GoogleTranslator(source=src_lang, target=tgt_lang)
    out: list[str] = []
    for t in texts:
        try:
            out.append(gt.translate(t) or t)
        except Exception:
            out.append(t)   # fall through; keep source
    return out


# ---------------------------------------------------------------------------
# Step 3 — orchestration
# ---------------------------------------------------------------------------

def _load_glossary_seed() -> dict:
    if not SEED_GLOSSARY.exists():
        return {}
    return json.loads(SEED_GLOSSARY.read_text(encoding="utf-8"))


def translate_strings(
    unique_strings: dict[str, str],
    freq: dict[str, int],
    src_lang: str,
    tgt_lang: str,
    llm_top_n: int = LLM_TOP_N,
) -> TranslationCache:
    """
    Translate every entry in unique_strings, populating the persistent cache.
    Skip entries already cached. Use Claude for the top-N by frequency,
    Google for the rest.
    """
    cache = TranslationCache.load(src_lang, tgt_lang)
    glossary = _load_glossary_seed().get(f"{src_lang}_{tgt_lang}", {})

    pending = [h for h in unique_strings if not cache.has(h)]
    if not pending:
        print(f"      cache hit: {len(cache.map):,} pre-existing translations")
        return cache

    # Rank pending hashes by frequency.
    pending.sort(key=lambda h: -freq.get(h, 0))
    top = pending[:llm_top_n]
    tail = pending[llm_top_n:]
    print(
        f"      translate: {len(top):,} via LLM + {len(tail):,} via Google "
        f"({len(cache.map):,} already cached)"
    )

    # LLM pass (Claude > OpenAI > Google fallback).
    has_llm = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))
    if top and has_llm and os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        # Async path for OpenAI — major speedup via concurrent batches.
        print(f"      LLM backend: OpenAI (concurrency {LLM_CONCURRENCY})")
        asyncio.run(_translate_openai_concurrent(
            top, unique_strings, cache, src_lang, tgt_lang, glossary,
        ))
    elif top and has_llm:
        # Sync path (Claude or fallback) — single-threaded.
        backend = "Claude"
        print(f"      LLM backend: {backend}")
        for i in range(0, len(top), LLM_BATCH):
            batch_h = top[i:i + LLM_BATCH]
            batch_txt = [unique_strings[h] for h in batch_h]
            try:
                trs = _llm_translate_batch(
                    batch_txt, src_lang, tgt_lang, glossary
                )
            except Exception as e:
                print(f"      {backend} batch {i}: {e}; falling back to Google")
                trs = _google_translate_batch(batch_txt, src_lang, tgt_lang)
            for h, tr in zip(batch_h, trs):
                cache.set(h, tr)
            if i % (LLM_BATCH * 10) == 0:
                cache.save()
                time.sleep(0.1)
        cache.save()
    elif not has_llm:
        print("      no LLM key set; sending top-N to Google as well")
        tail = top + tail
        top = []

    # Long tail beyond LLM_TOP_N is left untranslated. The remaining
    # values keep their source-language form in the dataframe; embeddings
    # over those rows will index source-language text. This is an
    # acceptable trade-off: the long tail consists of one-off rate
    # descriptions whose translation cost is high and reuse value is low.
    if tail:
        print(
            f"      long tail: {len(tail):,} strings left in source "
            f"language (LLM_TOP_N cutoff)"
        )

    # Google pass: only when no LLM key is configured at all (the default
    # path with OpenAI now leaves the tail untranslated rather than spending
    # an extra hour on Google for marginal-utility strings).
    if tail and not has_llm:
        for i in range(0, len(tail), GOOGLE_BATCH):
            batch_h = tail[i:i + GOOGLE_BATCH]
            batch_txt = [unique_strings[h] for h in batch_h]
            try:
                trs = _google_translate_batch(batch_txt, src_lang, tgt_lang)
            except Exception as e:
                print(f"      Google batch {i}: {e}; keeping originals")
                trs = batch_txt
            for h, tr in zip(batch_h, trs):
                cache.set(h, tr)
            if i % (GOOGLE_BATCH * 20) == 0:
                cache.save()
        cache.save()

    return cache


async def _translate_openai_concurrent(
    top_hashes: list[str],
    unique_strings: dict[str, str],
    cache: TranslationCache,
    src_lang: str,
    tgt_lang: str,
    glossary: dict | None,
) -> None:
    """
    Run LLM_CONCURRENCY async OpenAI requests in flight, draining batches
    from a shared queue. Save the cache every N completed batches so a
    crash never loses more than ~LLM_BATCH × N translations.
    """
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    batches: list[list[str]] = [
        top_hashes[i:i + LLM_BATCH]
        for i in range(0, len(top_hashes), LLM_BATCH)
    ]
    sem = asyncio.Semaphore(LLM_CONCURRENCY)
    completed = 0
    save_every = LLM_CONCURRENCY * 5

    async def _do_batch(batch_h: list[str]) -> None:
        nonlocal completed
        batch_txt = [unique_strings[h] for h in batch_h]
        async with sem:
            try:
                trs = await _openai_translate_batch_async(
                    client, batch_txt, src_lang, tgt_lang, glossary,
                )
            except Exception as e:
                print(f"      batch failed: {e}; keeping originals")
                trs = batch_txt
        for h, tr in zip(batch_h, trs):
            cache.set(h, tr)
        completed += 1
        if completed % save_every == 0:
            cache.save()
            print(
                f"      progress: {completed * LLM_BATCH:,}/"
                f"{len(top_hashes):,} translated"
            )

    await asyncio.gather(*(_do_batch(b) for b in batches))
    cache.save()
    await client.close()


# ---------------------------------------------------------------------------
# Step 4 — apply translations to dataframe
# ---------------------------------------------------------------------------

def remap(df: pd.DataFrame, cache: TranslationCache) -> pd.DataFrame:
    """Replace every TEXT_COLS value via the cached translation table."""
    out = df.copy()
    for col in TEXT_COLS:
        if col not in out.columns:
            continue

        def _tr(v):
            if not isinstance(v, str) or not v.strip():
                return v
            return cache.get(_hash(v)) or v

        out[col] = out[col].map(_tr)
    return out


# ---------------------------------------------------------------------------
# Public entrypoint used by add_country_track.py
# ---------------------------------------------------------------------------

def run(df: pd.DataFrame, source_lang: str, target_lang: str) -> pd.DataFrame:
    if source_lang == target_lang:
        return df
    print(f"      extracting unique strings ...")
    unique = extract_unique_strings(df)
    print(f"      {len(unique):,} unique strings to translate")

    print(f"      computing frequency rank ...")
    freq = frequency_rank(df)

    cache = translate_strings(unique, freq, source_lang, target_lang)

    print(f"      remapping dataframe via hash join ...")
    return remap(df, cache)
