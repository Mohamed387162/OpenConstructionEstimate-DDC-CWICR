"""Translate values in newly-added TEXT_COLS columns across all 14 pairs.

Newly added columns (composite price descriptors and price_region) hold
source-language descriptions that were never funnelled through the
translation pipeline. This script extracts their unique values from the
DE_BERLIN / UK_GBP sources, calls gpt-4o-mini in parallel via raw
urllib (the OpenAI SDK hangs on Windows), and writes back to each
de_*/en_*.json cache.

After this script, run add_country_track.py to apply the new cache.
"""
from __future__ import annotations
import sys, os, json, hashlib, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_loader  # noqa
import pandas as pd

NEW_COLS = (
    "price_abstract_resource_common_start",
    "price_abstract_resource_variable_parts",
    "price_abstract_resource_est_price_range",
    "price_abstract_resource_est_price_all_values",
    "price_abstract_resource_group_per_unit",
    "price_abstract_resource_common_start_per_unit",
    "price_abstract_resource_variable_parts_per_unit",
    "price_abstract_resource_est_price_range_per_unit",
    "price_abstract_resource_est_price_all_values_per_unit",
    "price_region",
)

REPO = r'C:\Users\Artem Boiko\Desktop\CodeProjects\legal-restructure-2026-04\OpenConstructionEstimate-DDC-CWICR'
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, 'logs', 'translate_new_cols.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

DE_PARQUET = os.path.join(REPO, 'DE___DDC_CWICR',
                          'DE_BERLIN_workitems_costs_resources_DDC_CWICR.parquet')
UK_PARQUET = os.path.join(REPO, 'UK___DDC_CWICR',
                          'UK_GBP_workitems_costs_resources_DDC_CWICR.parquet')

PAIRS = [
    ('de', 'it', 'Italian',    DE_PARQUET, 'German'),
    ('de', 'ro', 'Romanian',   DE_PARQUET, 'German'),
    ('de', 'bg', 'Bulgarian',  DE_PARQUET, 'German'),
    ('de', 'hr', 'Croatian',   DE_PARQUET, 'German'),
    ('de', 'cs', 'Czech',      DE_PARQUET, 'German'),
    ('de', 'pl', 'Polish',     DE_PARQUET, 'German'),
    ('de', 'nl', 'Dutch',      DE_PARQUET, 'German'),
    ('de', 'sv', 'Swedish',    DE_PARQUET, 'German'),
    ('de', 'tr', 'Turkish',    DE_PARQUET, 'German'),
    ('en', 'id', 'Indonesian', UK_PARQUET, 'English'),
    ('en', 'vi', 'Vietnamese', UK_PARQUET, 'English'),
    ('en', 'ja', 'Japanese',   UK_PARQUET, 'English'),
    ('en', 'ko', 'Korean',     UK_PARQUET, 'English'),
    ('en', 'th', 'Thai',       UK_PARQUET, 'English'),
]

API_URL = 'https://api.openai.com/v1/chat/completions'


def strhash(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:16]


def log(msg: str) -> None:
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def extract_uniques(parquet_path: str) -> set[str]:
    df = pd.read_parquet(parquet_path, columns=list(NEW_COLS))
    uniq: set[str] = set()
    for col in NEW_COLS:
        if col in df.columns:
            for v in df[col].dropna().astype(str).unique():
                if v and v != 'nan':
                    uniq.add(v)
    return uniq


def call_openai(sys_prompt: str, user_prompt: str, *,
                api_key: str, timeout: float = 120.0) -> str:
    body = {
        'model': 'gpt-4o-mini',
        'messages': [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'max_tokens': 8192,
        'temperature': 0.0,
    }
    req = urllib.request.Request(
        API_URL,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        data=json.dumps(body).encode('utf-8'),
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        d = json.loads(resp.read().decode('utf-8'))
    return d['choices'][0]['message']['content'] or ''


def parse_numbered(raw: str, n: int, fallback: list[str]) -> list[str]:
    out = [''] * n
    for ln in raw.splitlines():
        body = ln.lstrip()
        if not body:
            continue
        for sep in ('.', ')'):
            if sep in body[:6]:
                head, _, rest = body.partition(sep)
                if head.strip().isdigit():
                    idx = int(head.strip()) - 1
                    if 0 <= idx < n:
                        out[idx] = rest.strip()
                        break
    for i, p in enumerate(out):
        if not p:
            out[i] = fallback[i]
    return out


def build_prompt(texts: list[str], src_full: str, tgt_full: str) -> tuple[str, str]:
    sys_prompt = (
        f"You are a professional construction-industry translator. Translate "
        f"every input from {src_full} to {tgt_full}. The strings are composite "
        f"price-catalog descriptors: bullet-separated (•) lists of resource "
        f"variants, or 'unit=tech_group' maps. They describe construction "
        f"materials, geotextiles, and equipment.\n"
        f"Rules:\n"
        f"- Translate ALL natural-language words (descriptions, group names, "
        f"adjectives) into {tgt_full}.\n"
        f"- Preserve ALL numbers, dimensions, units (kg, m, mm, m2, m3, kN, "
        f"°C, g/m2, kN/m, W/(m*K)), separator characters (• , = ; / -), and "
        f"any code with digits (DN-200, MM-01-1).\n"
        f"- Composite items: keep the structure. 'unit=description' stays "
        f"'unit=translated_description'. Bullets between items stay '•'.\n"
        f"- NEVER copy a {src_full} word verbatim into the output.\n"
        f"- Respond as a numbered list, one translation per input, no "
        f"preamble, no commentary."
    )
    user_prompt = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    return sys_prompt, user_prompt


def main() -> int:
    api_key = os.environ['OPENAI_API_KEY']

    log('Loading source parquets and extracting uniques in NEW_COLS...')
    de_uniq = extract_uniques(DE_PARQUET)
    en_uniq = extract_uniques(UK_PARQUET)
    log(f'DE source: {len(de_uniq):,} unique strings')
    log(f'EN source: {len(en_uniq):,} unique strings')

    BATCH = 8  # smaller batch — composite values are long and timeout-prone
    CONCURRENCY = int(os.environ.get('LLM_CONCURRENCY', 30))
    grand_total = 0

    for src_lang, tgt_lang, tgt_full, _src_path, src_full in PAIRS:
        cache_fp = os.path.join(HERE, 'glossary', 'translations',
                                f'{src_lang}_{tgt_lang}.json')
        cache = json.load(open(cache_fp, encoding='utf-8'))
        m = cache.setdefault('map', {})
        uniq = de_uniq if src_lang == 'de' else en_uniq
        missing = [s for s in uniq if strhash(s) not in m]
        log(f'{src_lang}_{tgt_lang}: {len(missing)} missing entries')
        if not missing:
            continue

        chunks = [missing[i:i+BATCH] for i in range(0, len(missing), BATCH)]

        def do_chunk(chunk: list[str]) -> tuple[list[str], list[str]]:
            sys_p, user_p = build_prompt(chunk, src_full, tgt_full)
            try:
                raw = call_openai(sys_p, user_p, api_key=api_key)
            except Exception as e:
                return chunk, [f'__ERR__:{type(e).__name__}'] * len(chunk)
            return chunk, parse_numbered(raw, len(chunk), chunk)

        n_done = n_err = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futures = [ex.submit(do_chunk, c) for c in chunks]
            for k, fut in enumerate(as_completed(futures)):
                chunk, outs = fut.result()
                if outs and outs[0].startswith('__ERR__'):
                    n_err += 1
                    if n_err <= 3:
                        log(f'  {src_lang}_{tgt_lang} batch err: {outs[0]}')
                    continue
                for src, tgt in zip(chunk, outs):
                    m[strhash(src)] = tgt
                n_done += len(chunk)
                if (k + 1) % 30 == 0:
                    tmp = cache_fp + '.tmp'
                    json.dump(cache, open(tmp, 'w', encoding='utf-8'),
                              ensure_ascii=False, indent=2)
                    os.replace(tmp, cache_fp)
                    log(f'  {src_lang}_{tgt_lang}: progress '
                        f'{n_done}/{len(missing)} ({time.time()-t0:.0f}s, '
                        f'err={n_err})')

        tmp = cache_fp + '.tmp'
        json.dump(cache, open(tmp, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        os.replace(tmp, cache_fp)
        grand_total += n_done
        log(f'{src_lang}_{tgt_lang}: done ({n_done}/{len(missing)}, '
            f'errors={n_err}, total {grand_total}, {time.time()-t0:.0f}s)')

    log(f'TOTAL: {grand_total} entries translated across {len(PAIRS)} pairs')
    return 0


if __name__ == '__main__':
    sys.exit(main())
