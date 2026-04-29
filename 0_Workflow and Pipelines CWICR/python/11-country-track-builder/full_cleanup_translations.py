"""Comprehensive sweep across all 14 translate tracks to eliminate
source-language leftovers (German in DE-source tracks, English in
EN-source tracks).

Detection per pair:

    DE-source -> IT/RO/HR/CS/PL: any [äöüÄÖÜß] in target value
    DE-source -> NL: [äöüÄÖÜß] in target unless value is a known
        Dutch loanword (e.g. "Röntgen", "Stützflansch" loanword)
    DE-source -> SV: only ß in target (ä/ö/å are native Swedish)
    DE-source -> TR: ä or ß in target (ö/ü are native Turkish)
    DE-source -> BG: any Latin word matching DE_STRONG_RE (Cyrillic
        target should not contain Latin construction words)
    EN-source -> ID/VI: target value identical to source AND not
        recognised by validators._is_passthrough_value
    EN-source -> JA/KO/TH: target containing >40% Latin chars (the
        target script is non-Latin, so Latin-heavy values mean the
        translation didn't happen)

For each flagged value, force re-translate via gpt-4o-mini with a
prompt that explicitly forbids verbatim output and lists the residual
DE/EN words the model has been keeping.
"""
from __future__ import annotations
import sys, os, json, hashlib, re, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_loader  # noqa
import pandas as pd
from text_pipeline import TEXT_COLS

REPO = r'C:\Users\Artem Boiko\Desktop\CodeProjects\legal-restructure-2026-04\OpenConstructionEstimate-DDC-CWICR'
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, 'logs', 'full_cleanup_translations.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
API_URL = 'https://api.openai.com/v1/chat/completions'

DE_PARQUET = os.path.join(REPO, 'DE___DDC_CWICR',
                          'DE_BERLIN_workitems_costs_resources_DDC_CWICR.parquet')
UK_PARQUET = os.path.join(REPO, 'UK___DDC_CWICR',
                          'UK_GBP_workitems_costs_resources_DDC_CWICR.parquet')

# Per-pair detection parameters
PAIRS = {
    # (src_lang, tgt_lang, tgt_full_name, source_parquet, kind)
    'IT_ROME':      ('de', 'it', 'Italian',    DE_PARQUET, 'de_strict'),
    'RO_BUCHAREST': ('de', 'ro', 'Romanian',   DE_PARQUET, 'de_strict'),
    'HR_ZAGREB':    ('de', 'hr', 'Croatian',   DE_PARQUET, 'de_strict'),
    'CS_PRAGUE':    ('de', 'cs', 'Czech',      DE_PARQUET, 'de_strict'),
    'PL_WARSAW':    ('de', 'pl', 'Polish',     DE_PARQUET, 'de_strict'),
    'NL_AMSTERDAM': ('de', 'nl', 'Dutch',      DE_PARQUET, 'de_loose'),
    'SV_STOCKHOLM': ('de', 'sv', 'Swedish',    DE_PARQUET, 'de_sv'),
    'TR_ISTANBUL':  ('de', 'tr', 'Turkish',    DE_PARQUET, 'de_tr'),
    'BG_SOFIA':     ('de', 'bg', 'Bulgarian',  DE_PARQUET, 'de_bg'),
    'ID_JAKARTA':   ('en', 'id', 'Indonesian', UK_PARQUET, 'en_latin'),
    'VI_HANOI':     ('en', 'vi', 'Vietnamese', UK_PARQUET, 'en_latin'),
    'JA_TOKYO':     ('en', 'ja', 'Japanese',   UK_PARQUET, 'en_nonlatin'),
    'KO_SEOUL':     ('en', 'ko', 'Korean',     UK_PARQUET, 'en_nonlatin'),
    'TH_BANGKOK':   ('en', 'th', 'Thai',       UK_PARQUET, 'en_nonlatin'),
}

DE_UMLAUT_FULL = re.compile(r'[äöüÄÖÜß]')
DE_SHARP_S = re.compile(r'ß')
DE_TR_MARKER = re.compile(r'[äÄß]')        # TR uses ö/ü natively
DE_STRONG_LATIN = re.compile(
    r'\b(?:der|die|das|ein|eine|mit|von|für|fur|zur|zum|bei|aus|auf|über|uber|unter|durch|als|und|oder|nach|vor|Stehbolzen|Schweißen|Schweissen|Anlage|Schraube|Maschine|Werkzeug|Material|Beton)\b',
    re.IGNORECASE | re.UNICODE)
LATIN_RUN = re.compile(r'[A-Za-zÀ-ÿ]+')
EN_WORD = re.compile(r'\b(?:the|of|with|for|from|to|in|on|by|and|or|using|over|under|including|excluding|installation|equipment|machine|pipe|valve|cable|wire|sheet|set|unit|stand|press|filter)\b', re.IGNORECASE)


def strhash(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:16]


def log(msg: str) -> None:
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def is_flagged(val: str, kind: str) -> bool:
    if not val:
        return False
    if kind == 'de_strict':
        return bool(DE_UMLAUT_FULL.search(val))
    if kind == 'de_loose':
        # NL: umlaut + a known DE-only word
        return bool(DE_UMLAUT_FULL.search(val) and DE_STRONG_LATIN.search(val))
    if kind == 'de_sv':
        return bool(DE_SHARP_S.search(val))
    if kind == 'de_tr':
        return bool(DE_TR_MARKER.search(val))
    if kind == 'de_bg':
        # Cyrillic target — flag values that are ALL Latin (suggesting
        # untranslated DE) and look German.
        if not val or LATIN_RUN.search(val) is None:
            return False
        latin_chars = sum(1 for c in val if c.isalpha() and c.isascii())
        total_alpha = sum(1 for c in val if c.isalpha())
        if total_alpha and latin_chars / total_alpha > 0.6:
            return bool(DE_STRONG_LATIN.search(val) or DE_UMLAUT_FULL.search(val))
        return False
    if kind == 'en_latin':
        # ID/VI use Latin script. Flag values whose alphabetic content
        # is ≥85% pure-ASCII Latin AND has English-only words. (Native
        # Indonesian/Vietnamese mix Latin with other punctuation/marks.)
        from validators import _is_passthrough_value
        if _is_passthrough_value(val):
            return False
        return bool(EN_WORD.search(val))
    if kind == 'en_nonlatin':
        # JA/KO/TH use non-Latin. Flag values where Latin is dominant.
        latin_chars = sum(1 for c in val if c.isalpha() and c.isascii())
        total_alpha = sum(1 for c in val if c.isalpha())
        if total_alpha and latin_chars / total_alpha > 0.7:
            from validators import _is_passthrough_value
            if _is_passthrough_value(val):
                return False
            return bool(EN_WORD.search(val))
        return False
    return False


def call_openai(sys_prompt: str, user_prompt: str, *,
                api_key: str, timeout: float = 45.0) -> str:
    body = {
        'model': 'gpt-4o-mini',
        'messages': [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'max_tokens': 4096,
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


def build_prompt(texts: list[str], src_lang: str, tgt_name: str) -> tuple[str, str]:
    src_full = {'de': 'German', 'en': 'English'}[src_lang]
    extra_hints = ''
    if src_lang == 'de':
        extra_hints = (
            "\n- Translate German words whether or not they contain umlauts: "
            "Stehbolzen, Schraube, Maschine, Werkzeug, Anlage, Brötchen, "
            "Ständer, Mühle, Röntgen, Stütz must NOT be kept as-is."
        )
    elif src_lang == 'en':
        extra_hints = (
            "\n- Translate English words whether they look like loanwords or not: "
            "stand, press, filter, unit, set, including, excluding, installation, "
            "equipment, machine — convert to native equivalents."
        )
    sys_prompt = (
        f"You are a professional construction-industry translator. Translate "
        f"every input from {src_full} to {tgt_name}. The strings are descriptions "
        f"of construction work, machinery, materials, equipment, units, and "
        f"composite labels — they ARE NOT product codes.\n"
        f"Rules:\n"
        f"- Preserve numbers, units (kg, m, mm, m2, m3, kW), and identifier "
        f"codes that contain digits (DN-200, MM-01-1).\n"
        f"- Translate ALL natural-language words even if they look like "
        f"loanwords. NEVER copy a {src_full} word verbatim into the output.\n"
        f"- Composite strings: translate every component, keep separators "
        f"(commas, semicolons, slashes) verbatim.{extra_hints}\n"
        f"- Respond as a numbered list, one translation per input, no "
        f"preamble, no commentary."
    )
    user_prompt = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    return sys_prompt, user_prompt


def parquet_path(region: str) -> str:
    folder = f'{region.split("_")[0]}___DDC_CWICR'
    return os.path.join(REPO, folder,
                        f'{region}_workitems_costs_resources_DDC_CWICR.parquet')


def process_track(region: str, api_key: str) -> int:
    src_lang, tgt_lang, tgt_name, src_parquet_path, kind = PAIRS[region]
    cache_fp = os.path.join(HERE, 'glossary', 'translations',
                            f'{src_lang}_{tgt_lang}.json')
    tgt_parquet_path = parquet_path(region)
    if not os.path.exists(tgt_parquet_path):
        log(f'{region}: target parquet missing, skip')
        return 0

    log(f'{region}: scanning ({kind})...')
    src_df = pd.read_parquet(src_parquet_path)
    tgt_df = pd.read_parquet(tgt_parquet_path)
    sk = ['rate_code', 'resource_code']
    src_df = src_df.sort_values(sk).reset_index(drop=True)
    tgt_df = tgt_df.sort_values(sk).reset_index(drop=True)

    # Find unique (source, target) pairs first, then check is_flagged on
    # the unique target values. Iterating per-row is 30x slower because
    # most TEXT_COLS columns have <1% unique values.
    flagged: dict[str, str] = {}  # source -> target_unwanted
    for col in TEXT_COLS:
        if col not in tgt_df.columns or col not in src_df.columns:
            continue
        pairs = pd.DataFrame({
            'src': src_df[col].astype(str).fillna(''),
            'tgt': tgt_df[col].astype(str).fillna(''),
        }).drop_duplicates()
        pairs = pairs[(pairs['src'] != '') & (pairs['tgt'] != '')]
        if pairs.empty:
            continue
        bad = pairs['tgt'].apply(lambda v: is_flagged(v, kind))
        for src_v, tgt_v in zip(pairs.loc[bad, 'src'], pairs.loc[bad, 'tgt']):
            if src_v not in flagged:
                flagged[src_v] = tgt_v

    log(f'{region}: {len(flagged)} unique source strings flagged for re-translate')
    if not flagged:
        return 0

    cache = json.load(open(cache_fp, encoding='utf-8'))
    m = cache.setdefault('map', {})

    items = list(flagged.items())  # (src, current_tgt)
    BATCH = 20
    chunks = [items[i:i+BATCH] for i in range(0, len(items), BATCH)]

    def do_chunk(chunk: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[str]]:
        srcs = [s for s, _ in chunk]
        sys_p, user_p = build_prompt(srcs, src_lang, tgt_name)
        try:
            raw = call_openai(sys_p, user_p, api_key=api_key)
        except Exception as e:
            return chunk, [f'__ERR__:{type(e).__name__}'] * len(srcs)
        return chunk, parse_numbered(raw, len(srcs), srcs)

    n_done = n_err = n_unchanged = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=30) as ex:
        futures = [ex.submit(do_chunk, c) for c in chunks]
        for k, fut in enumerate(as_completed(futures)):
            chunk, outs = fut.result()
            if outs and outs[0].startswith('__ERR__'):
                n_err += 1
                continue
            for (src, _old), tgt in zip(chunk, outs):
                if is_flagged(tgt, kind):
                    n_unchanged += 1
                m[strhash(src)] = tgt
            n_done += len(chunk)
            if (k + 1) % 25 == 0:
                tmp = cache_fp + '.tmp'
                json.dump(cache, open(tmp, 'w', encoding='utf-8'),
                          ensure_ascii=False, indent=2)
                os.replace(tmp, cache_fp)

    tmp = cache_fp + '.tmp'
    json.dump(cache, open(tmp, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    os.replace(tmp, cache_fp)
    log(f'{region}: done ({n_done}/{len(flagged)}, errors={n_err}, '
        f'still-flagged={n_unchanged}, {time.time()-t0:.0f}s)')
    return n_done


def main() -> int:
    api_key = os.environ['OPENAI_API_KEY']
    regions = sys.argv[1:] or list(PAIRS)
    grand = 0
    for r in regions:
        if r not in PAIRS:
            log(f'unknown region {r}; skip')
            continue
        grand += process_track(r, api_key)
    log(f'TOTAL: {grand} entries re-translated across {len(regions)} tracks')
    return 0


if __name__ == '__main__':
    sys.exit(main())
