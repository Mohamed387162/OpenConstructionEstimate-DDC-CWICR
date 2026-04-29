"""Find cache entries with German umlauts in target and re-translate.

LLM occasionally returns German loan-words verbatim because the umlaut
makes the model treat them as proper nouns / codes. We sweep each DE→X
cache for any target value containing [äöüÄÖÜß] and force a proper
re-translation, even if target != source (because partial verbatim like
'Instalare Mühlenoxid' from source 'Anlage Mühlenoxid' won't be caught
by simple target==source detection).
"""
from __future__ import annotations
import sys, os, json, hashlib, re, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_loader  # noqa
import pandas as pd
from text_pipeline import TEXT_COLS

DE_PARQUET = r'C:\Users\Artem Boiko\Desktop\CodeProjects\legal-restructure-2026-04\OpenConstructionEstimate-DDC-CWICR\DE___DDC_CWICR\DE_BERLIN_workitems_costs_resources_DDC_CWICR.parquet'

PAIRS = {
    'it': 'Italian', 'ro': 'Romanian', 'bg': 'Bulgarian',
    'hr': 'Croatian', 'cs': 'Czech', 'pl': 'Polish',
    'nl': 'Dutch', 'sv': 'Swedish', 'tr': 'Turkish',
}
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, 'logs', 'fix_umlaut_leftovers.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
API_URL = 'https://api.openai.com/v1/chat/completions'

UMLAUT_RE = re.compile(r'[äöüÄÖÜß]')


def strhash(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:16]


def log(msg: str) -> None:
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


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


def main() -> int:
    api_key = os.environ['OPENAI_API_KEY']
    log('Loading DE source for hash lookup...')
    de = pd.read_parquet(DE_PARQUET)
    # Map hash(src) → src for any text value in DE (so we can recover
    # source from cache key).
    de_unique: set[str] = set()
    for col in TEXT_COLS:
        if col in de.columns:
            for v in de[col].dropna().astype(str).unique():
                if v: de_unique.add(v)
    hash_to_src = {strhash(s): s for s in de_unique}
    log(f'DE source: {len(de_unique):,} unique strings indexed')

    grand = 0
    for tgt_lang, tgt_name in PAIRS.items():
        cache_fp = os.path.join(HERE, 'glossary', 'translations', f'de_{tgt_lang}.json')
        cache = json.load(open(cache_fp, encoding='utf-8'))
        m = cache.setdefault('map', {})

        # Find entries where target contains umlaut.
        flagged_keys: list[str] = []
        for k, v in m.items():
            if isinstance(v, str) and UMLAUT_RE.search(v):
                flagged_keys.append(k)
        log(f'de_{tgt_lang}: {len(flagged_keys)} cache entries with umlaut in target')
        if not flagged_keys:
            continue

        # Resolve source strings.
        items = [(k, hash_to_src.get(k)) for k in flagged_keys]
        items = [(k, s) for k, s in items if s]
        log(f'de_{tgt_lang}: {len(items)} resolvable source strings')
        if not items:
            continue

        BATCH = 20
        chunks = [items[i:i+BATCH] for i in range(0, len(items), BATCH)]

        def do_chunk(chunk: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[str]]:
            srcs = [s for _, s in chunk]
            sys_p = (
                f"Translate every input from German to {tgt_name}. The strings are "
                f"construction terms — translate ALL natural-language words including "
                f"'Brötchen', 'Ständer', 'Mühle', 'Röntgen', 'Stützflansch'. NEVER keep "
                f"the German word as-is. Preserve numbers, units (kg, m, mm), and codes "
                f"with digits ('DN-200', 'MM-01-1'). Respond as a numbered list, one "
                f"translation per input, no preamble."
            )
            user_p = "\n".join(f"{i+1}. {s}" for i, s in enumerate(srcs))
            try:
                raw = call_openai(sys_p, user_p, api_key=api_key)
            except Exception as e:
                return chunk, [f'__ERR__:{type(e).__name__}'] * len(srcs)
            return chunk, parse_numbered(raw, len(srcs), srcs)

        n_done = n_err = n_unchanged = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = [ex.submit(do_chunk, c) for c in chunks]
            for k, fut in enumerate(as_completed(futures)):
                chunk, outs = fut.result()
                if outs and outs[0].startswith('__ERR__'):
                    n_err += 1
                    continue
                for (key, src), tgt in zip(chunk, outs):
                    if UMLAUT_RE.search(tgt) and tgt == m.get(key):
                        n_unchanged += 1
                    m[key] = tgt
                n_done += len(chunk)

        tmp = cache_fp + '.tmp'
        json.dump(cache, open(tmp, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        os.replace(tmp, cache_fp)
        log(f'de_{tgt_lang}: done ({n_done}/{len(items)}, errors={n_err}, '
            f'still-umlaut={n_unchanged}, {time.time()-t0:.0f}s)')
        grand += n_done

    log(f'TOTAL: {grand} entries re-translated')
    return 0


if __name__ == '__main__':
    sys.exit(main())
