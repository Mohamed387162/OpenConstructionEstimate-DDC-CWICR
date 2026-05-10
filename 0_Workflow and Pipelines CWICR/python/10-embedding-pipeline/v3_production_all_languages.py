"""
V3 production encoder for ALL 30 CWICR languages.

Loops over each <COUNTRY>___DDC_CWICR/ folder, encodes its parquet
into cwicr_<lang>_v3, then saves snapshot back to the country folder
with naming: <COUNTRY>_<CITY>_workitems_costs_resources_EMBEDDINGS_BGEM3_V3_DDC_CWICR.snapshot

Idempotent: if collection already has correct N points, skips.
Re-uses BGE-M3 model across countries (loaded once).

Expected wall time: ~55 min/country × 29 = ~26 hours on RTX 2050.
"""

from __future__ import annotations

import re
import sys
import time
import uuid
import urllib.request
from pathlib import Path

import pandas as pd

from v3_classifier import enrich_v3
from v3_production_ru import (
    compose_bim, BATCH_SIZE, MAX_LENGTH, EMBED_DIM, ID_NAMESPACE,
    UPSERT_BATCH, QDRANT_URL,
)
from ost_ifc_mapping import base_code

ROOT = Path(r"C:\Users\Artem Boiko\Desktop\CodeProjects\legal-restructure-2026-04\OpenConstructionEstimate-DDC-CWICR")
RU_PARQUET = ROOT / "RU___DDC_CWICR" / "RU_STPETERSBURG_workitems_costs_resources_DDC_CWICR.parquet"
SKIP_COUNTRIES = {"RU", "EN", "AU", "ES", "ID", "JA", "KO"}  # Confirmed 100% coverage; AR/BG/CS/DE/FR/HI/HR/IT need re-encode w/ extended INLINE_LANG_TAGS


# ===========================================================================
# RU classification cache — built once, propagated to all other languages
# via base_code() lookup (98.3% coverage between RU↔EN/DE/FR/...)
# ===========================================================================
_RU_CACHE = None

def get_ru_classifications():
    """Build {base_code(ru_rate_code): enrich_v3_result} once and reuse."""
    global _RU_CACHE
    if _RU_CACHE is not None:
        return _RU_CACHE
    print(f"\nBuilding RU classification cache from {RU_PARQUET.name} ...")
    t0 = time.time()
    df = pd.read_parquet(RU_PARQUET).drop_duplicates("rate_code")
    cache = {}
    n_high = 0
    for r in df.itertuples(index=False):
        rd = r._asdict()
        e = enrich_v3(rd)
        bc = base_code(rd["rate_code"])
        cache[bc] = e
        if e.get("classification_confidence") == "high":
            n_high += 1
    print(f"  cached {len(cache):,} base_codes in {time.time()-t0:.0f}s "
          f"({n_high:,} high confidence)")
    _RU_CACHE = cache
    return cache


def classify_with_ru_propagation(rate_code: str, rd: dict, ru_cache: dict) -> dict:
    """Use RU classification if base_code matches, else fall back to local."""
    bc = base_code(rate_code)
    if bc in ru_cache:
        return ru_cache[bc]
    # Fallback: try local enrich (may give empty for non-RU text)
    return enrich_v3(rd)


# ===========================================================================
# Country discovery
# ===========================================================================
def discover_countries():
    """Return list of (country_code, parquet_path) sorted alphabetically."""
    out = []
    for d in sorted(ROOT.glob("*___DDC_CWICR")):
        country = d.name.split("___")[0]
        parquets = list(d.glob("*workitems_costs_resources_DDC_CWICR.parquet"))
        if parquets:
            out.append((country, parquets[0]))
    return out


def derive_snapshot_name(parquet_path: Path) -> str:
    """RU_STPETERSBURG_workitems_..._DDC_CWICR.parquet → RU_STPETERSBURG_..._BGEM3_V3_DDC_CWICR.snapshot"""
    base = parquet_path.stem
    return base.replace("_workitems_costs_resources_DDC_CWICR",
                         "_workitems_costs_resources_EMBEDDINGS_BGEM3_V3_DDC_CWICR") + ".snapshot"


# ===========================================================================
# Per-country pipeline (re-uses model)
# ===========================================================================
def encode_country(country: str, parquet_path: Path, model, qdrant):
    from qdrant_client.models import (
        Distance, VectorParams, PayloadSchemaType, PointStruct,
        SparseVectorParams, SparseIndexParams, SparseVector,
    )

    collection = f"cwicr_{country.lower()}_v3"
    print(f"\n{'='*70}")
    print(f"Country: {country} | parquet: {parquet_path.name}")
    print(f"Collection: {collection}")
    print(f"{'='*70}")
    t_start = time.time()

    # Load + aggregate
    print("[1/5] Loading parquet ...")
    df_full = pd.read_parquet(parquet_path)
    print(f"  {len(df_full):,} rows")

    print("[2/5] Aggregating per rate_code ...")
    wc_lookup = (df_full.groupby("rate_code")["work_composition_text"]
                  .apply(lambda s: list(s.dropna().astype(str).unique())).to_dict())
    res_lookup = (df_full.groupby("rate_code")["resource_name"]
                   .apply(lambda s: list(s.dropna().astype(str).unique())).to_dict())
    rate_df = df_full.drop_duplicates("rate_code").reset_index(drop=True)
    n_rates = len(rate_df)
    print(f"  {n_rates:,} unique rate_codes")

    # Skip if already done
    if qdrant.collection_exists(collection):
        existing = qdrant.count(collection).count
        if existing == n_rates:
            print(f"  ✓ Collection already has {existing:,} points — skipping encoding")
            return _save_snapshot(country, parquet_path, qdrant, collection)

    # Classify (RU propagation for non-RU countries)
    if country == "RU":
        ru_cache = None
        print(f"[3/5] Classifying {n_rates:,} rates (native RU) ...")
    else:
        ru_cache = get_ru_classifications()
        print(f"[3/5] Classifying {n_rates:,} rates (propagated from RU) ...")
    rows_info = []
    n_with_ifc = 0
    n_propagated = 0
    t_clf = time.time()
    for idx, r in enumerate(rate_df.itertuples(index=False)):
        rd = r._asdict()
        if ru_cache is None:
            e = enrich_v3(rd)
        else:
            bc = base_code(rd["rate_code"])
            if bc in ru_cache:
                e = ru_cache[bc]
                n_propagated += 1
            else:
                e = enrich_v3(rd)
        if e.get("ifc_class"):
            n_with_ifc += 1
        wc = wc_lookup.get(rd["rate_code"], [])
        res = res_lookup.get(rd["rate_code"], [])
        text = compose_bim(rd, e, wc, res)
        payload = {
            "rate_code": str(rd["rate_code"]), "country": country,
            "category_type": rd.get("category_type"),
            "collection_name": rd.get("collection_name"),
            "department_code": rd.get("department_code"),
            "subsection_code": rd.get("subsection_code"),
            "masterformat_division": rd.get("masterformat_division"),
            "rate_unit": rd.get("rate_unit"),
            "is_abstract": bool(rd.get("is_abstract") or False),
            "is_machine": bool(rd.get("is_machine") or False),
            "is_material": bool(rd.get("is_material") or False),
            "csi_division_2": e["csi_division_2"], "unit_type": e["unit_type"],
            "equipment_class": e["equipment_class"], "ifc_class": e["ifc_class"],
            "ifc_predefined_type": e["ifc_predefined_type"],
            "ost_category": e["ost_category"],
            "applies_to_ifc_classes": e["applies_to_ifc_classes"],
            "material_class": e["material_class"],
            "nominal_size_mm": e["nominal_size_mm"],
            "installation_method": e["installation_method"],
            "construction_stage": e["construction_stage"],
            "uniformat_group": e["uniformat_group"],
            "is_external": e["is_external"], "is_loadbearing": e["is_loadbearing"],
            "is_structural": e["is_structural"], "is_finishing": e["is_finishing"],
            "is_temporary": e["is_temporary"], "is_compound": e["is_compound"],
            "classification_confidence": e["classification_confidence"],
        }
        payload = {k: v for k, v in payload.items()
                    if v is not None and v != "" and v != []}
        rows_info.append({
            "rate_code": str(rd["rate_code"]),
            "text": text, "payload": payload,
        })
        if (idx + 1) % 10000 == 0:
            print(f"    {idx+1:,}/{n_rates:,} in {time.time()-t_clf:.0f}s")
    pct_prop = n_propagated/n_rates*100 if n_rates else 0
    pct_ifc = n_with_ifc/n_rates*100 if n_rates else 0
    print(f"  classified in {time.time()-t_clf:.0f}s "
          f"({n_with_ifc:,} = {pct_ifc:.1f}% with ifc_class, "
          f"{n_propagated:,} = {pct_prop:.1f}% propagated from RU)")

    # Encode
    print(f"[4/5] Encoding {n_rates:,} texts ...")
    t_enc = time.time()
    out = model.encode([info["text"] for info in rows_info],
                        batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
                        return_dense=True, return_sparse=True,
                        return_colbert_vecs=False)
    dense, sparse = out["dense_vecs"], out["lexical_weights"]
    print(f"  encoded in {time.time()-t_enc:.0f}s")

    # Upload
    print(f"[5/5] Uploading to '{collection}' ...")
    if qdrant.collection_exists(collection):
        qdrant.delete_collection(collection)
    qdrant.create_collection(
        collection_name=collection,
        vectors_config={"dense": VectorParams(size=EMBED_DIM, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams())},
    )
    INDEX_FIELDS = [
        ("rate_code", PayloadSchemaType.KEYWORD),
        ("country", PayloadSchemaType.KEYWORD),
        ("collection_name", PayloadSchemaType.KEYWORD),
        ("category_type", PayloadSchemaType.KEYWORD),
        ("department_code", PayloadSchemaType.KEYWORD),
        ("subsection_code", PayloadSchemaType.KEYWORD),
        ("masterformat_division", PayloadSchemaType.KEYWORD),
        ("csi_division_2", PayloadSchemaType.KEYWORD),
        ("unit_type", PayloadSchemaType.KEYWORD),
        ("equipment_class", PayloadSchemaType.KEYWORD),
        ("ifc_class", PayloadSchemaType.KEYWORD),
        ("ifc_predefined_type", PayloadSchemaType.KEYWORD),
        ("ost_category", PayloadSchemaType.KEYWORD),
        ("applies_to_ifc_classes", PayloadSchemaType.KEYWORD),
        ("material_class", PayloadSchemaType.KEYWORD),
        ("installation_method", PayloadSchemaType.KEYWORD),
        ("construction_stage", PayloadSchemaType.KEYWORD),
        ("uniformat_group", PayloadSchemaType.KEYWORD),
        ("nominal_size_mm", PayloadSchemaType.INTEGER),
        ("classification_confidence", PayloadSchemaType.KEYWORD),
        ("is_abstract", PayloadSchemaType.BOOL),
        ("is_machine", PayloadSchemaType.BOOL),
        ("is_material", PayloadSchemaType.BOOL),
        ("is_external", PayloadSchemaType.BOOL),
        ("is_loadbearing", PayloadSchemaType.BOOL),
        ("is_structural", PayloadSchemaType.BOOL),
        ("is_finishing", PayloadSchemaType.BOOL),
        ("is_temporary", PayloadSchemaType.BOOL),
        ("is_compound", PayloadSchemaType.BOOL),
    ]
    for f, schema in INDEX_FIELDS:
        try:
            qdrant.create_payload_index(collection, field_name=f, field_schema=schema)
        except Exception:
            pass

    # Smaller batches (50 vs 200) to avoid overwhelming Qdrant under sustained load
    UPSERT_BATCH_LOCAL = 50
    t_up = time.time()
    for i in range(0, n_rates, UPSERT_BATCH_LOCAL):
        chunk = rows_info[i:i+UPSERT_BATCH_LOCAL]
        points = []
        for j, info in enumerate(chunk):
            pid = str(uuid.uuid5(ID_NAMESPACE, f"{country}|{info['rate_code']}"))
            dvec = dense[i + j]
            svec = sparse[i + j]
            points.append(PointStruct(
                id=pid,
                vector={
                    "dense": list(map(float, dvec)),
                    "sparse": SparseVector(
                        indices=[int(k) for k in svec.keys()],
                        values=[float(x) for x in svec.values()],
                    ),
                },
                payload=info["payload"],
            ))
        qdrant.upsert(collection_name=collection, points=points)
        # Brief pause every 50 batches to let Qdrant index in background
        if (i // UPSERT_BATCH_LOCAL) % 50 == 49:
            time.sleep(2)
    print(f"  uploaded in {time.time()-t_up:.0f}s")

    # Save snapshot + push to GitHub
    snap_path = _save_snapshot(country, parquet_path, qdrant, collection)
    pushed = _git_push_snapshot(country, snap_path)

    if not pushed:
        print(f"  ⚠ KEEPING local snapshot and Qdrant collection — push failed, "
              f"data must persist for retry")
        print(f"\n✗ {country} encoded but NOT pushed in {(time.time()-t_start)/60:.1f} min")
        return

    # CRITICAL: free Qdrant disk+memory by deleting collection after snapshot persisted on disk+git.
    # Snapshot can be re-imported any time from GitHub LFS.
    print(f"  freeing Qdrant: deleting {collection} (snapshot on GitHub)")
    try:
        qdrant.delete_collection(collection)
    except Exception as e:
        print(f"  ⚠ could not delete collection: {e}")

    # Delete LOCAL snapshot file too — it's safe on GitHub LFS, frees host disk.
    try:
        snap_path.unlink()
        print(f"  removed local {snap_path.name} (still on GitHub LFS)")
    except Exception as e:
        print(f"  ⚠ could not remove local snapshot: {e}")

    # Prune local LFS object cache: pushed objects sit in .git/lfs/objects/ even
    # after working tree deletion (HEAD pointer keeps them retained). Free them
    # since they're confirmed on GitHub remote.
    try:
        import subprocess
        lfs_dir = ROOT / ".git" / "lfs" / "objects"
        if lfs_dir.exists():
            now = time.time()
            freed = 0
            for f in lfs_dir.rglob("*"):
                if f.is_file() and f.stat().st_size > 100 * 1024 * 1024:
                    age_min = (now - f.stat().st_mtime) / 60
                    if age_min > 5:  # don't touch the just-pushed object
                        sz = f.stat().st_size
                        f.unlink()
                        freed += sz
            if freed > 0:
                print(f"  pruned LFS cache: freed {freed/1024/1024:.0f}MB")
    except Exception as e:
        print(f"  ⚠ LFS prune failed: {e}")

    print(f"\n✓ {country} done in {(time.time()-t_start)/60:.1f} min")


def _save_snapshot(country: str, parquet_path: Path, qdrant, collection: str):
    """Trigger Qdrant snapshot, download to country folder."""
    print(f"  Creating snapshot ...")
    snap = qdrant.create_snapshot(collection_name=collection)
    snap_name = snap.name
    target = parquet_path.parent / derive_snapshot_name(parquet_path)
    url = f"{QDRANT_URL}/collections/{collection}/snapshots/{snap_name}"
    print(f"  Downloading {snap_name} → {target.name} ...")
    urllib.request.urlretrieve(url, target)
    size_mb = target.stat().st_size / 1024 / 1024
    print(f"  Saved snapshot: {size_mb:.0f}MB")
    return target


def _git_push_snapshot(country: str, snapshot_path: Path) -> bool:
    """Stage, commit, push the snapshot to GitHub via LFS.
    Returns True on successful push, False otherwise.
    """
    import subprocess
    repo = ROOT
    rel = snapshot_path.relative_to(repo).as_posix()
    print(f"  git add + commit + push {rel} ...")

    # Pre-check: ensure LFS filter applies to this path. If not, abort —
    # otherwise we'd commit a 400MB raw blob that GitHub will reject.
    chk = subprocess.run(["git", "-C", str(repo), "check-attr", "filter", rel],
                         capture_output=True, text=True)
    if "filter: lfs" not in (chk.stdout or ""):
        print(f"  ⚠ LFS filter NOT set for {rel} (got: {chk.stdout.strip()}). "
              f"Aborting push — fix .gitattributes first.")
        return False

    msg = (
        f"data({snapshot_path.stem.split('_workitems')[0]}): "
        f"BGE-M3 V3 vector DB snapshot (1024d, 55719 points, hybrid dense+sparse)\n\n"
        f"V3 collection cwicr_{country.lower()}_v3 with 29 indexed payload fields.\n"
        f"Classifications propagated from RU via base_code() lookup.\n\n"
        f"Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
    )
    try:
        subprocess.run(["git", "-C", str(repo), "add", rel],
                        check=True, capture_output=True, text=True)

        # Verify the staged blob is an LFS pointer (~135 bytes), not raw 400MB.
        sz = subprocess.run(["git", "-C", str(repo), "cat-file", "-s", f":{rel}"],
                             check=True, capture_output=True, text=True)
        staged_size = int(sz.stdout.strip())
        if staged_size > 10000:
            print(f"  ⚠ staged blob is {staged_size} bytes — NOT an LFS pointer. "
                  f"Aborting before commit.")
            subprocess.run(["git", "-C", str(repo), "reset", "HEAD", rel],
                           capture_output=True, text=True)
            return False

        subprocess.run(["git", "-C", str(repo), "commit", "-m", msg],
                        check=True, capture_output=True, text=True)
        out = subprocess.run(["git", "-C", str(repo), "push", "origin", "main"],
                              check=True, capture_output=True, text=True, timeout=900)
        last_line = (out.stderr or out.stdout).strip().split("\n")[-1]
        print(f"  ✓ pushed: {last_line}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ⚠ git error: {e.stderr or e.stdout}")
        return False
    except subprocess.TimeoutExpired as e:
        print(f"  ⚠ git push timed out after {e.timeout}s")
        return False
    except subprocess.TimeoutExpired:
        print(f"  ⚠ git push timed out (>15min) — continuing")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="Encode only this country code (e.g. EN)", default=None)
    args = ap.parse_args()

    countries = discover_countries()
    print(f"Found {len(countries)} countries:")
    for c, p in countries:
        marker = " (skip)" if c in SKIP_COUNTRIES else ""
        print(f"  {c}: {p.name}{marker}")

    if args.only:
        only = args.only.upper()
        countries = [(c, p) for c, p in countries if c == only]
        if not countries:
            print(f"ERROR: country '{only}' not found")
            return 1
        print(f"\n--only {only}: encoding 1 country.\n")
    else:
        countries = [(c, p) for c, p in countries if c not in SKIP_COUNTRIES]
        print(f"\nWill encode {len(countries)} countries.\n")

    # Load model once
    print("Loading BGE-M3 (one time) ...")
    import torch
    from FlagEmbedding import BGEM3FlagModel
    use_cuda = torch.cuda.is_available()
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_cuda,
                            device="cuda" if use_cuda else "cpu")
    print(f"  device={'cuda' if use_cuda else 'cpu'} fp16={use_cuda}\n")

    from qdrant_client import QdrantClient
    qdrant = QdrantClient(QDRANT_URL, timeout=600, check_compatibility=False)

    t_total = time.time()
    for i, (country, parquet) in enumerate(countries, 1):
        print(f"\n[{i}/{len(countries)}] {country}")
        # Retry up to 3 times with exponential backoff if Qdrant flakes
        attempts = 0
        max_attempts = 3
        while attempts < max_attempts:
            attempts += 1
            try:
                encode_country(country, parquet, model, qdrant)
                break
            except Exception as e:
                err = str(e)[:120]
                print(f"  ⚠ ERROR for {country} (attempt {attempts}/{max_attempts}): {err}")
                if attempts >= max_attempts:
                    print(f"  ✗ giving up on {country}")
                    break
                # Re-init Qdrant client + wait for recovery
                wait = 60 * attempts
                print(f"  sleeping {wait}s, then re-checking Qdrant ...")
                time.sleep(wait)
                try:
                    qdrant.get_collections()  # probe
                    print(f"  Qdrant responsive again, retrying ...")
                except Exception as probe_err:
                    print(f"  Qdrant still down ({probe_err}); reconnecting client ...")
                    qdrant = QdrantClient(QDRANT_URL, timeout=600, check_compatibility=False)
        # cooldown between countries to let Qdrant settle
        print(f"  cooldown 30s ...")
        time.sleep(30)

    print(f"\n{'='*70}")
    print(f"ALL DONE — total {(time.time()-t_total)/60:.0f} min")
    print(f"{'='*70}")


if __name__ == "__main__":
    sys.exit(main())
