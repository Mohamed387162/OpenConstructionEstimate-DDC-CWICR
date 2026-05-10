# V3 CWICR Multi-Language Encoder — Handover Document

**Date written:** 2026-05-10
**Last verified state:** CS encoding in progress (3/23 in restarted queue)
**Project root:** `C:\Users\Artem Boiko\Desktop\CodeProjects\legal-restructure-2026-04\OpenConstructionEstimate-DDC-CWICR\`
**Pipeline folder:** `0_Workflow and Pipelines CWICR\python\10-embedding-pipeline\`

---

## Mission

Encode 30 multilingual CWICR construction-rate parquets (~55K rates × 30 langs) into Qdrant V3 collections using BGE-M3 (1024d dense + sparse), and push each as a snapshot to GitHub via Git LFS for distribution.

**Required quality:** 100% rate_code coverage (every rate gets classification propagated from RU via `base_code()` lookup).

---

## Current Status (live)

### Done — 16 V3 snapshots on GitHub LFS (✓ at 100% coverage)

| Country | Encoded by | Notes |
|---------|------------|-------|
| RU      | First run  | Native (source of classification) |
| EN      | First run  | INLINE_LANG_TAGS extension already applied |
| AR      | RESTART 2 | Re-encoded 2026-05-10 with extended tags (was 98.18%, now 100%) |
| AU, ES, ID, JA, KO, MX | First run | Were already 100% with original tags |
| BG, CS*, DE, FR, HI, HR, IT | First run | **OLD versions still on GitHub** — re-encode in progress |

*CS old version IS on GitHub but is being replaced right now.

**Note:** the OLD GitHub commits for BG/DE/FR/HI/HR/IT are still pointed-to by LFS, even though they have <100% coverage. They will be OVERWRITTEN as each one is re-encoded (each new snapshot push creates a new commit with a new LFS object).

### Still to be encoded — 20 countries remaining in queue

After CS completes, queue order (alphabetical):
`DE → FR → HI → HR → IT → MX → NG → NL → NZ → PL → PT → RO → SV → TH → TR → UK → US → VI → ZA → ZH`

**MX caveat:** MX is already 100% on GitHub (encoded with old tags) but is in the re-encode queue. Adding MX to `SKIP_COUNTRIES` would save ~50 min.

### Currently running

- **Background task ID:** `bacfjeiks` (started 2026-05-10 ~23:30)
- **Process:** `python -u v3_production_all_languages.py` piped to `v3_production_all_run.log`
- **Current step:** CS encoding (step 4/5)
- **Pace:** ~50 min/country sequential

---

## Critical Fixes Already Applied (DO NOT UNDO)

### 1. INLINE_LANG_TAGS extension — `ost_ifc_mapping.py:493`

Added 18 language-specific inline tags so `base_code()` strips them and codes align with RU. **Verified: 0 unmapped codes across all 29 non-RU languages.**

```python
INLINE_LANG_TAGS = (
    "КАП", "ПУС", "MAJ", "COM", "MIN", "REM", "REP", "MOD", "EKS", "EXP",  # RU/EN
    "INB", "GRU",            # 10 European langs (BG/CS/DE/HR/IT/NL/PL/RO/SV/TR)
    "RÉN",                   # FR
    "REF",                   # PT
    "निर", "प्र", "उपक",     # HI (Hindi/Devanagari)
    "建筑工", "调试工", "设备大",  # ZH (Chinese)
    "أعم", "إصل",            # AR (Arabic)
)
```

### 2. Per-folder `.gitattributes` LFS bypass fix

19 per-country `.gitattributes` files (AU/BG/CS/HR/ID/IT/JA/KO/MX/NG/NL/NZ/PL/RO/SV/TH/TR/VI/ZA) used to override the root `*.snapshot filter=lfs` rule with `*.snapshot -filter`. This caused 400MB snapshots to be committed as raw blobs and rejected by GitHub. **Fixed in commit `ecdcc18`** — removed the `*.snapshot` line, kept `*.pdf` line for placeholder PDFs.

### 3. Encoder hardening — `v3_production_all_languages.py:_git_push_snapshot`

- Pre-check `git check-attr filter` returns `lfs` → else abort
- Post-`git add` verify staged blob size <10KB (LFS pointer, not raw 400MB) → else `reset` and abort
- Returns `bool`; caller skips Qdrant deletion + local-file deletion when push failed
- After successful push: prunes `.git/lfs/objects/` for files >100MB and older than 5 min (frees disk; objects re-fetchable from remote)

### 4. SKIP_COUNTRIES — `v3_production_all_languages.py:34`

```python
SKIP_COUNTRIES = {"RU", "EN", "AU", "ES", "ID", "JA", "KO"}
```

These are KNOWN 100% coverage in their pushed snapshots. AR was here originally but had to be removed for the re-encode (now done at 100%).

---

## ⚠️ Pipelined Encoder — REFACTORED but NOT YET DEPLOYED

I refactored the encoder to pipeline GPU encoding with IO upload+push (saves ~4h total), but the running encoder is the OLD sequential version. The refactored code is in the SAME file (`v3_production_all_languages.py`) — committed but not running yet.

**The refactor splits `encode_country()` into:**
- `encode_phase_gpu(country, parquet, model, qdrant)` — GPU work, returns dict with vectors
- `encode_phase_io(gpu_result, qdrant)` — upload, snapshot, push, cleanup

**`main()` runs an `io_worker` thread** consuming a `Queue(maxsize=1)`. While IO worker uploads/pushes country N, main thread starts encoding country N+1 on GPU. RAM cost: ~700MB extra (one extra country's vectors in memory).

**To deploy the pipelined version (saves ~4h on remaining 20 countries):**

1. Wait for current cycle (CS) to push successfully — do NOT kill mid-push
2. During the 30s `cooldown` between countries, kill the encoder:
   ```powershell
   Get-Process -Name python | Where-Object { $_.StartTime -gt (Get-Date).AddHours(-12) } | Stop-Process -Force
   ```
3. Update `SKIP_COUNTRIES` to add already-completed countries:
   ```python
   SKIP_COUNTRIES = {"RU", "EN", "AU", "ES", "ID", "JA", "KO", "AR", "BG", "CS"}
   # Also add MX if you don't want to re-encode it (saves 50min)
   ```
4. Restart in background:
   ```bash
   cd "C:/Users/Artem Boiko/Desktop/CodeProjects/legal-restructure-2026-04/OpenConstructionEstimate-DDC-CWICR/0_Workflow and Pipelines CWICR/python/10-embedding-pipeline" && PYTHONIOENCODING=utf-8 python -u v3_production_all_languages.py 2>&1 | tee -a v3_production_all_run.log
   ```

**Risk if pipelined version has bugs:** GPU phase will keep working but IO worker may swallow exceptions. Monitor log for "IO worker exception". Worst case, revert to sequential by reading prior commits in `v3_production_all_languages.py`.

---

## Action Plan for New Agent

### Immediate (within first hour of session)

1. **Check encoder is still running:**
   ```bash
   tail -50 "0_Workflow and Pipelines CWICR/python/10-embedding-pipeline/v3_production_all_run.log" | grep -vE "(pre tokenize|Inference Embeddings|it/s|Fetching)" | tail -20
   ```
   - If it finished or crashed, skip to "Restart" below.
   - If running, check which country and step it's on.

2. **Verify GitHub state:**
   ```bash
   cd "<repo>" && git lfs ls-files | grep BGEM3_V3 | wc -l
   ```
   - Should equal (16 + countries completed since this handover).

3. **Check disk and Qdrant:**
   ```powershell
   Get-PSDrive C | Select-Object @{n='FreeGB';e={[math]::Round($_.Free/1GB,2)}}
   ```
   ```bash
   curl -s http://localhost:6333/collections
   docker stats --no-stream 0_workflowandpipelinescwicr-qdrant-1
   ```
   - Disk should be >10 GB free. If <5 GB, run LFS prune (see "Disk management" section).
   - Qdrant should have `cwicr_ru` (V1, untouched) + maybe one `cwicr_<lang>_v3` mid-encode.

### Decision: deploy pipelined version or keep sequential?

- **If user is OK with ~13h more wait:** deploy pipelined version per "To deploy" steps above. **Do this BETWEEN cycles, never mid-cycle.**
- **If user prefers no risk:** keep sequential version running (current). Just monitor.

### Monitoring rhythm

- Wake up every 60–90 min
- Per check: log tail, V3 count, disk, Qdrant collections
- After every 2-3 successful pushes, run LFS prune if disk drops below 12 GB:
  ```bash
  cd "<repo>" && find .git/lfs/objects -type f -size +100M -mmin +30 -delete
  ```

### Final verification (when all 30 done)

```bash
cd "<repo>" && git lfs ls-files | grep BGEM3_V3 | wc -l
# Expected: 30
```

For each, you can spot-check coverage by re-running classification on the parquet vs RU bases — script template in `coverage_full.txt` (in pipeline folder).

---

## Known Gotchas

### A. Qdrant slowdown over time

After ~6 hours of continuous use, Qdrant upload time can grow from 380s → 1700s for a single collection. GPU memory is fine (1-2 GB used), Qdrant container memory is fine (~1 GB used out of 15 GB). Cause is likely segment optimization in background. **Don't restart Qdrant mid-cycle** — collection in flight will be lost. If restart needed, do it during the 30s cooldown between countries.

### B. GPU temperature

RTX 2050 (4 GB VRAM, mobile). Hits 74°C steady state. Throttles around 87°C. If `nvidia-smi` shows temp >82°C consistently, encoding will slow down. Pause if needed (can SIGSTOP the python process).

### C. LFS objects accumulate

Each pushed snapshot leaves a 400MB blob in `.git/lfs/objects/`. `git lfs prune` doesn't help (HEAD pointers retain them). Manual `find ... -delete` works. Encoder does this auto, but only for files older than 5 min — manual prune fills the gaps.

### D. Per-country `.gitattributes` overrides

These are FIXED in commit `ecdcc18` — DO NOT recreate them with `*.snapshot -filter` rules. The current `.gitattributes` files keep `*.pdf -filter` only.

### E. Docker VHD bloat (informational)

`%LOCALAPPDATA%\Docker\wsl\disk\docker_data.vhdx` is 165 GB on this machine. Won't grow naturally during encoding (Qdrant stores under bind-mount, not VHD). But background Docker activity can fill it.

### F. SKIP_COUNTRIES hygiene

Always update `SKIP_COUNTRIES` before restart to skip already-100% countries. Otherwise wasted GPU time. **Current value:** `{"RU", "EN", "AU", "ES", "ID", "JA", "KO"}`. After AR/BG/CS done, also add those. Don't add MX unless you've checked the GitHub snapshot is at 100% (it is — encoded with old tags but had no inline tag issues).

---

## File Inventory (key files)

| File | Purpose |
|------|---------|
| `v3_production_all_languages.py` | Main encoder. Pipelined refactor done but still SEQUENTIAL when started. |
| `v3_production_ru.py` | Original RU-only encoder; provides constants (`BATCH_SIZE`, `EMBED_DIM`, `compose_bim`) |
| `v3_classifier.py:enrich_v3` | Classification logic — 29 indexed payload fields |
| `ost_ifc_mapping.py:base_code` | Cross-language rate_code stripping (LANG_SUFFIXES + UNIT_SUFFIXES + INLINE_LANG_TAGS) |
| `v3_production_all_run.log` | Live encoder log — append-only, ~700 KB |
| `v3_production_quality_check.py` | Cross-language smoke test (NOT YET RUN — held back to avoid GPU contention) |
| `SNAPSHOT_RESTORE.md` | User-facing guide for downloading + restoring snapshots from GitHub LFS |
| `MAPPING_PROCESS.md` | Full BIM → CWICR mapping pipeline doc |
| `coverage_full.txt`, `coverage_after_fix.txt`, `coverage_final.txt` | Coverage audit results — useful templates |
| `ar_diag.txt` | Diagnostic of AR unmapped codes (showed أعم/إصل tags) |
| `HANDOVER.md` | This file |

---

## Quick Recovery Recipes

### If Qdrant container hangs
```powershell
# Restart Docker Desktop entirely (Qdrant container survives restart)
Stop-Process -Name "Docker Desktop","com.docker.backend" -Force
wsl --shutdown
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
# Wait 60s for Docker to come up, then:
docker start 0_workflowandpipelinescwicr-qdrant-1
```

### If git push fails for a single country
- Check `.gitattributes` for that country's folder — must NOT have `*.snapshot -filter`
- Run `git check-attr filter <path-to-snapshot>` — must return `filter: lfs`
- The encoder will print `⚠ git error: ...` and KEEP the local snapshot. Re-run encoder; it will re-snapshot (skipping the encoding step since collection still exists in Qdrant).

### If you accidentally delete a local snapshot before push confirmed
- Recoverable! Re-run encoder for that country. It will re-snapshot from Qdrant collection (if collection still exists) or re-encode from scratch.

### If encoder process is in zombie state
```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like "*v3_production*" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

---

## Recent Git Commits (for context)

```
905758b data(AR_DUBAI): BGE-M3 V3 vector DB snapshot (re-encode at 100%)
f876ddb fix(cwicr): extend INLINE_LANG_TAGS for 100% rate_code coverage across all 30 languages
1febe5b data(KO_SEOUL): BGE-M3 V3 vector DB snapshot
... (12 more single-country snapshot commits)
ecdcc18 fix(.gitattributes): enable LFS for V3 snapshots in 19 per-country folders
```

---

## Memory references (auto-memory system)

User has a `~/.claude/projects/.../memory/` folder with:
- `user_role.md` — Artem Boiko (DDC), CWICR + OpenConstructionERP work
- `feedback_quality_first.md` — validate via quality_eval.py before scaling
- `feedback_qdrant_stability.md` — retry+cooldown+reconnect pattern
- `feedback_git_lfs_attributes.md` — verify LFS filter before pushing
- `project_cwicr_v2.md` — V3-bim chosen as production winner

Read these when starting the session for context on the user's preferences and prior decisions.
