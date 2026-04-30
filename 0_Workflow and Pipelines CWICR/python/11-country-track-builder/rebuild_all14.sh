#!/usr/bin/env bash
# Rebuild all 14 translate tracks serially. Skip LLM during rebuild
# (cache populated by translate_new_cols.py). Skip embeddings.
# Skip IT_ROME — already rebuilt successfully in pass 1.
set -u
export PYTHONUNBUFFERED=1
export LLM_TOP_N=0
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
LOG="$HERE/logs/rebuild_all14.log"
mkdir -p "$(dirname "$LOG")"
TRACKS=(
  NL_AMSTERDAM SV_STOCKHOLM HR_ZAGREB
  CS_PRAGUE PL_WARSAW RO_BUCHAREST TR_ISTANBUL BG_SOFIA
  ID_JAKARTA VI_HANOI JA_TOKYO KO_SEOUL TH_BANGKOK
)
echo "=== rebuild_all14 v2 (LLM_TOP_N=0) start: $(date) ===" | tee -a "$LOG"
for t in "${TRACKS[@]}"; do
  echo "" | tee -a "$LOG"
  echo "=== $t @ $(date '+%H:%M:%S') ===" | tee -a "$LOG"
  python -u add_country_track.py --config "configs/$t.yaml" --skip-embeddings 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  echo "[$t exit=$rc]" | tee -a "$LOG"
done
echo "" | tee -a "$LOG"
echo "=== rebuild_all14 v2 done: $(date) ===" | tee -a "$LOG"
