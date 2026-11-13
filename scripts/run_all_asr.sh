#!/usr/bin/env bash
# Reproduce the ASR evaluation exactly as reported in docs/results.md.
#
#   CTC models    -> full NICT-Tib1 test set (1,547 utts, fast)
#   Whisper models-> the common 300-utterance speaker-stratified sample
#   All models    -> OpenPecha 8-domain sample (160 clips)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
LOG=outputs/logs
mkdir -p "$LOG"

for m in wav2vec2-base openpecha-general; do
  echo "=== $m / nict (full test set) ==="
  $PY scripts/run_asr.py --model "$m" --dataset nict >> "$LOG/asr_${m}_nict.log" 2>&1 || echo "FAILED $m nict"
done

for m in whisper-tiny whisper-base whisper-small tenzin-small; do
  echo "=== $m / nict (300-sample) ==="
  $PY scripts/run_asr.py --model "$m" --dataset nict --limit 300 >> "$LOG/asr_${m}_nict.log" 2>&1 || echo "FAILED $m nict"
done

for m in wav2vec2-base openpecha-general whisper-small tenzin-small; do
  echo "=== $m / openpecha ==="
  $PY scripts/run_asr.py --model "$m" --dataset openpecha >> "$LOG/asr_${m}_openpecha.log" 2>&1 || echo "FAILED $m openpecha"
done

$PY scripts/score_asr.py
$PY scripts/summarize_asr.py --dataset both
