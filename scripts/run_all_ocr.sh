#!/usr/bin/env bash
# Run both OCR engines on the 60-page stratified sample.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
LOG=outputs/logs
mkdir -p "$LOG"

$PY scripts/run_ocr.py --engine yigdzin --split sample --use-dry >> "$LOG/ocr_yigdzin_sample.log" 2>&1
$PY scripts/run_ocr.py --engine tesseract --split sample >> "$LOG/ocr_tesseract_sample.log" 2>&1
$PY scripts/score_ocr.py
