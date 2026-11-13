#!/usr/bin/env bash
# Launch the local Tibetan OCR/ASR evaluation GUI.
#   ./scripts/gui.sh                 # http://127.0.0.1:8765 (opens browser)
#   ./scripts/gui.sh --port 9000
#   ./scripts/gui.sh --no-browser
set -euo pipefail
cd "$(dirname "$0")/.."

PORT=8765
OPEN=1
while [ $# -gt 0 ]; do
  case "$1" in
    --port)      PORT="$2"; shift 2 ;;
    --no-browser) OPEN=0; shift ;;
    -h|--help)   sed -n '2,5p' "$0"; exit 0 ;;
    *)           echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ "$OPEN" = "1" ]; then
  ( sleep 1.5; open "http://127.0.0.1:${PORT}" >/dev/null 2>&1 || true ) &
fi

exec ./.venv/bin/python gui/server.py --port "$PORT" --offline
