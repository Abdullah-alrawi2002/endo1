#!/bin/sh
set -eu

# Railway / Render inject PORT. Next listens on it; CT stays on 8000 internal-only.
export PORT="${PORT:-3000}"
export ENABLE_CT_MODULE="${ENABLE_CT_MODULE:-true}"
export CT_SIDECAR_URL="${CT_SIDECAR_URL:-http://127.0.0.1:8000}"
export CT_ANALYSIS_STORE_PATH="${CT_ANALYSIS_STORE_PATH:-/data/ct-analyses}"
export CORRECTIONS_JSON_PATH="${CORRECTIONS_JSON_PATH:-/data/corrections.json}"

mkdir -p /data/ct-analyses /models

echo "Starting all-in-one Endodontic Agent (web :${PORT} + CT :8000)…"
exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/endo.conf
