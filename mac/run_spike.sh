#!/usr/bin/env bash
# Runs both ranks: the iPad app (rank 1, launched over USB with devicectl) and the Mac
# script (rank 0). Usage: mac/run_spike.sh [mac_ip ipad_ip]
# Defaults are the USB link's link-local addresses; pass the Wi-Fi addresses to compare.
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${SPIKE_DEVICE:-00008142-000A05623E47801C}"
MAC_IP="${1:-169.254.147.232}"
IPAD_IP="${2:-169.254.154.199}"
PORT="${SPIKE_PORT:-5555}"
PYTHON="${SPIKE_PYTHON:-$HOME/Documents/dev/exo/.venv/bin/python}"

HOSTFILE_JSON="[[\"$MAC_IP:$PORT\"],[\"$IPAD_IP:$PORT\"]]"
HOSTFILE="$(mktemp -t mlx-hostfile).json"
echo "$HOSTFILE_JSON" > "$HOSTFILE"
echo "hostfile: $HOSTFILE_JSON"

ENVIRONMENT=$(printf '{"SPIKE_RANK":"1","SPIKE_HOSTFILE_JSON":%s}' \
  "$(printf '%s' "$HOSTFILE_JSON" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")

# The iPad streams its stdout here; the Mac rank runs alongside it
xcrun devicectl device process launch --device "$DEVICE" --terminate-existing --console \
  --environment-variables "$ENVIRONMENT" com.eaglstun.ExoIPadSpike &
IPAD_PID=$!
sleep 3
MLX_HOSTFILE="$HOSTFILE" MLX_RANK=0 "$PYTHON" mac/mac_rank.py
wait "$IPAD_PID" || true
