#!/usr/bin/env bash
# Llama-3.2-1B split across the Mac (layers 0-7, rank 0) and the iPad (layers 8-15, rank 1).
# Needs build/ipad_shard.safetensors copied to the app's Documents (see README steps).
# Usage: mac/run_llama.sh [mac_ip ipad_ip]   (defaults: the USB link)
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

ENVIRONMENT=$(printf '{"SPIKE_MODE":"llama","SPIKE_RANK":"1","SPIKE_HOSTFILE_JSON":%s}' \
  "$(printf '%s' "$HOSTFILE_JSON" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")

MLX_HOSTFILE="$HOSTFILE" MLX_RANK=0 "$PYTHON" mac/llama_split.py &
MAC_PID=$!
sleep 2
xcrun devicectl device process launch --device "$DEVICE" --terminate-existing --console \
  --environment-variables "$ENVIRONMENT" com.eaglstun.ExoIPadSpike &
IPAD_PID=$!
wait "$MAC_PID"
wait "$IPAD_PID" || true
