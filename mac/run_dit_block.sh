#!/usr/bin/env bash
# Copies a block captured by fastframes' scripts/lingbot_block_capture.py to the iPad and checks it there
# (DiTBlockCheck: MLX-Swift vs PyTorch's output, and timing). Usage: mac/run_dit_block.sh <capture path without suffix>
set -euo pipefail
DEVICE="${SPIKE_DEVICE:-00008142-000A05623E47801C}"
STEM="${1:?usage: mac/run_dit_block.sh <path/to/block0 (without .safetensors)>}"
NAME="$(basename "$STEM")"
for suffix in safetensors json; do
  xcrun devicectl device copy to --device "$DEVICE" --domain-type appDataContainer \
    --domain-identifier com.eaglstun.ExoIPadSpike --source "$STEM.$suffix" \
    --destination "Documents/$NAME.$suffix" > /dev/null
done
xcrun devicectl device process launch --device "$DEVICE" --terminate-existing --console \
  --environment-variables "{\"SPIKE_MODE\":\"ditblock\",\"SPIKE_BLOCK\":\"$NAME\"}" com.eaglstun.ExoIPadSpike
