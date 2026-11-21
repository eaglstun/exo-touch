#!/usr/bin/env bash
# Starts the iPad's share of a split FastFrames-LingBot DiT pass (SplitStage, port 5570) and streams its
# log here; then run fastframes' scripts/lingbot_ipad_split.py on the Mac. Usage: mac/run_split_stage.sh
set -euo pipefail
DEVICE="${SPIKE_DEVICE:-00008142-000A05623E47801C}"
xcrun devicectl device process launch --device "$DEVICE" --terminate-existing --console \
  --environment-variables '{"SPIKE_MODE":"split"}' com.eaglstun.ExoIPadSpike
