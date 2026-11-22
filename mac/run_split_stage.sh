#!/usr/bin/env bash
# Starts the iPad's share of a split FastFrames-LingBot DiT pass (SplitStage, port 5570) and streams its
# log here; then run fastframes' scripts/lingbot_ipad_split.py on the Mac. Usage: mac/run_split_stage.sh
set -euo pipefail
DEVICE="${SPIKE_DEVICE:?set SPIKE_DEVICE to the iPad UDID from xcrun devicectl list devices}"
xcrun devicectl device process launch --device "$DEVICE" --terminate-existing --console \
  --environment-variables '{"SPIKE_MODE":"split"}' "${SPIKE_BUNDLE_ID:-com.eaglstun.ExoIPadSpike}"
