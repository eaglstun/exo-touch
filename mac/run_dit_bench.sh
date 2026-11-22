#!/usr/bin/env bash
# Runs DiTBench on the iPad (FastFrames-LingBot DiT block shapes) and streams its log here.
# Compare with fastframes' scripts/bench_dit_block.py on the Mac. Usage: mac/run_dit_bench.sh
set -euo pipefail
DEVICE="${SPIKE_DEVICE:?set SPIKE_DEVICE to the iPad UDID from xcrun devicectl list devices}"
xcrun devicectl device process launch --device "$DEVICE" --terminate-existing --console \
  --environment-variables '{"SPIKE_MODE":"dit"}' "${SPIKE_BUNDLE_ID:-com.eaglstun.ExoIPadSpike}"
