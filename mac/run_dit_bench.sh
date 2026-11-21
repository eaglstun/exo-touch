#!/usr/bin/env bash
# Runs DiTBench on the iPad (FastFrames-LingBot DiT block shapes) and streams its log here.
# Compare with fastframes' scripts/bench_dit_block.py on the Mac. Usage: mac/run_dit_bench.sh
set -euo pipefail
DEVICE="${SPIKE_DEVICE:-00008142-000A05623E47801C}"
xcrun devicectl device process launch --device "$DEVICE" --terminate-existing --console \
  --environment-variables '{"SPIKE_MODE":"dit"}' com.eaglstun.ExoIPadSpike
