#!/usr/bin/env bash
# Starts exo's remote stage on the iPad (RemoteStageServer, port 5560) and streams its log here; then point an
# exo node or exo's stage tests at it with EXO_REMOTE_STAGE=<ipad address>:5560. Usage: mac/run_stage.sh
# STAGE_KEEP_WARM=0 lets the GPU idle between decode forwards; STAGE_TIMING=1 logs decode timing.
set -euo pipefail
DEVICE="${SPIKE_DEVICE:-00008142-000A05623E47801C}"
xcrun devicectl device process launch --device "$DEVICE" --terminate-existing --console \
  --environment-variables "{\"SPIKE_MODE\":\"stage\",\"STAGE_KEEP_WARM\":\"${STAGE_KEEP_WARM:-1}\",\"STAGE_TIMING\":\"${STAGE_TIMING:-0}\"}" \
  com.eaglstun.ExoIPadSpike
