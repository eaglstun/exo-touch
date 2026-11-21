#!/usr/bin/env bash
# Fetches mlx-swift 0.33.0 (MLX 36969c0, the same commit exo runs) into Vendor/mlx-swift and applies
# patches/mlx-swift-0.33.0-sdpa-lse.patch: MLXFast.scaledDotProductAttentionWithLSE, the fused attention
# kernel's log-sum-exp (MLX computes it for training only), which SplitStage needs to attend over the
# cache's older rows while the Mac's K/V are in flight. Run once before xcodegen. Usage: mac/vendor_mlx_swift.sh
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET=Vendor/mlx-swift
if [ -d "$TARGET" ]; then
  echo "$TARGET exists; delete it to fetch again"
  exit 0
fi
git clone --quiet --depth 1 --branch 0.33.0 --recurse-submodules --shallow-submodules \
  https://github.com/ml-explore/mlx-swift "$TARGET"
patch --quiet -p1 -d "$TARGET" < patches/mlx-swift-0.33.0-sdpa-lse.patch
echo "patched $TARGET"
