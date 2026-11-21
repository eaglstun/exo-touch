# Patches

`mlx-swift-0.33.0-sdpa-lse.patch` is applied by `mac/vendor_mlx_swift.sh` to mlx-swift 0.33.0 in
`Vendor/mlx-swift`. It exposes the log-sum-exp from MLX's fused attention as
`MLXFast.scaledDotProductAttentionWithLSE`, which SplitStage uses to hide the K/V exchange. Apply it
from the mlx-swift root with `patch -p1`. It spans mlx-swift and its `Source/Cmlx/mlx` and
`Source/Cmlx/mlx-c` submodules. What it changes, how to regenerate it and how to move it to a newer
mlx-swift: README.md, "The patched MLX in `Vendor/`".
