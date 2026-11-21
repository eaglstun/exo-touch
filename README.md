# exo iPad spike

Can an iPad Pro join an MLX distributed ring with a Mac and run part of a model?
Yes. mlx-swift 0.33.0 pins MLX v0.32.4, the same commit exo runs, so both ranks speak one ring protocol.

## Setup

    mac/vendor_mlx_swift.sh    # once: patched mlx-swift 0.33.0 into Vendor/ (see "The patched MLX" below)
    xcodegen generate
    xcodebuild -project ExoIPadSpike.xcodeproj -scheme ExoIPadSpike -configuration Release \
      -destination 'id=<device udid>' -derivedDataPath build -allowProvisioningUpdates build
    xcrun devicectl device install app --device <udid> build/Build/Products/Release-iphoneos/ExoIPadSpike.app

On the first run, iOS asks for Local Network access. Until you tap Allow, the ring fails with
`Couldn't connect (error: 60)`.

### The patched MLX in `Vendor/`

The app builds against a local copy of mlx-swift, not the package from GitHub: `project.yml` points
`MLXSwift` at `Vendor/mlx-swift`, which git ignores. Run `mac/vendor_mlx_swift.sh` once per checkout
(it does nothing if `Vendor/mlx-swift` exists). It clones mlx-swift **0.33.0** with its submodules
(MLX 36969c0, v0.32.4, the commit exo runs; mlx-c v0.8.0) and applies
`patches/mlx-swift-0.33.0-sdpa-lse.patch`. Without it, `xcodegen`/`xcodebuild` can't find the
package; with an unpatched copy the build fails on `scaledDotProductAttentionWithLSE`.

**What the patch adds.** MLX's fused attention kernels (including the M5's NAX path) already compute
each query row's log-sum-exp, but only hand it out while differentiating. The patch exposes it for
inference, so SplitStage can attend over two sets of keys separately and merge them (section 5):

| layer | file | change |
|---|---|---|
| MLX C++ | `Source/Cmlx/mlx/mlx/fast.{h,cpp}` | `fast::scaled_dot_product_attention_lse(q, k, v, scale)` returns `{output, lse}`; lse is `[B, heads, L_q, 1]` float32, natural log. The existing function becomes a thin wrapper over a shared implementation with a `want_lse` flag, so its behaviour is unchanged. Throws if no fused kernel fits the inputs (no fallback). |
| mlx-c | `Source/Cmlx/mlx-c/mlx/c/fast.{h,cpp}` and the two header copies in `Source/Cmlx/include*/` | `mlx_fast_scaled_dot_product_attention_lse(&res, &lse, q, k, v, scale, stream)` |
| Swift | `Source/MLX/MLXFast.swift` | `MLXFast.scaledDotProductAttentionWithLSE(queries:keys:values:scale:stream:) -> (output, lse)` |

Unmasked attention only, which is all the split needs. `SPIKE_MODE=lsecheck` checks it on the device:
attention merged from two parts against one call, eager, compiled and async-evaluated.

**Changing the patch.** Edit the files in `Vendor/mlx-swift` (three git repos: mlx-swift and its
`Source/Cmlx/mlx` and `Source/Cmlx/mlx-c` submodules), rebuild, then regenerate the patch from all
three:

    V=Vendor/mlx-swift
    { git -C $V diff --ignore-submodules=all
      git -C $V/Source/Cmlx/mlx diff --src-prefix=a/Source/Cmlx/mlx/ --dst-prefix=b/Source/Cmlx/mlx/
      git -C $V/Source/Cmlx/mlx-c diff --src-prefix=a/Source/Cmlx/mlx-c/ --dst-prefix=b/Source/Cmlx/mlx-c/
    } > patches/mlx-swift-0.33.0-sdpa-lse.patch

Check that it applies to a fresh copy: `rm -rf Vendor/mlx-swift && mac/vendor_mlx_swift.sh`.

**Moving to a newer mlx-swift.** Keep MLX on the commit exo runs, since the ring needs one protocol on
every rank. Change the tag in `mac/vendor_mlx_swift.sh`, apply the patch to the new copy and fix any
rejects by hand, then save it as `patches/mlx-swift-<version>-sdpa-lse.patch` and update the script.
If upstream MLX ever exposes the log-sum-exp itself, drop the patch and point `project.yml` back at
the GitHub package (`url:` and `exactVersion:`).

## 1. Ring check: `mac/run_spike.sh [mac_ip ipad_ip]`

Join, allSum, an fp16 hop doubled on the iPad's GPU, decode round trips, bandwidth.
Over USB-C: 0.29 ms per round trip, 858 MB/s. Over Wi-Fi: 8.6 ms, 48 MB/s.

## 2. Llama-3.2-1B split: `mac/run_llama.sh [mac_ip ipad_ip]`

The Mac runs the embedding, layers 0-7 and sampling. The iPad runs layers 8-15 (`Sources/LlamaShard.swift`).

    ../exo/.venv/bin/python mac/make_shard.py 8 build/ipad_shard.safetensors
    xcrun devicectl device copy to --device <udid> --domain-type appDataContainer \
      --domain-identifier com.eaglstun.ExoIPadSpike \
      --source build/ipad_shard.safetensors --destination Documents/ipad_shard.safetensors
    mac/run_llama.sh

Greedy tokens are identical to the Mac running the whole model alone.
Decode: 168 tok/s split vs 400 tok/s on the Mac alone. Benchmark Release builds only: Debug compiles MLX at -O0.

## 3. FastFrames-LingBot DiT block: `mac/run_dit_bench.sh`

`SPIKE_MODE=dit` times one Wan 2.1 1.3B DiT block at the shapes of a FastFrames-LingBot demo chunk
(4 latent frames of queries against an 18-frame KV cache, BF16; `Sources/DiTBench.swift`). The Mac
runs the same shapes with `scripts/bench_dit_block.py` in fastframes (branch `ipad-dit-bench`).

| per block | iPad Pro 11" M5, MLX | M4 Max, PyTorch MPS (the demo today) | M4 Max, MLX |
|---|---|---|---|
| 832x464 (6,032 x 27,144 tokens) | 160.4 ms | 174.8 ms | 124.4 ms |
| 576x320 | 50.2 ms | 39.5 ms | 39.2 ms |
| 416x240 | 22.9 ms | 17.7 ms | 17.7 ms |

The M5's GEMMs run at 10.7-13.6 TFLOPS and its attention at 9.9-11.1, close to the M4 Max
(12.7-14.8 and 10.5-12.5). The link is the limit: the iPad joins the Mac as a USB device at
10 Gb/s (858 MB/s, 0.29 ms round trip), not over Thunderbolt networking, whatever the cable.

The synthetic block above concatenates the KV cache every time, which the real DiT doesn't (it
writes this chunk's rows in place), so those whole-block times run high on both devices.

## 4. The real block: `mac/run_dit_block.sh <capture>`

fastframes' `scripts/lingbot_block_capture.py` saves LingBot's real DiT block 0 (weights, inputs and
PyTorch's output) during a full-cache chunk; `Sources/DiTBlockCheck.swift` runs it here with the same
dtype rules as `scripts/lingbot_block_mlx.py` and compares. At 832x464:

| | output vs PyTorch (relative L2) | per block |
|---|---|---|
| M4 Max, PyTorch MPS (the demo) | reference | 144.1 ms |
| M4 Max, MLX compiled | 8.4e-4 | 127.4 ms |
| iPad Pro M5, MLX-Swift eager | 1.2e-3 | 209.7 ms |
| iPad Pro M5, MLX-Swift compiled | 1.2e-3 | 156.8 ms |

The iPad runs the real block at 92% of the Mac's speed (compiled; `compile` is worth 1.34x here)
in a burst. After ~20-27 s of continuous GPU load it drops to ~0.5-0.6x and stays there (section 5).

## 5. LingBot's DiT shared with the Mac: `mac/run_split_stage.sh`

`Sources/SplitStage.swift` (`SPIKE_MODE=split`, port 5570) is the iPad side of fastframes'
`apps/lingbot/ipad.py` (IPadAssist): every denoising pass's chunk tokens are split, this device runs
its share through all 30 blocks, and in each block both sides swap their tokens' canonical (pre-RoPE)
keys and values, rotate the other's, and attend over the whole cache. A session mirrors the Mac's
cache: weights once (2.6 GB in ~3.7 s; footprint 8.2 GB at 832x464), the prompt's text K/V, resets,
the Mac's eviction copies, each chunk's RoPE table and camera tokens (it computes its own camera
rows), and the refined rows the Mac commits after a chunk. The Mac moves the split between chunks
to where both finish a pass together; the iPad reports its thermal state with each pass.

fastframes' `scripts/lingbot_ipad_split.py` generates a 10-chunk world on the Mac alone, split, and
alone again: full-cache chunks 21.6 s split vs 27.3 s alone in the same session (1.26x; ~1.2x against
the Mac's quieter chunks), frames clean and the same scene, drifting in detail as an autoregressive
model does (PSNR 43 dB at chunk 0, ~20 dB by chunk 9). `demo.py --ipad <address>` runs the browser
demo this way at 832x464.

The gain is capped by the iPad's GPU throttling under sustained load, which `thermalState` doesn't
report (it stays nominal, thermal pressure 0). `mac/split_fake_mac.py` speaks this protocol with
random weights and no Mac model; with `--solo` (no traffic at all) a 2960-token pass over 27144
cache rows takes 2.5 s for ~20 s, then 4.0-4.4 s; the K/V exchange adds ~20% only once throttled.
About 30 s idle buys back ~10 s at full speed. Zero-copy sockets and idling the GPU between passes
(`SPIKE_NO_WARM=1`) changed nothing.

**Hiding the K/V exchange.** When the Mac asks (pass meta `"overlap"`), each block queues attention over
the cache's older rows (`asyncEval`) before sending its K/V, so it runs while the Mac's are in flight,
then attends over this chunk's rows and merges the two by their log-sum-exp. MLX computes that in its
fused attention for training only; `patches/mlx-swift-0.33.0-sdpa-lse.patch` exposes it as
`MLXFast.scaledDotProductAttentionWithLSE`, and `mac/vendor_mlx_swift.sh` fetches mlx-swift 0.33.0 into
`Vendor/` (ignored) and applies it. On the real block 0 the merged attention is 1.34e-3 from PyTorch
(1.24e-3 in one call) at 213.6 vs 210.1 ms eager; `SPIKE_MODE=lsecheck` checks the merge on random data
at the split's shapes. It didn't make the split faster (19.62 vs 19.77 s per full-cache chunk): the
critical path is the transfer both ways plus the rest of each block, not attention, and the iPad's
waiting just became attention. The exchange is ~60-70 ms per block both ways, mostly bytes (the Mac's
~24 MB K/V frame at the link's 858 MB/s); frames now go out without copying the arrays (no change),
and with `"kv_bits": 8` both sides send int8 K/V with an FP32 scale per token and head (51.6% of the
bytes; frames as good as 16-bit, no gain measured under throttling, so the Mac leaves it off). Each
pass reports `turn_ms`, from each Mac K/V's arrival to the next send here, so the Mac can measure the
exchange's latency. Details in fastframes' MAC.md.
