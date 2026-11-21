# exo iPad spike

Can an iPad Pro join an MLX distributed ring with a Mac and run part of a model?
Yes. mlx-swift 0.33.0 pins MLX v0.32.4, the same commit exo runs, so both ranks speak one ring protocol.

## Setup

    xcodegen generate
    xcodebuild -project ExoIPadSpike.xcodeproj -scheme ExoIPadSpike -configuration Release \
      -destination 'id=<device udid>' -derivedDataPath build -allowProvisioningUpdates build
    xcrun devicectl device install app --device <udid> build/Build/Products/Release-iphoneos/ExoIPadSpike.app

On the first run, iOS asks for Local Network access. Until you tap Allow, the ring fails with
`Couldn't connect (error: 60)`.

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
