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

The iPad runs the real block at 92% of the Mac's speed (compiled; `compile` is worth 1.34x here).

## 5. A LingBot DiT pass split with the Mac: `mac/run_split_stage.sh`

`Sources/SplitStage.swift` (`SPIKE_MODE=split`, port 5570) runs a share of one denoising pass's
tokens through all 30 blocks, in step with fastframes' `scripts/lingbot_ipad_split.py` on the Mac:
in each block both sides send their tokens' keys and values, write both into their own KV cache,
and attend over all of it. Setup sends the weights, caches, text K/V and camera rows once (8.1 GB
in 11 s; the app's footprint is then 8.3 GB). While waiting for the Mac's K/V the GPU stays busy,
so it doesn't clock down. One pass at 832x464, 55% of the tokens here:

| | per pass |
|---|---|
| M4 Max alone (PyTorch MPS), quiet | 4.53 s |
| split with this iPad | 3.14-3.25 s (~1.4x) |
| M4 Max alone while Photos analyses media | ~7.8 s |
| split, same moment | ~3.7 s (~2.1x) |

The split pass matches the Mac alone to 4.9e-3 relative L2.
