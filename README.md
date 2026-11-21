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
