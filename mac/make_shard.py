"""Writes the iPad's half of Llama-3.2-1B-Instruct-4bit: layers SPLIT..15, renamed to
layers.0.., plus the llama3 RoPE frequencies exactly as mlx_lm computes them, so the
Swift side never has to reimplement that scaling.

    ../exo/.venv/bin/python mac/make_shard.py [split] [out.safetensors]
"""

import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

MODEL = Path.home() / ".exo/models/mlx-community--Llama-3.2-1B-Instruct-4bit"


def main() -> None:
    split = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    out = sys.argv[2] if len(sys.argv) > 2 else "build/ipad_shard.safetensors"

    model, _ = load(str(MODEL))
    layer_count = len(model.model.layers)
    weights = mx.load(str(MODEL / "model.safetensors"))

    shard: dict[str, mx.array] = {}
    for name, array in weights.items():
        parts = name.split(".")
        if parts[:2] == ["model", "layers"] and int(parts[2]) >= split:
            shard[".".join(["layers", str(int(parts[2]) - split), *parts[3:]])] = array
    shard["rope_freqs"] = model.model.layers[0].self_attn.rope._freqs

    mx.save_safetensors(out, shard)
    size = sum(array.nbytes for array in shard.values()) / 1e6
    print(f"layers {split}..{layer_count - 1} -> {out}: {len(shard)} arrays, {size:.0f} MB")


if __name__ == "__main__":
    main()
