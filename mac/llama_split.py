"""Rank 0 of a two-stage Llama-3.2-1B pipeline: the Mac runs the embedding, layers
0..SPLIT-1, the final norm, the head and sampling; the iPad (rank 1, LlamaShard.swift)
runs layers SPLIT..15.

First it generates greedily with the whole model on the Mac alone, as the reference.
Then it generates the same prompt split across both devices and compares tokens.

Per step the Mac sends an int32 [L] header then a [1, L, 2048] float16 hidden state, and
receives the iPad's output of the same shape. L = 0 tells the iPad to stop.
"""

import os
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache

MODEL = Path.home() / ".exo/models/mlx-community--Llama-3.2-1B-Instruct-4bit"
SPLIT = int(os.environ.get("SPIKE_SPLIT", "8"))
MAX_TOKENS = int(os.environ.get("SPIKE_MAX_TOKENS", "128"))
PROMPT = os.environ.get(
    "SPIKE_PROMPT", "Write a short poem about an iPad helping a laptop think."
)


def log(line: str) -> None:
    print(f"[mac] {line}", flush=True)


def run_layers(hidden, layers, caches):
    mask = create_attention_mask(hidden, caches[0])
    for layer, cache in zip(layers, caches):
        hidden = layer(hidden, mask, cache=cache)
    return hidden


def generate(model, prompt_tokens, eos, middle):
    """Greedy generation; `middle(hidden, is_prefill)` runs layers SPLIT..15."""
    inner = model.model
    caches = make_prompt_cache(model)[:SPLIT]
    tokens: list[int] = []
    step_input = mx.array([prompt_tokens])
    started = time.perf_counter()
    first_token_at = 0.0
    for step in range(MAX_TOKENS):
        hidden = run_layers(inner.embed_tokens(step_input), inner.layers[:SPLIT], caches)
        hidden = middle(hidden, step == 0)
        logits = inner.embed_tokens.as_linear(inner.norm(hidden[:, -1:, :]))
        token = mx.argmax(logits, axis=-1).item()
        if step == 0:
            first_token_at = time.perf_counter()
        tokens.append(token)
        if token in eos:
            break
        step_input = mx.array([[token]])
    finished = time.perf_counter()
    decode_rate = (len(tokens) - 1) / (finished - first_token_at) if len(tokens) > 1 else 0
    return tokens, first_token_at - started, decode_rate


def main() -> None:
    model, tokenizer = load(str(MODEL))
    inner = model.model
    eos = set(tokenizer.eos_token_ids)
    prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True
    )
    log(f"prompt: {len(prompt_tokens)} tokens; split at layer {SPLIT} of {len(inner.layers)}")

    group = mx.distributed.init(backend="ring", strict=True)
    peer = 1 - group.rank()
    log(f"joined ring: rank {group.rank()} of {group.size()}")

    # Reference: the Mac runs every layer
    local_caches = make_prompt_cache(model)[SPLIT:]

    def local_middle(hidden, is_prefill):
        return run_layers(hidden, inner.layers[SPLIT:], local_caches)

    generate(model, prompt_tokens, eos, local_middle)  # warm up kernels
    local_caches = make_prompt_cache(model)[SPLIT:]
    reference, reference_ttft, reference_rate = generate(
        model, prompt_tokens, eos, local_middle
    )
    log(f"Mac alone: {len(reference)} tokens, TTFT {reference_ttft * 1000:.0f} ms, "
        f"{reference_rate:.1f} tok/s")

    # Split: layers SPLIT..15 on the iPad
    hop_seconds: list[float] = []
    prefill_pair: list[mx.array] = []

    def ipad_middle(hidden, is_prefill):
        hidden = hidden.astype(mx.float16)
        mx.eval(hidden)
        sent = time.perf_counter()
        mx.eval(mx.distributed.send(mx.array([hidden.shape[1]], dtype=mx.int32), peer, group=group))
        mx.eval(mx.distributed.send(hidden, peer, group=group))
        returned = mx.distributed.recv_like(hidden, peer, group=group)
        mx.eval(returned)
        hop_seconds.append(time.perf_counter() - sent)
        if is_prefill:
            prefill_pair.extend([hidden, returned])
        return returned

    split_tokens, split_ttft, split_rate = generate(model, prompt_tokens, eos, ipad_middle)
    mx.eval(mx.distributed.send(mx.array([0], dtype=mx.int32), peer, group=group))

    # Same layers on the Mac, fresh caches: how far apart are the two GPUs' numbers?
    sent_hidden, returned_hidden = prefill_pair
    check = run_layers(sent_hidden, inner.layers[SPLIT:], make_prompt_cache(model)[SPLIT:])
    difference = mx.abs(check.astype(mx.float32) - returned_hidden.astype(mx.float32))
    log(f"prefill: iPad vs Mac on layers {SPLIT}..15: max |diff| "
        f"{mx.max(difference).item():.4f}, mean {mx.mean(difference).item():.5f}, "
        f"max |value| {mx.max(mx.abs(check)).item():.1f}")

    decode_hops = hop_seconds[1:]
    log(f"Mac + iPad: {len(split_tokens)} tokens, TTFT {split_ttft * 1000:.0f} ms, "
        f"{split_rate:.1f} tok/s; iPad stage incl. network "
        f"{sum(decode_hops) / max(len(decode_hops), 1) * 1000:.2f} ms/token")

    matching = next(
        (i for i, (a, b) in enumerate(zip(reference, split_tokens)) if a != b),
        min(len(reference), len(split_tokens)),
    )
    if reference == split_tokens:
        log(f"tokens: IDENTICAL to Mac alone ({len(reference)})")
    else:
        log(f"tokens: first {matching} match, then diverge")
    log("text:\n" + tokenizer.decode(split_tokens))
    log("DONE")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
