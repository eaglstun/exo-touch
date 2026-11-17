"""The Mac's half of the spike: rank 0 of a two-rank MLX ring whose rank 1 is the iPad app.

Every step mirrors a step in Sources/RingSpike.swift, in the same order. Run it with
exo's Python, so both ranks run MLX v0.32.4:

    MLX_HOSTFILE=hostfile.json MLX_RANK=0 ../exo/.venv/bin/python mac/mac_rank.py
"""

import time

import mlx.core as mx

PREFILL_SHAPE = (1, 12, 2048)
DECODE_SHAPE = (1, 1, 2048)
ROUND_TRIPS = 100
BANDWIDTH_ELEMENTS = 4 * 1024 * 1024
BANDWIDTH_MESSAGES = 20


def log(line: str) -> None:
    print(f"[mac] {line}", flush=True)


def main() -> None:
    log(f"device: {mx.default_device()}, mlx {mx.__version__}")

    # 1. Join the ring
    started = time.perf_counter()
    group = mx.distributed.init(backend="ring", strict=True)
    log(f"joined: rank {group.rank()} of {group.size()} in {time.perf_counter() - started:.2f}s")
    peer = 1 - group.rank()

    # 2. A collective: each rank adds rank + 1, so two ranks give 3
    summed = mx.distributed.all_sum(mx.ones(4) * (group.rank() + 1), group=group)
    log(f"allSum: {summed.tolist()} (expect 3s)")

    # 3. A prefill-shaped float16 hidden state: the iPad doubles it on its GPU
    hidden = (mx.arange(12 * 2048).reshape(PREFILL_SHAPE) % 7).astype(mx.float16)
    mx.eval(hidden)
    started = time.perf_counter()
    mx.eval(mx.distributed.send(hidden, peer, group=group))
    returned = mx.distributed.recv_like(hidden, peer, group=group)
    mx.eval(returned)
    correct = mx.array_equal(returned, hidden * 2).item()
    log(
        f"prefill hop: {time.perf_counter() - started:.3f}s round trip, "
        f"iPad's GPU result {'CORRECT' if correct else 'WRONG'}"
    )

    # 4. Decode-shaped round trips: the per-token cost a pipeline stage adds
    token = mx.ones(DECODE_SHAPE, dtype=mx.float16)
    mx.eval(token)
    started = time.perf_counter()
    for _ in range(ROUND_TRIPS):
        mx.eval(mx.distributed.send(token, peer, group=group))
        mx.eval(mx.distributed.recv_like(token, peer, group=group))
    round_trip = (time.perf_counter() - started) * 1000 / ROUND_TRIPS
    log(f"decode round trip: {round_trip:.3f} ms average over {ROUND_TRIPS}")

    # 5. Bandwidth: send 20 messages of 16 MB; the closing collective waits for the iPad
    #    to have received them all
    payload = mx.ones(BANDWIDTH_ELEMENTS, dtype=mx.float32)
    mx.eval(payload)
    started = time.perf_counter()
    for _ in range(BANDWIDTH_MESSAGES):
        mx.eval(mx.distributed.send(payload, peer, group=group))
    mx.eval(mx.distributed.all_sum(mx.ones(1), group=group))
    elapsed = time.perf_counter() - started
    megabytes = BANDWIDTH_MESSAGES * BANDWIDTH_ELEMENTS * 4 / 1_000_000
    log(f"bandwidth: {megabytes:.0f} MB in {elapsed:.2f}s = {megabytes / elapsed:.0f} MB/s")
    log("DONE")


if __name__ == "__main__":
    main()
