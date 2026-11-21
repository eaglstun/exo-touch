#!/usr/bin/env python3
"""A stand-in for the Mac's side of a split DiT pass: SplitStage's protocol with random weights and no
model, so the iPad's pass times can be measured without the Mac's GPU in the picture.

Each pass runs at a fixed cache size (--rows). What it controls is *when* the Mac's keys and values
of a block reach the iPad:

  --delay S           the Mac's K/V of block i leave S seconds after the iPad's K/V of block i arrived:
                      the iPad is the faster side and waits for them (~40-55 ms a block even at S=0)
  --solo              no exchange at all: the same GPU work with no traffic or copies beside it
  --ahead --delay S   the Mac's K/V of block i+1 leave S seconds after the iPad's K/V of block i arrived:
                      the Mac is the faster side, its frames land while the iPad's GPU is computing

    mac/run_split_stage.sh &          # the iPad's stage
    python3 mac/split_fake_mac.py --passes 20 [--ahead] [--delay 0.02]

Needs numpy. The iPad's log has every pass's phases, a CPU probe and the thermal pressure.

What it found (iPad Pro 11" M5, 2960 tokens x 27144 rows): the iPad's GPU holds its benchmark speed for
about 20 s of continuous load (--solo: 2.5 s a pass), then drops to ~0.6x within two passes (4.0-4.4 s)
and stays there, whatever the cache size, the footprint or the traffic; 30 s idle buys ~10 s back
(--pause 30 --pause-every 12). ProcessInfo.thermalState stays nominal, the thermal pressure 0 and the
CPU probe unchanged. With the exchange running the throttled pass is ~4.9 s (--ahead).
"""
import argparse
import json
import queue
import socket
import struct
import threading
import time

import numpy as np

DIM, HEADS, HEAD_DIM, FFN, TEXT, LAYERS = 1536, 12, 128, 8960, 512, 30
TOKENS, FRAME_TOKENS, CACHE_FRAMES = 6032, 1508, 18
rng = np.random.default_rng(0)


def bf16(shape, scale=0.02, offset=0.0):
    """Random BF16 values as (bytes, shape, dtype name): a float32's upper 16 bits"""
    values = (rng.standard_normal(shape, dtype=np.float32) * scale + np.float32(offset)).view(np.uint32) >> 16
    return values.astype(np.uint16).tobytes(), list(shape), "bfloat16"


def f32(shape, scale=1.0, offset=0.0):
    return (rng.standard_normal(shape, dtype=np.float32) * scale + offset).tobytes(), list(shape), "float32"


def encode(op, layer=0, meta=None, arrays=()):
    specs = [{"name": name, "shape": shape, "dtype": dtype} for name, (_, shape, dtype) in arrays]
    header = json.dumps({"op": op, "layer": layer, "meta": meta or {}, "arrays": specs}).encode()
    return b"".join([struct.pack("<I", len(header)), header] + [raw for _, (raw, _, _) in arrays])


def exactly(sock, count):
    buffer = bytearray(count)
    view, received = memoryview(buffer), 0
    while received < count:
        n = sock.recv_into(view[received:], count - received)
        if n == 0:
            raise ConnectionError("the iPad closed the connection")
        received += n
    return buffer


def read(sock, op):
    (length,) = struct.unpack("<I", exactly(sock, 4))
    header = json.loads(exactly(sock, length))
    if header["op"] != op:
        raise RuntimeError(f"expected {op}, got {header}")
    for spec in header["arrays"]:
        exactly(sock, int(np.prod(spec["shape"])) * (4 if spec["dtype"] == "float32" else 2))
    return header


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ipad", default="169.254.154.199")
    parser.add_argument("--port", type=int, default=5570)
    parser.add_argument("--start", type=int, default=3072, help="the Mac's tokens: [0, start)")
    parser.add_argument("--rows", type=int, default=CACHE_FRAMES * FRAME_TOKENS, help="cache rows a pass attends over")
    parser.add_argument("--cache-rows", type=int, default=CACHE_FRAMES * FRAME_TOKENS,
                        help="rows of the caches the iPad allocates (its footprint)")
    parser.add_argument("--passes", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--ahead", action="store_true")
    parser.add_argument("--solo", action="store_true",
                        help="no exchange: the iPad reuses one K/V frame in every block and sends none")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds between passes")
    parser.add_argument("--pause-every", type=int, default=1, help="pause after every this many passes")
    args = parser.parse_args()

    sock = socket.create_connection((args.ipad, args.port), timeout=5)
    sock.settimeout(600)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    outbox = queue.Queue()

    def send_loop():
        while (item := outbox.get()) is not None:
            at, data = item
            if at > time.perf_counter():
                time.sleep(at - time.perf_counter())
            sock.sendall(data)
    sender = threading.Thread(target=send_loop, daemon=True)
    sender.start()

    def send(data, at=0.0):
        outbox.put((at, data))

    share = TOKENS - args.start
    meta = {"tokens": TOKENS, "start": args.start, "cache_rows": max(args.cache_rows, args.rows), "heads": HEADS,
            "layers": LAYERS, "eps": 1e-6, "attn_eps": 1e-6, "cross_eps": 1e-6, "norm3_eps": 1e-6}
    square = lambda: bf16((DIM, DIM))
    vector = lambda n=DIM: bf16((n,))
    send(encode("setup", meta=meta, arrays=[
        ("cam_patch_weight", square()), ("cam_patch_bias", vector()), ("c2ws1_weight", square()),
        ("c2ws1_bias", vector()), ("c2ws2_weight", square()), ("c2ws2_bias", vector())]))
    layer = [
        ("qkv_weight", bf16((3 * DIM, DIM))), ("qkv_bias", vector(3 * DIM)), ("modulation", bf16((1, 6, DIM), 0.1)),
        ("norm_q", bf16((DIM,), 0.0, 1.0)), ("norm_k", bf16((DIM,), 0.0, 1.0)),
        ("o_weight", square()), ("o_bias", vector()),
        ("norm3_weight", bf16((DIM,), 0.0, 1.0)), ("norm3_bias", bf16((DIM,), 0.0)),
        ("cross_q_weight", square()), ("cross_q_bias", vector()), ("cross_norm_q", bf16((DIM,), 0.0, 1.0)),
        ("cross_o_weight", square()), ("cross_o_bias", vector()),
        ("ffn_up_weight", bf16((FFN, DIM))), ("ffn_up_bias", vector(FFN)),
        ("ffn_down_weight", bf16((DIM, FFN))), ("ffn_down_bias", vector()),
        ("inj1_weight", square()), ("inj1_bias", vector()), ("inj2_weight", square()), ("inj2_bias", vector()),
        ("cam_weight", bf16((2 * DIM, DIM))), ("cam_bias", vector(2 * DIM)),
    ]
    for i in range(LAYERS):  # the same weights in every block: only their sizes matter here
        send(encode("layer", layer=i, arrays=layer))
    read(sock, "ready")
    text = [("text_k", bf16((1, TEXT, HEADS, HEAD_DIM), 1.0)), ("text_v", bf16((1, TEXT, HEADS, HEAD_DIM), 1.0))]
    for i in range(LAYERS):
        send(encode("context", layer=i, arrays=text))
    angles = rng.uniform(0, 2 * np.pi, (TOKENS, 1, HEAD_DIM // 2)).astype(np.float32)
    rope = np.stack([np.cos(angles), np.sin(angles)], -1)
    send(encode("chunk", meta={"start": args.start}, arrays=[
        ("rope", (rope.tobytes(), list(rope.shape), "float32")), ("camera", bf16((1, share, DIM), 1.0))]))
    kv = [("k", bf16((1, args.start, HEADS, HEAD_DIM), 1.0)), ("v", bf16((1, args.start, HEADS, HEAD_DIM), 1.0))]
    frames = [encode("kv", layer=i, arrays=kv) for i in range(LAYERS)]
    begin = encode("pass", meta={"end": args.rows, "solo": args.solo}, arrays=[("time_e0", f32((1, 1, 6, DIM), 0.1)),
                                                             ("value", bf16((1, share, DIM), 1.0))])
    print(f"setup done; {args.passes} passes of {share} tokens x {args.rows} rows, "
          f"{'Mac ahead' if args.ahead else 'iPad ahead'}, delay {args.delay * 1000:.0f} ms", flush=True)
    for p in range(args.passes):
        started = time.perf_counter()
        send(begin)
        if args.ahead or args.solo:
            send(frames[0])
        for i in range(0 if args.solo else LAYERS):
            read(sock, "kv")
            at = time.perf_counter() + args.delay
            if not args.ahead:
                send(frames[i], at)
            elif i + 1 < LAYERS:
                send(frames[i + 1], at)
        result = read(sock, "result")["meta"]
        print(f"pass {p + 1}: {time.perf_counter() - started:.2f} s here; iPad {result['ms']:.0f} ms, waited "
              f"{result['wait_ms']:.0f} ms, {result['thermal']}", flush=True)
        if args.pause and (p + 1) % args.pause_every == 0:
            time.sleep(args.pause)
    send(encode("bye"))
    outbox.put(None)
    sender.join(timeout=10)
    sock.close()


if __name__ == "__main__":
    main()
