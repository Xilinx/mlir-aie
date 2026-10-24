#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""A prefill pass's rows (64, or --rows) through layers 0..n - 1
(aie2_prefill_layer.py) on the NPU, bit for bit against
llama_oracle.Layer.forward.

The prompt's tokens are seeded random ids; the device makes each row's
embed seed from the token's lm_head slot, read from the lm_head blob, and
its id. The oracle runs the positions before block j one at a time, which
fills its KV caches; the device gets those caches and runs block j's rows
in one pass: its embed commands, then a command per layer, queued.
Checked, every row bit for bit: the residual leaving the last layer, block
j's KV rows in every layer, and every stage of the last layer the device
leaves in `stage` against the same steps taken one at a time
(forward_steps, itself checked against Layer.forward), which also locates
a mismatch; and that the block's rows past the pass's keep what the cache
held.

The oracle runs before the NPU lock is taken; `--lock` serializes the device
part with other users of the NPU.
"""

import argparse
import copy
import fcntl
import sys
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.utils.aie_fp32 import aie_add
from aie2_decode import ACT_BYTES, HACT_BYTES
from aie2_prefill import pack_trailers
from aie2_prefill_layer import (
    CHUNK_STRIDE,
    CONFIGURE,
    QKV_FP,
    ROW_STEP,
    PrefillStage,
    build_prefill,
    ids_words,
    pass_commands,
)
from llama_config import HD, HEAD_D, KV_TB, N_HEADS_KV, N_HEADS_Q, VOCAB, D
from llama_decode import DecodeShape, pack_lmw
from llama_npu import QUEUE_DEPTH, DispatchSession
from llama_oracle import (
    Layer,
    attention_blocks,
    dyn_scale,
    embed_seed_ref,
    mul_rne,
    requant,
    rmsnorm_f32,
    rope_q,
    round_to_i8,
    silu_mul,
)
from ml_dtypes import bfloat16
from numpy_llama_ref import rope_table


def forward_steps(layer, x, position, cos, sin):
    """Layer.forward one step at a time, keeping each step's result.

    Returns:
        (the residual leaving the layer, {step name: value}); appends the
        row's KV row as Layer.forward does.
    """
    s = {}
    x_fp = np.asarray(x, np.float32)
    s["h1"] = rmsnorm_f32(x_fp, layer.gamma_in)
    h1, act1 = s["h1"]
    half = HEAD_D // 2
    cf = np.asarray(cos, np.float32)
    sf = np.asarray(sin, np.float32)
    kh = layer.project("k", h1, act1).reshape(N_HEADS_KV, HEAD_D)
    k_rope = np.empty_like(kh)
    k_rope[:, :half] = aie_add(kh[:, :half] * cf[:half], -(kh[:, half:] * sf[:half]))
    k_rope[:, half:] = aie_add(kh[:, half:] * cf[half:], kh[:, :half] * sf[half:])
    vh = layer.project("v", h1, act1).reshape(N_HEADS_KV, HEAD_D)
    for cache, slot, rows in (
        (layer.kcache, layer.k_slot, k_rope),
        (layer.vcache, layer.v_slot, vh),
    ):
        scales = dyn_scale(rows, axis=1)
        slot[:, position] = scales
        cache[:, position] = requant(rows, (np.float32(1) / scales)[:, None])
    q_fp = layer.project("q", h1, act1)
    s["qkv"] = q_fp, kh.ravel(), vh.ravel()
    q = rope_q(q_fp, cos, sin).reshape(N_HEADS_Q, HEAD_D)
    q_scales = dyn_scale(q, axis=1)
    q_i8 = round_to_i8(mul_rne(q, (np.float32(1) / q_scales)[:, None]))
    s["q"] = q_i8, q_scales
    sv = attention_blocks(
        q_i8,
        q_scales,
        layer.kcache,
        layer.vcache,
        layer.k_slot,
        layer.v_slot,
        position + 1,
        KV_TB,
    )
    s["sv"] = sv.ravel()
    o_act_scale = dyn_scale(sv)
    af = round_to_i8(mul_rne(sv, np.float32(1) / o_act_scale)).ravel()
    s["af"] = af, o_act_scale
    s["o"] = layer.project("o", af, o_act_scale)
    x1_fp = aie_add(x_fp, s["o"])
    s["x1"] = x1_fp
    s["h2"] = rmsnorm_f32(x1_fp, layer.gamma_post)
    h2, act2 = s["h2"]
    g = layer.project("g", h2, act2)
    u = layer.project("u", h2, act2)
    s["gu"] = g, u
    sf_fp = silu_mul(g, u)
    silu_scale = dyn_scale(sf_fp)
    s["sf"] = round_to_i8(mul_rne(sf_fp, np.float32(1) / silu_scale)), silu_scale
    s["d"] = layer.project("d", s["sf"][0], silu_scale)
    out = aie_add(x1_fp, s["d"])
    return out, s


def stage_checks(stage, st, want):
    """Each stage the device leaves in `stage` against forward_steps'.

    Args:
        stage: The dispatch's PrefillStage.
        st: The stage buffer, int8.
        want: Per row, forward_steps' steps.

    Returns:
        [(stage name, rows that differ)].
    """
    at = stage.at

    def f32(name, cols):
        return st[at[name] : at[name] + len(want) * cols * 4].view(np.float32)

    def acts(name, row_bytes, k):
        rows = st[at[name] : at[name] + len(want) * row_bytes].reshape(-1, row_bytes)
        return rows[:, :k], rows[:, k : k + 4].copy().view(np.float32)[:, 0]

    def same(a, b):
        return np.asarray(a).tobytes() == np.asarray(b).tobytes()

    out = []

    def check(name, got_of_row):
        bad = [r for r, w in enumerate(want) if not got_of_row(r, w)]
        out.append((name, bad))

    h, hs = acts("a_qkv", ACT_BYTES, D)
    check("h1", lambda r, w: same(h[r], w["h1"][0]) and same(hs[r], w["h1"][1]))
    qkv = f32("out_qkv", QKV_FP).reshape(-1, 8, 6 * HEAD_D)

    def qkv_row(r, w):
        q, k, v = w["qkv"]
        got = qkv[r]
        return (
            same(got[:, : 4 * HEAD_D].ravel(), q)
            and same(got[:, 4 * HEAD_D : 5 * HEAD_D].ravel(), k)
            and same(got[:, 5 * HEAD_D :].ravel(), v)
        )

    check("out_qkv", qkv_row)
    ch = st[at["chunks"] : at["chunks"] + N_HEADS_KV * 64 * CHUNK_STRIDE].reshape(
        N_HEADS_KV, 64, CHUNK_STRIDE
    )

    def chunk_row(r, w):
        q_i8, q_scales = w["q"]
        for g in range(N_HEADS_KV):
            c = ch[g, r]
            tail = c[4 * HEAD_D : 4 * HEAD_D + 32].copy().view(np.float32)
            if not (
                same(c[: 4 * HEAD_D], q_i8[4 * g : 4 * g + 4].ravel())
                and same(tail[0::2], q_scales[4 * g : 4 * g + 4])
            ):
                return False
        return True

    check("q chunks", chunk_row)
    sv = f32("sv", D).reshape(-1, D)
    check("sv", lambda r, w: same(sv[r], w["sv"]))
    af, afs = acts("a_o", ACT_BYTES, D)
    check("af", lambda r, w: same(af[r], w["af"][0]) and same(afs[r], w["af"][1]))
    o = f32("out_o", D).reshape(-1, D)
    check("out_o", lambda r, w: same(o[r], w["o"]))
    x1 = f32("x1", D).reshape(-1, D)
    check("x1", lambda r, w: same(x1[r], w["x1"]))
    h2, h2s = acts("a_gu", ACT_BYTES, D)
    check("h2", lambda r, w: same(h2[r], w["h2"][0]) and same(h2s[r], w["h2"][1]))
    gu = f32("out_gu", 2 * HD).reshape(-1, 2 * HD)
    check(
        "out_gu",
        lambda r, w: same(gu[r, :HD], w["gu"][0]) and same(gu[r, HD:], w["gu"][1]),
    )
    sf, sfs = acts("a_d", HACT_BYTES, HD)
    check("sf", lambda r, w: same(sf[r], w["sf"][0]) and same(sfs[r], w["sf"][1]))
    d = f32("out_d", D).reshape(-1, D)
    check("out_d", lambda r, w: same(d[r], w["d"]))
    return out


def main():
    ap = argparse.ArgumentParser(prog="Prefill layers against Layer.forward")
    ap.add_argument("--block", type=int, default=1, help="j: rows from 64j")
    ap.add_argument(
        "--rows",
        type=int,
        default=KV_TB,
        help=f"the pass's rows, a multiple of {ROW_STEP} to {KV_TB}",
    )
    ap.add_argument("--layers", type=int, default=1, help="layers 0 .. n - 1")
    ap.add_argument("--t", type=int, default=256, help="KV cache rows")
    ap.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--synthetic-weights", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=0, help="timed dispatches; 0: none")
    ap.add_argument("--lock", type=Path, help="flock this file around NPU work")
    opts = ap.parse_args()

    j = opts.block
    n = opts.layers
    m = opts.rows
    p0 = j * KV_TB
    if p0 + KV_TB > opts.t:
        ap.error(f"block {j} is past t={opts.t}")
    if m < ROW_STEP or m > KV_TB or m % ROW_STEP:
        ap.error(f"--rows {m} is not a multiple of {ROW_STEP} to {KV_TB}")
    rng = np.random.default_rng(opts.seed)
    if opts.synthetic_weights:
        layers = [Layer.random(rng, opts.t) for _ in range(n)]
        embed = rng.integers(-128, 128, (VOCAB, D), dtype=np.int8)
        embed_sc = rng.uniform(1e-3, 1e-2, VOCAB).astype(np.float32)
    else:
        layers = [Layer.real(L, opts.data, opts.t) for L in range(n)]
        embed = np.memmap(opts.data / "embed.i8.bin", np.int8, "r", shape=(VOCAB, D))
        embed_sc = np.fromfile(opts.data / "embed.scales.f32.bin", np.float32)
    rope = rope_table(opts.t, HEAD_D)
    tokens = rng.integers(0, VOCAB, p0 + KV_TB)
    seeds = np.stack(
        [embed_seed_ref(np.asarray(embed[t]), float(embed_sc[t])) for t in tokens]
    )

    t0 = time.perf_counter()
    for p in range(p0):
        x = seeds[p]
        for layer in layers:
            x = layer.forward(x, p, rope[p, :HEAD_D], rope[p, HEAD_D:])
    shape = DecodeShape(n, opts.t)
    kv_before = shape.pack_kv(layers)
    steps_layer = copy.deepcopy(layers[-1])
    want, steps = [], []
    for i in range(m):
        p = p0 + i
        cos, sin = rope[p, :HEAD_D], rope[p, HEAD_D:]
        x = seeds[p]
        for layer in layers[:-1]:
            x = layer.forward(x, p, cos, sin)
        want.append(layers[-1].forward(x, p, cos, sin))
        x_out, s = forward_steps(steps_layer, x, p, cos, sin)
        assert np.asarray(x_out).tobytes() == np.asarray(want[-1]).tobytes()
        steps.append(s)
    want = np.stack(want)
    rows = np.arange(p0, p0 + m)
    print(
        f"oracle: {n} layers, {p0} positions, then block {j}'s {m} rows, in "
        f"{time.perf_counter() - t0:.1f} s"
    )

    stage = PrefillStage(n)
    st = np.zeros(stage.total, np.int8)
    for L, layer in enumerate(layers):
        for post, g in enumerate((layer.gamma_in, layer.gamma_post)):
            at = stage.gamma(L, post) * 4
            st[at : at + 2 * D] = np.asarray(g, bfloat16).view(np.int8)
    lmw = pack_lmw(embed, embed_sc, np.zeros(D, np.float32))
    ids = np.zeros(ids_words(opts.t), np.int32)
    ids[: tokens.size] = tokens
    wblob = shape.pack_wblob(layers, rope)
    commands = [(CONFIGURE, {})] + pass_commands(tokens[p0 : p0 + m], j, n, p0)
    kwargs = dict(N_LAYERS=n, LAYER=0, W_LAYERS=n, T=opts.t)

    def run_pass():
        """The pass's commands, queued QUEUE_DEPTH at most; its wall time
        in ms."""
        start = time.perf_counter_ns()
        queued = []
        for command, params in commands:
            if len(queued) == QUEUE_DEPTH:
                queued.pop(0).wait()
            queued.append(session.submit(tensors, command, **params))
        for d in queued:
            d.wait()
        return (time.perf_counter_ns() - start) / 1e6

    build_prefill.specialize(full_elf=True, **kwargs).compile()

    lock = open(opts.lock, "w") if opts.lock else None
    if lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        session = DispatchSession(build_prefill, None, **kwargs)
        kv_t = iron.tensor(kv_before.copy(), dtype=np.int32)
        st_t = iron.tensor(st.view(np.int32).copy(), dtype=np.int32)
        tensors = [
            iron.tensor(wblob.view(np.int32), dtype=np.int32),
            iron.tensor(pack_trailers(wblob, n).view(np.int32).ravel(), dtype=np.int32),
            kv_t,
            st_t,
            iron.tensor(lmw.view(np.int32), dtype=np.int32),
            iron.tensor(ids, dtype=np.int32),
        ]
        first_ms = run_pass()
        kv_t.to("cpu")
        st_t.to("cpu")
        kv_got = kv_t.numpy().copy()
        st_got = st_t.numpy().copy().view(np.int8)
        times = []
        if opts.iters:
            for _ in range(opts.warmup):
                run_pass()
            times = [run_pass() for _ in range(opts.iters)]
    finally:
        if lock:
            fcntl.flock(lock, fcntl.LOCK_UN)

    print(f"first pass: {first_ms:.3f} ms")
    stages_ok = True
    for name, bad in stage_checks(stage, st_got, steps):
        stages_ok &= not bad
        print(
            f"  layer {n - 1} {name}: {m - len(bad)}/{m} rows"
            + (f", rows {bad[:8]} differ" if bad else "")
        )
    at = stage.at[("xa", "xb")[n % 2]]
    x_out = st_got[at : at + m * D * 4].view(np.float32).reshape(m, D)
    out_ok = (x_out.view(np.uint32) == want.view(np.uint32)).all(axis=1)
    kv_ok = np.ones(m, bool)
    # The block's rows past the pass's keep what the cache held.
    rest = np.arange(p0 + m, p0 + KV_TB)
    rest_ok = True
    for L, layer in enumerate(layers):
        kv_at = shape.kv_word_index(L, rows)
        kv_ok &= (kv_got[kv_at] == shape.kv_row_words(layer, rows)).all(axis=(0, 2))
        rest_at = shape.kv_word_index(L, rest)
        rest_ok &= bool((kv_got[rest_at] == kv_before[rest_at]).all())
    print(f"residual leaving layer {n - 1}: {out_ok.sum()}/{m} rows bit-exact")
    print(f"block {j}'s KV rows, all layers: {kv_ok.sum()}/{m} rows bit-exact")
    print(
        f"block {j}'s {rest.size} rows past the pass's: {'kept' if rest_ok else 'WRITTEN'}"
    )
    exact = bool(out_ok.all() and kv_ok.all() and stages_ok and rest_ok)
    print(f"prefill of {n} layers, block {j}, {m} rows: {'PASS' if exact else 'FAIL'}")
    if times:
        print(f"pass: min {min(times):.3f} ms, median {float(np.median(times)):.3f} ms")
    return 0 if exact else 1


if __name__ == "__main__":
    sys.exit(main())
