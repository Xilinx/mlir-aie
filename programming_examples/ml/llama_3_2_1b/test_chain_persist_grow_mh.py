#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Persistent-loop GROWING-KV test (capstone increment 2): `--passes` tokens in ONE
dispatch, token feeding back on-chip, AND the KV cache GROWS across tokens --
real autoregressive decode where context accumulates.

vs the fixed-position test (test_chain_persist_mh): here the KV cache is NOT
restored between tokens. The growing-append (llama_kv_append_combined_grow) writes
slot = T_used and advances T_used -> T_used+1 ON-CHIP, so token t appends at slot
P+t and attends [0, P+t]. Each token fills from one half of the KV buffer and
drains to the other, carrying the grown cache forward; the host names the half a
dispatch starts from (dispatch_params).

token-0 KV prefix T_used = P (NOT P+1): the grow-append's slot = T_used, so token
0 writes slot P and sets T_used = P+1 (flowkv attends P+1, incl the new slot).

Run (compiles the design, or reuses its JIT cache entry):
  python test_chain_persist_grow_mh.py --synthetic-weights
  LLAMA_CHAIN_N=16 python test_chain_persist_grow_mh.py --time
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie2_chain_dynscale_mh import (
    CS_BYTES,
    HEAD_D,
    KV_BLOCK_WORDS,
    KV_ROW_WORDS,
    KV_TB,
    N_HEADS_KV,
    N_TILE,
    SAMPLE_SEED,
    SAMPLE_TEMP,
    SAMPLE_TOPK,
    SILU_GATE_SCALE,
    VOCAB,
    WLM_TOTAL,
    D,
    kv_blocks,
    kv_words,
)
from aie2_decode import (
    ACT_BYTES,
    GAMMA_AT,
    LANE_BYTES,
    REC_BYTES,
    build,
    stage_layout,
    wblob_layout,
)
from aie2_gemv_array import LAYER, N_CORES, RESID_OFF
from gen_exp_lut import exp_lut
from gen_llama_data import quant_int8_perchan_absmax
from gen_silu_lut import silu_lut
from llama_npu import DispatchSession
from ml_dtypes import bfloat16
from numpy_layer_mh import gen_layer_mh, numpy_layer_mh_forward, requant
from numpy_llama_ref import EMB_DIM, VOCAB_SIZE, _load_bf16, _load_qw, rope_table
from test_flowkv import EXP_QUANT_SCALE as FKV_EXP_QUANT_SCALE
from test_gemv_array import pack_lane
from test_layer_mh import pack_perchan_slots
from test_rmsnorm_int8_dyn import numpy_rmsnorm_int8_dyn

T = int(os.environ.get("LLAMA_CHAIN_T", "128"))
N_LAYERS = int(os.environ.get("LLAMA_CHAIN_N", "2"))
ONESTREAM_KSET = int(os.environ.get("LLAMA_CHAIN_ONESTREAM_KSET", "8"))
P_MAX = 4
ATTN_CORES = 2
DATA_DIR = Path(__file__).parent / "data"


def synthetic_lm_weights(seed, rows_per_chunk=8192):
    """A seeded stand-in for data/'s embed table and final_norm gamma.

    Same shapes, dtypes and per-row INT8 quantization as gen_llama_data.py
    writes; the distributions follow Llama 3.2 1B's (embed std ~0.022, gamma
    ~N(2.35, 0.28)). Built in row chunks: the full fp32 table would be 1 GB.
    """
    rng = np.random.default_rng(seed)
    embed_i8 = np.empty((VOCAB, D), np.int8)
    embed_sc = np.empty(VOCAB, np.float32)
    for r0 in range(0, VOCAB, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, VOCAB)
        rows = rng.normal(0.0, 0.022, size=(r1 - r0, D)).astype(np.float32)
        embed_i8[r0:r1], embed_sc[r0:r1] = quant_int8_perchan_absmax(rows)
    gamma = rng.normal(2.35, 0.28, size=D).astype(bfloat16)
    return embed_i8, embed_sc, gamma


def pack_lmw(embed_i8, embed_sc, gamma):
    bias = np.zeros(VOCAB, np.int32)
    wpacked = pack_perchan_slots(
        embed_i8, embed_sc.astype(np.float32), bias, N_TILE, prefix_bytes=b"\x00" * 64
    )
    assert wpacked.size == WLM_TOTAL, (wpacked.size, WLM_TOTAL)
    gbytes = np.frombuffer(gamma.tobytes(), dtype=np.int8)
    blob = np.zeros(gbytes.size + WLM_TOTAL, dtype=np.int8)
    blob[: gbytes.size] = gbytes
    blob[gbytes.size :] = wpacked
    return blob


def embed_seed_ref(row_i8, embed_sc):
    amax = int(np.max(np.abs(row_i8.astype(np.int32))))
    if amax < 1:
        amax = 1
    inv = (np.float32(127.0) / np.float32(amax)).astype(np.float32)
    xin = np.empty(D, np.int8)
    for i in range(D):
        v = np.float32(np.float32(row_i8[i]) * inv)
        r = np.floor(v + 0.5) if v >= 0 else np.ceil(v - 0.5)
        xin[i] = np.int8(np.clip(r, -128, 127))
    scale = np.float32(
        np.float32(amax) * np.float32(embed_sc) * np.float32(1.0 / 127.0)
    )
    return xin, scale


def build_fixtures(seed, p0, n_tok):
    """Random layers and a T-row KV cache with p0 rows filled, the rows of
    the n_tok tokens generated from position p0 on left zero."""
    rng = np.random.default_rng(seed)
    x_fp = rng.uniform(-1.6, 1.6, size=D).astype(np.float32)
    res_scale = np.float32(np.maximum(np.abs(x_fp).max(), 1e-12) / 127.0)
    x_i8 = requant(x_fp, np.float32(1.0) / res_scale)

    lut_exp = exp_lut(FKV_EXP_QUANT_SCALE).astype(np.float32)
    lut_silu = silu_lut(SILU_GATE_SCALE)
    layers = []
    for L in range(N_LAYERS):
        layer = gen_layer_mh(rng)
        layer["lut_exp"] = lut_exp
        layer["lut_silu"] = lut_silu
        for name in ("kcaches", "vcaches"):
            layer[name] = [
                rng.integers(-32, 33, size=T * HEAD_D, dtype=np.int8)
                for _ in range(N_HEADS_KV)
            ]
        k_slot = rng.uniform(0.02, 0.08, size=(N_HEADS_KV, T)).astype(np.float32)
        v_slot = rng.uniform(0.02, 0.08, size=(N_HEADS_KV, T)).astype(np.float32)
        for h in range(N_HEADS_KV):
            layer["kcaches"][h][p0 * HEAD_D :] = 0
            layer["vcaches"][h][p0 * HEAD_D :] = 0
        k_slot[:, p0:] = 0.0
        v_slot[:, p0:] = 0.0
        layer["k_scales_slot"] = k_slot
        layer["v_scales_slot"] = v_slot
        layers.append(layer)
    return x_i8, res_scale, layers


def chain_forward_logits(x_i8, res_scale, layers, position, embed_i8, embed_sc, gamma):
    """One token's forward at `position` (caches NOT restored -> grow), attends
    [0, position]. Returns the fp32 logits over the vocab + populates
    layer['scales']. The append at `position` mutates the caches in place."""
    x_cur = (x_i8.copy(), float(res_scale))
    for L in range(N_LAYERS):
        x_cur, scales = numpy_layer_mh_forward(
            x_cur,
            layers[L],
            position=position,
            residual_dyn=True,
            attn_perslot=True,
            attn_lut=True,
            attn_tb=KV_TB,
            attn_cores=ATTN_CORES,
            attn_interleaved=True,
        )
        layers[L]["scales"] = scales
    xo_i8, xo_scale = x_cur
    normed_i8, norm_scale = numpy_rmsnorm_int8_dyn(xo_i8, gamma, np.float32(xo_scale))
    acc = embed_i8.astype(np.int64) @ normed_i8.astype(np.int64)
    logits = (
        acc.astype(np.float32) * np.float32(norm_scale) * embed_sc.astype(np.float32)
    ).astype(np.float32)
    return logits


SEED_AT = stage_layout(N_LAYERS)[0][0, "o"] + RESID_OFF
DDR_BW_GBPS = 120.0  # NPU2 LPDDR5 ~120 GB/s
LMW_GAMMA_BYTES = 2 * D


def dispatch_params(t0, passes):
    """The design's ScratchpadParameters for one dispatch.

    Args:
        t0: The position of the dispatch's first token.
        passes: The tokens it generates, 1..P_MAX.

    Returns:
        The parameter values by name: passes; t0; its RoPE table row's
        offset; per token slot t < P_MAX, whether it runs (pass{t}) and, for
        a token at position p = t0 + t, the KV blocks it reads (kvb{t}) and
        its row's word offset within a (layer, half) region (row{t}).
    """
    params = {"passes": passes, "t0": t0, "cs_off": t0 * CS_BYTES}
    for t in range(P_MAX):
        p = t0 + t
        on = t < passes
        params[f"pass{t}"] = int(on)
        params[f"kvb{t}"] = p // KV_TB + 1 if on else 0
        params[f"row{t}"] = (
            (p // KV_TB) * KV_BLOCK_WORDS + (p % KV_TB) * KV_ROW_WORDS if on else 0
        )
    return params


# The decode's time a token at position p, fit to the README's N=16
# measurements: the lm_head, then per layer its weights and its attention
# over p rows. amdxdna times a dispatch out after about 5 s.
LM_HEAD_MS = 19.0
LAYER_MS = 1.85
LAYER_MS_PER_ROW = 0.0006
DISPATCH_BUDGET_MS = 4000.0


def passes_for(t0):
    """The tokens a dispatch from position t0 generates.

    Args:
        t0: The position of its first token.

    Returns:
        As many as P_MAX allows, the cache has room for and the dispatch
        budget fits, and at least one.
    """
    n, ms = 0, 0.0
    while n < min(P_MAX, T - t0):
        ms += LM_HEAD_MS + N_LAYERS * (LAYER_MS + LAYER_MS_PER_ROW * (t0 + n))
        if n and ms > DISPATCH_BUDGET_MS:
            break
        n += 1
    return max(n, 1)


def pack_wblob(layers, rope):
    """The array decode's weight blob: each layer's eight weight lanes, then
    the RoPE table.

    Args:
        layers: The oracle's layers.
        rope: bf16 (T, 2 * HEAD_D), numpy_llama_ref.rope_table.

    Returns:
        int8, wblob_layout(N_LAYERS, T)'s.
    """
    off_lanes, off_rope, total = wblob_layout(N_LAYERS, T)
    w = np.zeros(total, np.int8)
    for L, layer in enumerate(layers):
        weights = {
            p: (
                layer[f"w{p}_i8"],
                layer.get(f"b{p}", np.zeros(layer[f"w{p}_i8"].shape[0], np.int32)),
                layer[f"w{p}_sc"],
            )
            for p in LAYER
        }
        for c in range(N_CORES):
            at = off_lanes + (L * N_CORES + c) * LANE_BYTES
            w[at : at + LANE_BYTES] = pack_lane(weights, LAYER, c)
    w[off_rope : off_rope + T * CS_BYTES] = rope.view(np.int8).ravel()
    return w


def pack_stage(layers, final_gamma, x_i8, res_scale):
    """The array decode's stage buffer at the start of a generation: each
    norm's gamma after the residual it norms, and the first token's seed.

    Args:
        layers: The oracle's layers.
        final_gamma: The final norm's bf16 gamma.
        x_i8: The first token's seed, int8[D].
        res_scale: Its scale.

    Returns:
        int8, stage_layout(N_LAYERS)'s.
    """
    slots, _, final, _, total = stage_layout(N_LAYERS)
    st = np.zeros(total, np.int8)
    gammas = [(slots[L, "o"], layer["gamma_in"]) for L, layer in enumerate(layers)]
    gammas += [(slots[L, "d"], layer["gamma_post"]) for L, layer in enumerate(layers)]
    gammas.append((final, final_gamma))
    for at, g in gammas:
        at += RESID_OFF + GAMMA_AT
        st[at : at + 2 * D] = np.asarray(g, bfloat16).view(np.int8)
    st[SEED_AT : SEED_AT + D] = x_i8
    st[SEED_AT + D : SEED_AT + D + 4] = np.frombuffer(
        np.float32(res_scale).tobytes(), np.int8
    )
    return st


def pack_kv(layers):
    """The device's KV cache holding the oracle's caches.

    Args:
        layers: Per layer, the oracle's T-row kcaches/vcaches and k/v slot
            scales.

    Returns:
        int32 (kv_words(T, N_LAYERS),), block-major as the design reads it.
    """
    nb = kv_blocks(T)
    rows = np.zeros((N_LAYERS, N_HEADS_KV, nb * KV_TB, KV_ROW_WORDS * 4), np.int8)
    for L, layer in enumerate(layers):
        for h in range(N_HEADS_KV):
            rows[L, h, :T, :HEAD_D] = layer["kcaches"][h].reshape(T, HEAD_D)
            rows[L, h, :T, HEAD_D : 2 * HEAD_D] = layer["vcaches"][h].reshape(T, HEAD_D)
            for i, name in enumerate(("k_scales_slot", "v_scales_slot")):
                at = 2 * HEAD_D + 4 * i
                rows[L, h, :T, at : at + 4] = (
                    layer[name][h].astype(np.float32).view(np.int8).reshape(T, 4)
                )
    blocks = rows.reshape(N_LAYERS, 2, N_HEADS_KV // 2, nb, KV_TB, -1)
    packed = np.ascontiguousarray(blocks.transpose(0, 1, 3, 2, 4, 5))
    assert packed.size == kv_words(T, N_LAYERS) * 4
    return packed.view(np.int32).ravel()


def time_decode(session, buffers, params, opts):
    """Per-token device time + DMA floor (bytes/token counts the live KV
    prefix the first token reads)."""
    passes = params["passes"]
    print(
        f"timing: warmup x{opts.n_warmup}, time x{opts.n_iters}  ({passes} tokens/dispatch)",
        flush=True,
    )
    for _ in range(opts.n_warmup):
        session(buffers, **params)
    times_ms = [session(buffers, **params) for _ in range(opts.n_iters)]
    arr = np.array(times_ms)
    disp_med = float(np.median(arr))
    per_token_ms = disp_med / passes
    kv_read = 2 * N_LAYERS * params["kvb0"] * KV_BLOCK_WORDS * 4
    bytes_per_token = (
        wblob_layout(N_LAYERS, T)[1] + LMW_GAMMA_BYTES + WLM_TOTAL + kv_read
    )
    dma_floor_ms = bytes_per_token / (DDR_BW_GBPS * 1e9) * 1e3
    ratio = per_token_ms / dma_floor_ms
    verdict = (
        "DMA-bound" if ratio < 1.5 else "COMPUTE-bound (vectorization is the lever)"
    )
    print(
        f"per-dispatch: n={opts.n_iters} median={disp_med:.2f} ms "
        f"min={arr.min():.2f} max={arr.max():.2f} std={arr.std():.2f}"
    )
    print(f"per-token:    {per_token_ms:.2f} ms  ({1000.0/per_token_ms:.2f} tok/s)")
    print(
        f"bytes/token:  {bytes_per_token/1e6:.1f} MB  DMA floor {dma_floor_ms:.2f} ms"
    )
    print(f"ratio:        {ratio:.2f}x  -> {verdict}")
    return 0


# Device-trajectory replay tolerances: the oracle follows the DEVICE's tokens
# (not its own greedy trajectory) since the token feeds back ON-CHIP. A device
# token passes if it is the oracle argmax OR a near-tie judged RELATIVE to the
# logit spread (gap/std): the vector fp32 multiplies the oracle does not mirror
# yet (rmsnorm, silu, up_requant) leave rare 1-ULP differences that grow with
# depth. A flip within the top cluster is quality-neutral; one far down would be
# real drift.
TOL_GAP_STD = 0.15  # gap as a fraction of the logit std
TOL_RANK = 5  # dev_tok must be within the oracle top-K


def set_pristine(layers, pristine):
    kcaches, vcaches, kslot, vslot = pristine
    for L in range(N_LAYERS):
        layers[L]["kcaches"] = [c.copy() for c in kcaches[L]]
        layers[L]["vcaches"] = [c.copy() for c in vcaches[L]]
        layers[L]["k_scales_slot"] = kslot[L].copy()
        layers[L]["v_scales_slot"] = vslot[L].copy()


def replay(dev_tokens, x_i8, res_scale, layers, p0, rope, lm):
    """Check each device token against the numpy oracle, from the caches the
    caller set and position p0 on, following the device's tokens; returns the
    number that fail."""
    embed_i8, embed_sc, gamma = lm
    fails = 0
    cur_x, cur_sc = x_i8, res_scale
    for t, dev_tok in enumerate(dev_tokens):
        for L in range(N_LAYERS):
            layers[L]["cos"] = rope[p0 + t, :HEAD_D]
            layers[L]["sin"] = rope[p0 + t, HEAD_D:]
        logits = chain_forward_logits(
            cur_x, cur_sc, layers, p0 + t, embed_i8, embed_sc, gamma
        )
        true_max = np.float32(logits.max())
        ref_argmax = int(np.argmax(logits))
        gap = float(true_max - logits[dev_tok])
        spread = float(logits.std())
        order = np.argsort(logits)[::-1]
        rank = int(np.where(order == dev_tok)[0][0])
        near_tie = (gap / spread <= TOL_GAP_STD) and (rank < TOL_RANK)
        ok = (dev_tok == ref_argmax) or near_tie
        fails += not ok
        if dev_tok == ref_argmax:
            tag = "exact"
        else:
            tag = f"near-tie gap/std={gap/spread:.3f} rank={rank}"
        print(
            f"  token {t} (pos {p0+t}): {'PASS' if ok else 'FAIL'}  "
            f"dev={dev_tok} ref_argmax={ref_argmax}  [{tag}]",
            flush=True,
        )
        cur_x, cur_sc = embed_seed_ref(embed_i8[dev_tok], float(embed_sc[dev_tok]))
    return fails


def tokens_of(packed, passes):
    return [
        int(
            np.frombuffer(
                packed[t * REC_BYTES + D + 4 : t * REC_BYTES + D + 8].tobytes(),
                np.int32,
            )[0]
        )
        for t in range(passes)
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--elf",
        help="run this full ELF, built by aie2_chain_dynscale_mh.py --full-elf-path "
        "at this test's configuration, instead of the JIT cache's",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time", action="store_true", help="measure per-token device time")
    p.add_argument("--n-iters", type=int, default=1)
    p.add_argument("--n-warmup", type=int, default=0)
    p.add_argument(
        "--synthetic-weights",
        action="store_true",
        help="use a seeded synthetic embed table + final_norm (same layout as data/) "
        "instead of the real Llama weights; the oracle runs on the same table",
    )
    p.add_argument(
        "--dispatches",
        type=int,
        default=1,
        help="dispatches run back to back like a generation loop: each carries the "
        "KV cache and the last token's seed into the next; every token is checked",
    )
    p.add_argument(
        "--idle-s",
        type=float,
        default=0.0,
        help="seconds to leave the NPU idle before each later dispatch; past "
        "amdxdna's autosuspend delay (5 s) the device suspends in between",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="run the first dispatch this many times in one process from the same "
        "inputs; every repeat must produce the first's tokens",
    )
    p.add_argument(
        "--start-pos",
        type=int,
        default=None,
        help="KV prefix length before the first generated token "
        "(default: T//2, or as much as leaves room for every generated token)",
    )
    p.add_argument(
        "--passes",
        type=int,
        choices=range(1, P_MAX + 1),
        help="tokens generated per dispatch (default: passes_for's, per dispatch)",
    )
    p.add_argument(
        "--split",
        action="store_true",
        help="also generate the first dispatch's tokens one per dispatch, from the "
        "same inputs; the records, KV cache and stage must be byte-identical",
    )
    opts = p.parse_args()
    fseed = opts.seed
    n_disp = opts.dispatches
    p0 = opts.start_pos
    if p0 is None:
        p0 = min(T // 2, T - n_disp * (opts.passes or P_MAX))
    schedule = []
    pos = p0
    for _ in range(n_disp):
        schedule.append((pos, opts.passes or passes_for(pos)))
        pos += schedule[-1][1]
    passes = schedule[0][1]
    n_tok = pos - p0
    if p0 < 0 or p0 + n_tok > T:
        p.error(f"{n_tok} tokens from position {p0} do not fit in T={T}")
    if (n_disp > 1) + (opts.repeats > 1) + opts.split > 1:
        p.error("--dispatches, --repeats and --split are separate modes")

    assert VOCAB == VOCAB_SIZE and D == EMB_DIM
    if opts.synthetic_weights:
        print(f"building synthetic embed/final_norm (seed {fseed}) ...", flush=True)
        embed_i8, embed_sc, gamma = synthetic_lm_weights(fseed)
    else:
        print(f"loading embed/final_norm from {DATA_DIR} ...", flush=True)
        embed_i8, embed_sc = _load_qw(DATA_DIR, "embed", (VOCAB, D))
        gamma = _load_bf16(DATA_DIR, "final_norm", (D,))
    print("packing lm_head weights (262 MB) ...", flush=True)
    lmw = pack_lmw(embed_i8, embed_sc, gamma)
    lm = (embed_i8, embed_sc, gamma)

    x_i8, res_scale, layers = build_fixtures(fseed, p0, n_tok)
    rope = rope_table(T, HEAD_D)

    # Snapshot the PRISTINE caches (before any append) so the device gets the same
    # starting KV the numpy loop starts from. The device's growing-append then
    # accumulates the tokens on-chip exactly as numpy does in place.
    pristine = (
        [[c.copy() for c in lyr["kcaches"]] for lyr in layers],
        [[c.copy() for c in lyr["vcaches"]] for lyr in layers],
        [lyr["k_scales_slot"].copy() for lyr in layers],
        [lyr["v_scales_slot"].copy() for lyr in layers],
    )

    w_t = iron.tensor(pack_wblob(layers, rope), dtype=np.int8)
    kv0 = pack_kv(layers)
    stage0 = pack_stage(layers, gamma, x_i8, res_scale)

    session = DispatchSession(
        build,
        opts.elf,
        P_MAX=P_MAX,
        N_LAYERS=N_LAYERS,
        T=T,
        KSET=ONESTREAM_KSET,
        ATTN_CORES=ATTN_CORES,
    )
    lmw_t = iron.tensor(lmw, dtype=np.int8)
    print(
        f"chain_persist_grow_mh N_LAYERS={N_LAYERS} passes={[n for _, n in schedule]}"
        f" P0={p0} KSET={ONESTREAM_KSET}"
        f"  temp={SAMPLE_TEMP} topk={SAMPLE_TOPK} seed={SAMPLE_SEED}  fixture_seed={fseed}"
        f"  dispatches={n_disp} repeats={opts.repeats}",
        flush=True,
    )
    if opts.time:
        return time_decode(
            session,
            [
                w_t,
                iron.tensor(kv0, dtype=np.int32),
                lmw_t,
                iron.zeros([P_MAX * REC_BYTES], dtype=np.int8),
                iron.tensor(stage0, dtype=np.int8),
            ],
            dispatch_params(p0, passes),
            opts,
        )

    # Each dispatch gets its own input, weight and output tensors; the KV
    # tensor is created once per run of dispatches and stays on the device,
    # so a generation loop's dispatches carry the cache with no host copy.
    def dispatch(stage_t, kv_t, params):
        out_t = iron.zeros([P_MAX * REC_BYTES], dtype=np.int8)
        session([w_t, kv_t, lmw_t, out_t, stage_t], **params)
        out_t.to("cpu")
        return out_t.numpy().copy()

    # The oracle follows the device's tokens across dispatches, from the
    # seed each dispatch left; after each, the device's whole KV cache must
    # equal the oracle's.
    dev_tokens = []
    tok_fails = carry_fails = 0
    if opts.split:
        # One dispatch of `passes` tokens, then the same tokens one per
        # dispatch: a token slot that does not run moves nothing.
        kv_t = iron.tensor(kv0, dtype=np.int32)
        stage_t = iron.tensor(stage0, dtype=np.int8)
        whole = dispatch(stage_t, kv_t, dispatch_params(p0, passes))
        kv_t.to("cpu")
        stage_t.to("cpu")
        kv_whole, stage_whole = kv_t.numpy().copy(), stage_t.numpy().copy()
        kv_t = iron.tensor(kv0, dtype=np.int32)
        stage_t = iron.tensor(stage0, dtype=np.int8)
        recs = [
            dispatch(stage_t, kv_t, dispatch_params(p0 + t, 1)) for t in range(passes)
        ]
        kv_t.to("cpu")
        stage_t.to("cpu")
        n_rec = passes * REC_BYTES
        checks = {
            "records": np.array_equal(
                whole[:n_rec], np.concatenate([r[:REC_BYTES] for r in recs])
            ),
            "untouched records": not np.any(whole[n_rec:])
            and not any(np.any(r[REC_BYTES:]) for r in recs),
            "KV cache": np.array_equal(kv_whole, kv_t.numpy()),
            "stage": np.array_equal(stage_whole, stage_t.numpy()),
        }
        for name, same in checks.items():
            print(
                f"  {passes} tokens vs {passes} x 1, {name}: {'same' if same else 'DIFFER'}"
            )
            carry_fails += not same
        dev_tokens = tokens_of(whole, passes)
        tok_fails = replay(dev_tokens, x_i8, res_scale, layers, p0, rope, lm)
        wrong = int(np.count_nonzero(kv_whole != pack_kv(layers)))
        print(f"  KV cache: {'PASS' if not wrong else f'{wrong} words FAIL'}")
        carry_fails += wrong != 0
    elif opts.repeats > 1:
        for r in range(opts.repeats):
            toks = tokens_of(
                dispatch(
                    iron.tensor(stage0, dtype=np.int8),
                    iron.tensor(kv0, dtype=np.int32),
                    dispatch_params(p0, passes),
                ),
                passes,
            )
            if r == 0:
                dev_tokens = toks
            same = toks == dev_tokens
            print(
                f"repeat {r}: tokens {toks} {'same' if same else 'DIFFER'}", flush=True
            )
            if not same:
                print(f"repeat {r} differs from repeat 0", file=sys.stderr)
                return 1
        set_pristine(layers, pristine)
        tok_fails = replay(dev_tokens, x_i8, res_scale, layers, p0, rope, lm)
    else:
        # Like the KV tensor, the stage tensor stays on the device: the host
        # writes the first dispatch's seed into it, and each dispatch leaves
        # the next one's there.
        kv_t = iron.tensor(kv0, dtype=np.int32)
        stage_t = iron.tensor(stage0, dtype=np.int8)
        x_d, sc_d = x_i8, res_scale
        for d, (pos, passes) in enumerate(schedule):
            if d:
                time.sleep(opts.idle_s)
            toks_out = dispatch(stage_t, kv_t, dispatch_params(pos, passes))
            toks = tokens_of(toks_out, passes)
            dev_tokens += toks
            if n_disp > 1:
                print(f"dispatch {d} (pos {pos}): tokens {toks}", flush=True)
            tok_fails += replay(toks, x_d, sc_d, layers, pos, rope, lm)
            # The device appended exactly the rows the oracle did, and wrote
            # nothing else.
            kv_t.to("cpu")
            wrong = int(np.count_nonzero(kv_t.numpy() != pack_kv(layers)))
            print(f"  KV cache: {'PASS' if not wrong else f'{wrong} words FAIL'}")
            carry_fails += wrong != 0
            # The oracle's next layer-0 input is the last token's seed (body +
            # scale), which the device left in the seed tensor with a zero pad.
            last = (passes - 1) * REC_BYTES
            x_d = toks_out[last : last + D].copy()
            sc_d = float(
                np.frombuffer(toks_out[last + D : last + D + 4].tobytes(), np.float32)[
                    0
                ]
            )
            stage_t.to("cpu")
            seed_left = np.zeros(ACT_BYTES, dtype=np.int8)
            seed_left[: D + 4] = toks_out[last : last + D + 4]
            if not np.array_equal(
                stage_t.numpy()[SEED_AT : SEED_AT + ACT_BYTES], seed_left
            ):
                print(f"  seed carry: FAIL (dispatch {d})", flush=True)
                carry_fails += 1

    n = len(dev_tokens)
    print(
        f"\nchain_persist_grow_mh: {n - tok_fails}/{n} tokens PASS  "
        f"(on-chip feedback + GROWING KV + per-position rope)"
    )
    if opts.split:
        print(
            f"{passes} tokens vs {passes} x 1: {'PASS' if not carry_fails else 'FAIL'}"
        )
    if n_disp > 1:
        print(
            f"carry across {n_disp} dispatches: {'PASS' if not carry_fails else 'FAIL'}"
        )
    return 0 if tok_fails == 0 and carry_fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
