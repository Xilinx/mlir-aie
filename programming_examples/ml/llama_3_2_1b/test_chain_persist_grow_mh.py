#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Persistent-loop GROWING-KV test (capstone increment 2): PT tokens in ONE
dispatch, token feeding back on-chip, AND the KV cache GROWS across tokens --
real autoregressive decode where context accumulates.

vs the fixed-position test (test_chain_persist_mh): here the KV cache is NOT
restored between tokens. The growing-append (llama_kv_append_combined_grow) writes
slot = T_used and advances T_used -> T_used+1 ON-CHIP, so token t appends at slot
P+t and attends [0, P+t]. The host ping-pongs two KV regions (token t fills
region t%2, drains region (t+1)%2), carrying the grown cache forward. Rope cos/sin
is held fixed (position P) this increment -- only the append slot + attention
window advance.

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
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.utils import DefaultNPURuntime
from aie2_chain_dynscale_mh import (
    CS_BYTES,
    HEAD_D,
    N_HEADS_KV,
    N_TILE,
    SAMPLE_SEED,
    SAMPLE_TEMP,
    SAMPLE_TOPK,
    SILU_GATE_SCALE,
    T_USED_BYTES,
    VOCAB,
    WLM_TOTAL,
    D,
    build,
    derived_sizes,
)
from gen_exp_lut import exp_lut
from gen_llama_data import quant_int8_perchan_absmax
from gen_silu_lut import silu_lut
from llama_npu import add_artifact_args, design_kernel
from ml_dtypes import bfloat16
from numpy_layer_mh import gen_layer_mh, numpy_layer_mh_forward, requant
from numpy_llama_ref import EMB_DIM, VOCAB_SIZE, _load_bf16, _load_qw
from numpy_topk_sample import topk_sample_reference
from test_chain_mh import pack_blobs
from test_flowkv import EXP_QUANT_SCALE as FKV_EXP_QUANT_SCALE
from test_layer_mh import pack_perchan_slots
from test_rmsnorm_int8_dyn import numpy_rmsnorm_int8_dyn

T = int(os.environ.get("LLAMA_CHAIN_T", "128"))
N_LAYERS = int(os.environ.get("LLAMA_CHAIN_N", "2"))
ONESTREAM_KSET = int(os.environ.get("LLAMA_CHAIN_ONESTREAM_KSET", "8"))
PT = int(os.environ.get("LLAMA_CHAIN_PT", "4"))
_sizes = derived_sizes(T=T, N_LAYERS=N_LAYERS, PT=PT, PERSIST_GROW=True)
KV_BYTES = _sizes.KV_BYTES
PER_LAYER_KV = _sizes.PER_LAYER_KV
PER_KV_HEAD_BYTES = _sizes.PER_KV_HEAD_BYTES
WEIGHTS_BYTES = _sizes.WEIGHTS_BYTES
OFF_CS_PERPOS = _sizes.OFF_CS_PERPOS
KV_HEADER = _sizes.KV_HEADER
KCACHE_PADDED = _sizes.KCACHE_PADDED
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
    """Random layers, a KV prefix of p0 positions, and per-position cos/sin for
    the n_tok tokens generated from position p0 on."""
    rng = np.random.default_rng(seed)
    x_fp = rng.uniform(-1.6, 1.6, size=D).astype(np.float32)
    res_scale = np.float32(np.maximum(np.abs(x_fp).max(), 1e-12) / 127.0)
    x_i8 = requant(x_fp, np.float32(1.0) / res_scale)

    lut_exp = exp_lut(FKV_EXP_QUANT_SCALE).astype(np.float32)
    lut_silu = silu_lut(SILU_GATE_SCALE)
    half = HEAD_D // 2
    layers = []
    # Per-layer base rotary frequencies (random per layer, but the PHASE advances
    # with position so each token gets a distinct (cos,sin) -- real rope behavior).
    cs_perpos = np.zeros((n_tok, N_LAYERS, 2 * HEAD_D), dtype=bfloat16)  # [cos|sin]
    for L in range(N_LAYERS):
        layer = gen_layer_mh(rng)
        layer["lut_exp"] = lut_exp
        layer["lut_silu"] = lut_silu
        k_slot = rng.uniform(0.02, 0.08, size=(N_HEADS_KV, T)).astype(np.float32)
        v_slot = rng.uniform(0.02, 0.08, size=(N_HEADS_KV, T)).astype(np.float32)
        for h in range(N_HEADS_KV):
            for p in range(p0, p0 + n_tok):
                layer["kcaches"][h][p * HEAD_D : (p + 1) * HEAD_D] = 0
                layer["vcaches"][h][p * HEAD_D : (p + 1) * HEAD_D] = 0
                k_slot[h, p] = 0.0
                v_slot[h, p] = 0.0
        layer["k_scales_slot"] = k_slot
        layer["v_scales_slot"] = v_slot
        # Per-layer base frequencies; position t's angle = freq * (p0+t).
        freq = rng.uniform(0.0, 0.2, size=half).astype(np.float32)
        for t in range(n_tok):
            ang = (freq * np.float32(p0 + t)).astype(np.float32)
            cos_half = np.cos(ang).astype(bfloat16)
            sin_half = np.sin(ang).astype(bfloat16)
            cos = np.concatenate([cos_half, cos_half])
            sin = np.concatenate([sin_half, sin_half])
            cs_perpos[t, L] = np.concatenate([cos, sin])
        layers.append(layer)
    return x_i8, res_scale, layers, cs_perpos


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
        )
        layers[L]["scales"] = scales
    xo_i8, xo_scale = x_cur
    normed_i8, norm_scale = numpy_rmsnorm_int8_dyn(xo_i8, gamma, np.float32(xo_scale))
    acc = embed_i8.astype(np.int64) @ normed_i8.astype(np.int64)
    logits = (
        acc.astype(np.float32) * np.float32(norm_scale) * embed_sc.astype(np.float32)
    ).astype(np.float32)
    return logits


def chain_forward_grow(x_i8, res_scale, layers, position, embed_i8, embed_sc, gamma):
    """Greedy variant: forward -> token + next-token embed seed (used only to
    populate layer['scales'] in the throwaway trajectory)."""
    logits = chain_forward_logits(
        x_i8, res_scale, layers, position, embed_i8, embed_sc, gamma
    )
    tok = topk_sample_reference(
        logits, SAMPLE_TEMP, SAMPLE_TOPK, SAMPLE_SEED, k_set=ONESTREAM_KSET
    )
    xin, scale = embed_seed_ref(embed_i8[tok], float(embed_sc[tok]))
    return tok, xin, scale


def pack_kv_prefix(kvblob, t_used):
    """Overwrite each per-KV-head T_used prefix in a packed kvblob."""
    out = kvblob.copy()
    tub = np.frombuffer(np.int32(t_used).tobytes(), dtype=np.int8)
    for L in range(N_LAYERS):
        for h in range(N_HEADS_KV):
            off = L * PER_LAYER_KV + h * PER_KV_HEAD_BYTES
            out[off : off + 4] = tub
    return out


DDR_BW_GBPS = 120.0  # NPU2 LPDDR5 ~120 GB/s
LMW_GAMMA_BYTES = 2 * D


def time_decode(npu, buffers, opts):
    """Per-token device time + DMA floor (host-ferried grow: KV streamed every
    token, both directions, so bytes/token includes 2*KV_BYTES)."""
    print(
        f"timing: warmup x{opts.n_warmup}, time x{opts.n_iters}  (PT={PT} tokens/dispatch)",
        flush=True,
    )
    for _ in range(opts.n_warmup):
        DefaultNPURuntime.load_and_run(npu, buffers)
    times_ms = []
    for _ in range(opts.n_iters):
        _h, result = DefaultNPURuntime.load_and_run(npu, buffers)
        times_ms.append(result.npu_time / 1e6)
    arr = np.array(times_ms)
    disp_med = float(np.median(arr))
    per_token_ms = disp_med / PT
    bytes_per_token = (
        WEIGHTS_BYTES + LMW_GAMMA_BYTES + WLM_TOTAL + 2 * KV_BYTES  # KV in+out
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
# logit spread (gap/std): flowkv_mh's ~1-ULP noise accumulates with depth, so
# an absolute delta calibrated at N=2 is too tight at N=16. A flip within the
# top cluster is documented quality-neutral; one far down would be real drift.
TOL_GAP_STD = 0.15  # gap as a fraction of the logit std
TOL_RANK = 5  # dev_tok must be within the oracle top-K


def set_pristine(layers, pristine):
    kcaches, vcaches, kslot, vslot = pristine
    for L in range(N_LAYERS):
        layers[L]["kcaches"] = [c.copy() for c in kcaches[L]]
        layers[L]["vcaches"] = [c.copy() for c in vcaches[L]]
        layers[L]["k_scales_slot"] = kslot[L].copy()
        layers[L]["v_scales_slot"] = vslot[L].copy()


def load_kv(layers, kv):
    """Give the oracle a device cache (region A of the KV buffer): per KV head,
    a T_used word and pad, then K's per-slot fp32 scales and int8 body, then V's."""
    for L in range(N_LAYERS):
        for h in range(N_HEADS_KV):
            k_at = L * PER_LAYER_KV + h * PER_KV_HEAD_BYTES + T_USED_BYTES
            for at, body, scale in (
                (k_at, "kcaches", "k_scales_slot"),
                (k_at + KCACHE_PADDED, "vcaches", "v_scales_slot"),
            ):
                layers[L][scale][h] = np.frombuffer(
                    kv[at : at + KV_HEADER].tobytes(), np.float32
                )
                layers[L][body][h] = kv[
                    at + KV_HEADER : at + KV_HEADER + T * HEAD_D
                ].copy()


def check_carry(before, after, pos, n_new):
    """The cache one dispatch carries to the next: every T_used advanced to
    pos+n_new, and nothing outside slots pos..pos+n_new-1 changed. Returns the
    number of failures."""
    fails = 0
    appended = np.zeros(KV_BYTES, dtype=bool)
    for L in range(N_LAYERS):
        for h in range(N_HEADS_KV):
            tu = L * PER_LAYER_KV + h * PER_KV_HEAD_BYTES
            appended[tu : tu + T_USED_BYTES] = True
            t_used = int(np.frombuffer(after[tu : tu + 4].tobytes(), np.int32)[0])
            if t_used != pos + n_new:
                print(f"  L{L} h{h}: T_used={t_used}, want {pos + n_new}")
                fails += 1
            for at in (tu + T_USED_BYTES, tu + T_USED_BYTES + KCACHE_PADDED):
                appended[at + 4 * pos : at + 4 * (pos + n_new)] = True
                b0 = at + KV_HEADER + HEAD_D * pos
                appended[b0 : b0 + HEAD_D * n_new] = True
    changed = int((before[:KV_BYTES] != after[:KV_BYTES])[~appended].sum())
    if changed:
        print(f"  {changed} bytes outside the appended slots changed")
    return fails + changed


def replay(dev_tokens, x_i8, res_scale, layers, p0, cs_rows, lm):
    """Check each device token against the numpy oracle, from the caches the
    caller set and position p0 on, following the device's tokens; returns the
    number that fail."""
    embed_i8, embed_sc, gamma = lm
    fails = 0
    cur_x, cur_sc = x_i8, res_scale
    for t, dev_tok in enumerate(dev_tokens):
        for L in range(N_LAYERS):
            layers[L]["cos"] = cs_rows[t, L, :HEAD_D]
            layers[L]["sin"] = cs_rows[t, L, HEAD_D:]
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


def tokens_of(packed):
    return [
        int(
            np.frombuffer(
                packed[t * (D + 12) + D + 4 : t * (D + 12) + D + 8].tobytes(), np.int32
            )[0]
        )
        for t in range(PT)
    ]


def main():
    p = argparse.ArgumentParser()
    add_artifact_args(p)
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
    opts = p.parse_args()
    fseed = opts.seed
    n_disp = opts.dispatches
    n_tok = n_disp * PT
    p0 = opts.start_pos if opts.start_pos is not None else min(T // 2, T - n_tok)
    if p0 < 1 or p0 + n_tok > T:
        p.error(f"{n_tok} tokens from position {p0} do not fit in T={T}")
    if n_disp > 1 and PT % 2:
        p.error("--dispatches needs an even PT: an odd PT ends the cache in region B")
    if n_disp > 1 and opts.repeats > 1:
        p.error("--dispatches and --repeats are separate modes")

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

    x_i8, res_scale, layers, cs_perpos = build_fixtures(fseed, p0, n_tok)

    # Snapshot the PRISTINE caches (before any append) so the device gets the same
    # starting KV the numpy loop starts from. The device's growing-append then
    # accumulates the tokens on-chip exactly as numpy does in place.
    pristine = (
        [[c.copy() for c in lyr["kcaches"]] for lyr in layers],
        [[c.copy() for c in lyr["vcaches"]] for lyr in layers],
        [lyr["k_scales_slot"].copy() for lyr in layers],
        [lyr["v_scales_slot"].copy() for lyr in layers],
    )

    # One throwaway oracle pass (greedy argmax trajectory) just to populate
    # layer["scales"] for pack_blobs; the authoritative per-token check is the
    # device-trajectory replay (greedy argmax on random fixtures is brittle near
    # a near-tie -- the flowkv ~1-ULP non-bit-exactness can flip it, which is
    # documented quality-neutral). We don't compare against this trajectory.
    cur_x, cur_sc = x_i8, res_scale
    for t in range(PT):
        for L in range(N_LAYERS):
            layers[L]["cos"] = cs_perpos[t, L, :HEAD_D]
            layers[L]["sin"] = cs_perpos[t, L, HEAD_D:]
        _, nxin, nscale = chain_forward_grow(
            cur_x, cur_sc, layers, p0 + t, embed_i8, embed_sc, gamma
        )
        cur_x, cur_sc = nxin, nscale

    # Pack wblob with the (now-populated) real scales; the chain self-calibrates
    # on-device so these are placeholders, but pack_blobs requires the key. The
    # static OFF_CS cos/sin in wblob is unused by PERSIST_GROW (the per-position
    # block drives rope), so its value doesn't matter.
    wblob, _ = pack_blobs(layers, WEIGHTS_BYTES)
    assert wblob.size == WEIGHTS_BYTES, (wblob.size, WEIGHTS_BYTES)

    def wblob_for(d):
        """wblob with dispatch d's per-position cos/sin block at OFF_CS_PERPOS,
        laid out [token0: L0..L_{N-1} | token1: ... ] matching the chain's fill."""
        w = wblob.copy()
        for t in range(PT):
            for L in range(N_LAYERS):
                off = OFF_CS_PERPOS + (t * N_LAYERS + L) * CS_BYTES
                w[off : off + CS_BYTES] = cs_perpos[d * PT + t, L].view(np.int8)
        return w

    # The PRISTINE kvblob (restore caches, pack, T_used=p0 prefix). The
    # grow-append writes slot=T_used so token 0 writes slot p0 and advances
    # T_used -> p0+1.
    for L in range(N_LAYERS):
        layers[L]["kcaches"] = pristine[0][L]
        layers[L]["vcaches"] = pristine[1][L]
        layers[L]["k_scales_slot"] = pristine[2][L]
        layers[L]["v_scales_slot"] = pristine[3][L]
    _, kvblob_pristine = pack_blobs(layers, WEIGHTS_BYTES)
    kv_pristine = pack_kv_prefix(kvblob_pristine, p0)

    # The KV buffer is 2x [regionA | regionB]; token t fills region t%2 and
    # drains to the other, so after an even PT the grown cache is back in
    # region A, where the next dispatch's token 0 reads it.
    xin = np.zeros(D + 8, dtype=np.int8)
    xin[:D] = x_i8
    xin[D : D + 4] = np.frombuffer(np.float32(res_scale).tobytes(), dtype=np.int8)
    kv2 = np.zeros(2 * KV_BYTES, dtype=np.int8)
    kv2[:KV_BYTES] = kv_pristine

    npu = design_kernel(
        build,
        opts,
        SAMPLE=True,
        ONESTREAM=True,
        PERSIST=True,
        PERSIST_GROW=True,
        PT=PT,
        N_LAYERS=N_LAYERS,
        T=T,
        ONESTREAM_KSET=ONESTREAM_KSET,
    )
    lmw_t = iron.tensor(lmw, dtype=np.int8)
    print(
        f"chain_persist_grow_mh N_LAYERS={N_LAYERS} PT={PT} P0={p0} KSET={ONESTREAM_KSET}"
        f"  temp={SAMPLE_TEMP} topk={SAMPLE_TOPK} seed={SAMPLE_SEED}  fixture_seed={fseed}"
        f"  dispatches={n_disp} repeats={opts.repeats}",
        flush=True,
    )
    if opts.time:
        return time_decode(
            npu,
            [
                iron.tensor(xin, dtype=np.int8),
                iron.tensor(wblob_for(0), dtype=np.int8),
                iron.tensor(kv2, dtype=np.int8),
                lmw_t,
                iron.zeros([PT * (D + 12)], dtype=np.int8),
            ],
            opts,
        )

    # Each dispatch gets its own input, weight and output tensors; the KV
    # tensor is created once per run of dispatches and stays on the device,
    # so a generation loop's dispatches carry the cache with no host copy.
    def dispatch(xin_d, w_d, kv_t):
        out_t = iron.zeros([PT * (D + 12)], dtype=np.int8)
        DefaultNPURuntime.load_and_run(
            npu,
            [
                iron.tensor(xin_d, dtype=np.int8),
                iron.tensor(w_d, dtype=np.int8),
                kv_t,
                lmw_t,
                out_t,
            ],
        )
        out_t.to("cpu")
        return out_t.numpy().copy()

    # The oracle checks each dispatch from what that dispatch actually started
    # with -- the pristine cache and fixture seed for the first, the cache and
    # seed the device carried for the rest -- so dispatch k is held to the same
    # standard as a single dispatch. (One oracle trajectory over many dispatches
    # would compound flowkv's documented non-bit-exactness token after token.)
    dev_tokens = []
    tok_fails = carry_fails = 0
    if opts.repeats > 1:
        for r in range(opts.repeats):
            toks = tokens_of(
                dispatch(xin, wblob_for(0), iron.tensor(kv2, dtype=np.int8))
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
        tok_fails = replay(dev_tokens, x_i8, res_scale, layers, p0, cs_perpos, lm)
    else:
        kv_t = iron.tensor(kv2, dtype=np.int8)
        carried = kv2[:KV_BYTES].copy()
        x_d, sc_d, xin_d = x_i8, res_scale, xin
        for d in range(n_disp):
            toks_out = dispatch(xin_d, wblob_for(d), kv_t)
            toks = tokens_of(toks_out)
            dev_tokens += toks
            pos = p0 + d * PT
            if n_disp > 1:
                print(f"dispatch {d} (pos {pos}): tokens {toks}", flush=True)
            if d == 0:
                set_pristine(layers, pristine)
            else:
                load_kv(layers, carried)
            rows = cs_perpos[d * PT : (d + 1) * PT]
            tok_fails += replay(toks, x_d, sc_d, layers, pos, rows, lm)
            if n_disp > 1:
                kv_t.to("cpu")
                after = kv_t.numpy()[:KV_BYTES].copy()
                cf = check_carry(carried, after, pos, PT)
                print(f"  carry: {'PASS' if not cf else f'{cf} FAIL'}", flush=True)
                carry_fails += cf
                carried = after
            # The next dispatch's layer-0 input is the last token's seed (body +
            # scale; zero pad), exactly what the chain fed back on-chip.
            last = (PT - 1) * (D + 12)
            xin_d = np.zeros(D + 8, dtype=np.int8)
            xin_d[: D + 4] = toks_out[last : last + D + 4]
            x_d = xin_d[:D].copy()
            sc_d = float(np.frombuffer(xin_d[D : D + 4].tobytes(), np.float32)[0])

    n = len(dev_tokens)
    print(
        f"\nchain_persist_grow_mh: {n - tok_fails}/{n} tokens PASS  "
        f"(on-chip feedback + GROWING KV + per-position rope)"
    )
    if n_disp > 1:
        print(
            f"carry across {n_disp} dispatches: {'PASS' if not carry_fails else 'FAIL'}"
        )
    return 0 if tok_fails == 0 and carry_fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
