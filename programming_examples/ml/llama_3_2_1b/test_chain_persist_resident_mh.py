"""Persistent-loop RESIDENT-KV test (capstone, toward 100% NPU): PT tokens in ONE
dispatch, token feeding back on-chip, the KV cache GROWS across tokens, AND the
cache BODY is resident on the attn compute tiles -- the host streams ONLY weights
after a one-time token-0 seed (no per-token KV DMA).

Same growing autoregressive decode + per-position rope as the host-ferried grow
test, but the KV cache lives in worker-local b_kv buffers (N_LAYERS caches/head)
read-modify-written in place. The host supplies a 1x pristine KV blob used ONLY to
seed the resident cache on token 0; there is no KV drain. The growing-append
(llama_kv_append_combined_resident*) advances T_used on-chip each token.

token-0 KV prefix T_used = P (NOT P+1): the grow-append's slot = T_used, so token
0 writes slot P and sets T_used = P+1 (flowkv attends P+1, incl the new slot).

Run:
  make chain_persist_resident_mh CHAIN_MH_N=2 PT=4 ONESTREAM_KSET=8
  python test_chain_persist_resident_mh.py -x build/final_chain_persist_resident_N2_PT4.xclbin \\
      -i build/insts_chain_persist_resident_N2_PT4.bin -k MLIR_AIE
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_chain_dynscale_mh import (
    D,
    T,
    N_LAYERS,
    N_HEADS_KV,
    HEAD_D,
    VOCAB,
    N_TILE,
    KV_BYTES,
    PER_LAYER_KV,
    PER_KV_HEAD_BYTES,
    WLM_TOTAL,
    WEIGHTS_BYTES,
    OFF_CS_PERPOS,
    CS_BYTES,
    SAMPLE_TEMP,
    SAMPLE_TOPK,
    SAMPLE_SEED,
    ONESTREAM_KSET,
    PT,
)
from ml_dtypes import bfloat16
from numpy_layer_mh import gen_layer_mh, numpy_layer_mh_forward, requant
from numpy_llama_ref import _load_qw, _load_bf16, VOCAB_SIZE, EMB_DIM
from test_rmsnorm_int8_dyn import numpy_rmsnorm_int8_dyn
from test_ffn_half import pack_perchan_slots
from numpy_topk_sample import topk_sample_reference
from gen_exp_lut import exp_lut
from gen_silu_lut import silu_lut
from test_flowkv import EXP_QUANT_SCALE as FKV_EXP_QUANT_SCALE
from aie2_chain_dynscale_mh import SILU_GATE_SCALE
from test_chain_mh import pack_blobs

DATA_DIR = Path(__file__).parent / "data"


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


def build_fixtures(seed):
    rng = np.random.default_rng(seed)
    x_fp = rng.uniform(-1.6, 1.6, size=D).astype(np.float32)
    res_scale = np.float32(np.maximum(np.abs(x_fp).max(), 1e-12) / 127.0)
    x_i8 = requant(x_fp, np.float32(1.0) / res_scale)

    lut_exp = exp_lut(FKV_EXP_QUANT_SCALE).astype(np.float32)
    lut_silu = silu_lut(SILU_GATE_SCALE)
    # Start position P0: leave room for PT appended tokens within T.
    P0 = T // 2
    half = HEAD_D // 2
    layers = []
    # Per-layer base rotary frequencies (random per layer, but the PHASE advances
    # with position so each token gets a distinct (cos,sin) -- real rope behavior).
    cs_perpos = np.zeros((PT, N_LAYERS, 2 * HEAD_D), dtype=bfloat16)  # [cos|sin]
    for L in range(N_LAYERS):
        layer = gen_layer_mh(rng)
        layer["lut_exp"] = lut_exp
        layer["lut_silu"] = lut_silu
        k_slot = rng.uniform(0.02, 0.08, size=(N_HEADS_KV, T)).astype(np.float32)
        v_slot = rng.uniform(0.02, 0.08, size=(N_HEADS_KV, T)).astype(np.float32)
        for h in range(N_HEADS_KV):
            for p in range(P0, P0 + PT):
                layer["kcaches"][h][p * HEAD_D : (p + 1) * HEAD_D] = 0
                layer["vcaches"][h][p * HEAD_D : (p + 1) * HEAD_D] = 0
                k_slot[h, p] = 0.0
                v_slot[h, p] = 0.0
        layer["k_scales_slot"] = k_slot
        layer["v_scales_slot"] = v_slot
        # Per-layer base frequencies; position t's angle = freq * (P0+t).
        freq = rng.uniform(0.0, 0.2, size=half).astype(np.float32)
        for t in range(PT):
            ang = (freq * np.float32(P0 + t)).astype(np.float32)
            cos_half = np.cos(ang).astype(bfloat16)
            sin_half = np.sin(ang).astype(bfloat16)
            cos = np.concatenate([cos_half, cos_half])
            sin = np.concatenate([sin_half, sin_half])
            cs_perpos[t, L] = np.concatenate([cos, sin])
        layers.append(layer)
    return x_i8, res_scale, layers, P0, cs_perpos


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


DDR_BW_GBPS = 120.0  # NPU2 LPDDR5 ~120 GB/s (reference; the DMA floor denominator)
LMW_GAMMA_BYTES = 2 * D  # final_norm gamma bf16[D], prefix of the lm_head blob


def time_decode(npu, buffers, opts):
    """Measure per-token device time and compare to the per-token DMA floor.

    The persistent dispatch generates PT tokens, streaming the model from DDR each
    token: WEIGHTS_BYTES (N-layer blob) + lm_head (LMW_GAMMA_BYTES + WLM_TOTAL,
    262 MB) every token; the KV seed (KV_BYTES) is streamed ONCE (token 0) in
    resident mode, so it amortizes to KV_BYTES/PT. per_token = npu_time/PT.
    ratio = per_token / dma_floor: ~1 => DMA-bound (weight stream is the wall,
    prefetch working); >>1 => compute-bound (kernels starve the stream)."""
    print(
        f"timing: warmup x{opts.n_warmup}, time x{opts.n_iters}  (PT={PT} tokens/dispatch)",
        flush=True,
    )
    for _ in range(opts.n_warmup):
        DefaultNPURuntime.load_and_run(npu, buffers)
    times_ms = []
    for _ in range(opts.n_iters):
        _h, result = DefaultNPURuntime.load_and_run(npu, buffers)
        times_ms.append(result.npu_time / 1e6)  # ns -> ms
    arr = np.array(times_ms)
    disp_med = float(np.median(arr))
    per_token_ms = disp_med / PT

    bytes_per_token = (
        WEIGHTS_BYTES  # N-layer weight blob, streamed every token
        + LMW_GAMMA_BYTES
        + WLM_TOTAL  # 262 MB tied lm_head / embed table, every token
        + KV_BYTES / PT  # resident KV seed: streamed once, amortized over PT
    )
    dma_floor_ms = bytes_per_token / (DDR_BW_GBPS * 1e9) * 1e3
    ratio = per_token_ms / dma_floor_ms
    verdict = (
        "DMA-bound (weight stream is the wall; prefetch working)"
        if ratio < 1.5
        else "COMPUTE-bound (kernels starve the DMA; vectorization is the lever)"
    )
    print(
        f"per-dispatch: n={opts.n_iters} median={disp_med:.2f} ms "
        f"min={arr.min():.2f} max={arr.max():.2f} std={arr.std():.2f}"
    )
    print(f"per-token:    {per_token_ms:.2f} ms  ({1000.0/per_token_ms:.1f} tok/s)")
    print(
        f"bytes/token:  {bytes_per_token/1e6:.1f} MB  "
        f"(weights {WEIGHTS_BYTES/1e6:.1f} + lm_head {(LMW_GAMMA_BYTES+WLM_TOTAL)/1e6:.1f} "
        f"+ KV/PT {KV_BYTES/PT/1e6:.2f})"
    )
    print(f"DMA floor:    {dma_floor_ms:.2f} ms/token @ {DDR_BW_GBPS:.0f} GB/s")
    print(f"ratio:        {ratio:.2f}x  -> {verdict}")
    return 0


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--time",
        action="store_true",
        help="measure per-token device time (DMA-bound vs compute-bound verdict)",
    )
    # Each persistent dispatch is ~400 ms (compute-bound); back-to-back dispatches
    # can exceed XRT's wait() timeout, so default to a single timed dispatch.
    # result.npu_time is the device clock, so one clean dispatch is enough.
    p.add_argument("--n-iters", type=int, default=1)
    p.add_argument("--n-warmup", type=int, default=0)
    opts = p.parse_args()
    fseed = opts.seed

    assert VOCAB == VOCAB_SIZE and D == EMB_DIM
    print(f"loading embed/final_norm from {DATA_DIR} ...", flush=True)
    embed_i8, embed_sc = _load_qw(DATA_DIR, "embed", (VOCAB, D))
    gamma = _load_bf16(DATA_DIR, "final_norm", (D,))
    print("packing lm_head weights (262 MB) ...", flush=True)
    lmw = pack_lmw(embed_i8, embed_sc, gamma)

    x_i8, res_scale, layers, P0, cs_perpos = build_fixtures(fseed)

    # Snapshot the PRISTINE caches (before any append) so the device gets the same
    # starting KV the numpy loop starts from. The device's growing-append then
    # accumulates the PT tokens on-chip exactly as numpy does in place.
    pristine_kcaches = [[c.copy() for c in lyr["kcaches"]] for lyr in layers]
    pristine_vcaches = [[c.copy() for c in lyr["vcaches"]] for lyr in layers]
    pristine_kslot = [lyr["k_scales_slot"].copy() for lyr in layers]
    pristine_vslot = [lyr["v_scales_slot"].copy() for lyr in layers]

    # One throwaway oracle pass (greedy argmax trajectory) just to populate
    # layer["scales"] for pack_blobs; the authoritative per-token check is the
    # device-trajectory replay below (greedy argmax on random fixtures is brittle
    # near a near-tie -- the flowkv ~1-ULP non-bit-exactness can flip it, which is
    # documented quality-neutral). We don't compare against this trajectory.
    cur_x, cur_sc = x_i8, res_scale
    for t in range(PT):
        for L in range(N_LAYERS):
            layers[L]["cos"] = cs_perpos[t, L, :HEAD_D]
            layers[L]["sin"] = cs_perpos[t, L, HEAD_D:]
        _, nxin, nscale = chain_forward_grow(
            cur_x, cur_sc, layers, P0 + t, embed_i8, embed_sc, gamma
        )
        cur_x, cur_sc = nxin, nscale

    # Pack wblob with the (now-populated) real scales; the chain self-calibrates
    # on-device so these are placeholders, but pack_blobs requires the key. The
    # static OFF_CS cos/sin in wblob is unused by PERSIST_GROW (per-position block
    # below drives rope), so its value doesn't matter.
    wblob, _ = pack_blobs(layers)
    # Append the per-position cos/sin block at OFF_CS_PERPOS, laid out
    # [token0: L0..L_{N-1} | token1: ... ] matching the chain's (tok,L) fill.
    assert wblob.size == WEIGHTS_BYTES, (wblob.size, WEIGHTS_BYTES)
    for t in range(PT):
        for L in range(N_LAYERS):
            off = OFF_CS_PERPOS + (t * N_LAYERS + L) * CS_BYTES
            wblob[off : off + CS_BYTES] = cs_perpos[t, L].view(np.int8)

    # Build the PRISTINE kvblob from the snapshot (restore caches, pack, set
    # T_used=P0 prefix). The grow-append writes slot=T_used so token 0 writes
    # slot P0 and advances T_used -> P0+1.
    for L in range(N_LAYERS):
        layers[L]["kcaches"] = pristine_kcaches[L]
        layers[L]["vcaches"] = pristine_vcaches[L]
        layers[L]["k_scales_slot"] = pristine_kslot[L]
        layers[L]["v_scales_slot"] = pristine_vslot[L]
    _, kvblob_pristine = pack_blobs(layers)
    kv_pristine = pack_kv_prefix(kvblob_pristine, P0)

    # Device dispatch. KV buffer is 1x (the resident seed); host fills it on token 0 only
    # = the pristine cache (T_used=P0). Growing-append advances on-chip.
    xin = np.zeros(D + 8, dtype=np.int8)
    xin[:D] = x_i8
    xin[D : D + 4] = np.frombuffer(np.float32(res_scale).tobytes(), dtype=np.int8)
    kv2 = np.zeros(KV_BYTES, dtype=np.int8)
    kv2[:] = kv_pristine

    npu = test_utils.create_npu_kernel(opts).npu_kernel
    x_t = iron.tensor(xin, dtype=np.int8)
    w_t = iron.tensor(wblob, dtype=np.int8)
    kv_t = iron.tensor(kv2, dtype=np.int8)
    lmw_t = iron.tensor(lmw, dtype=np.int8)
    out_t = iron.zeros([PT * (D + 12)], dtype=np.int8)
    print(
        f"chain_persist_resident_mh N_LAYERS={N_LAYERS} PT={PT} P0={P0} KSET={ONESTREAM_KSET}"
        f"  temp={SAMPLE_TEMP} topk={SAMPLE_TOPK} seed={SAMPLE_SEED}  fixture_seed={fseed}",
        flush=True,
    )
    if opts.time:
        return time_decode(npu, [x_t, w_t, kv_t, lmw_t, out_t], opts)

    rc = DefaultNPURuntime.run_test(
        npu, [x_t, w_t, kv_t, lmw_t, out_t], {}, verify=False, verbosity=opts.verbosity
    )
    if rc != 0:
        print(f"dispatch returned {rc}", file=sys.stderr)
        return rc
    out_t.to("cpu")
    packed = out_t.numpy()
    dev_tokens = [
        int(
            np.frombuffer(
                packed[t * (D + 12) + D + 4 : t * (D + 12) + D + 8].tobytes(), np.int32
            )[0]
        )
        for t in range(PT)
    ]

    # Device-trajectory replay with a near-tie tolerance. Because the token feeds
    # back ON-CHIP, the oracle must follow the DEVICE's tokens (not its own greedy
    # trajectory). At each step we recompute the oracle logits for the device's
    # input, then accept the device token if it is the oracle argmax OR its logit
    # is within TOL of the true max -- the flowkv ~1-ULP non-bit-exactness
    # (llama_flowkv_mh.cc: "NOT byte-exact... flipping an exp-LUT bucket... proven
    # quality-neutral") can flip greedy argmax only when the top logits are
    # near-tied. A flip OUTSIDE TOL would be a real divergence.
    TOL = np.float32(0.10)  # logit units; flowkv ULP noise is <~0.05 (see diag)
    # Restore pristine caches for the replay (the throwaway pass mutated them).
    for L in range(N_LAYERS):
        layers[L]["kcaches"] = [c.copy() for c in pristine_kcaches[L]]
        layers[L]["vcaches"] = [c.copy() for c in pristine_vcaches[L]]
        layers[L]["k_scales_slot"] = pristine_kslot[L].copy()
        layers[L]["v_scales_slot"] = pristine_vslot[L].copy()

    fails = 0
    cur_x, cur_sc = x_i8, res_scale
    for t in range(PT):
        for L in range(N_LAYERS):
            layers[L]["cos"] = cs_perpos[t, L, :HEAD_D]
            layers[L]["sin"] = cs_perpos[t, L, HEAD_D:]
        logits = chain_forward_logits(
            cur_x, cur_sc, layers, P0 + t, embed_i8, embed_sc, gamma
        )
        dev_tok = dev_tokens[t]
        true_max = np.float32(logits.max())
        ref_argmax = int(np.argmax(logits))
        gap = float(true_max - logits[dev_tok])
        ok = (dev_tok == ref_argmax) or (gap <= float(TOL))
        fails += not ok
        tag = "exact" if dev_tok == ref_argmax else f"near-tie gap={gap:.4f}"
        print(
            f"  token {t} (pos {P0+t}): {'PASS' if ok else 'FAIL'}  "
            f"dev={dev_tok} ref_argmax={ref_argmax}  [{tag}]"
        )
        # Follow the DEVICE token forward (on-chip feedback trajectory).
        nx, nscale = embed_seed_ref(embed_i8[dev_tok], float(embed_sc[dev_tok]))
        cur_x, cur_sc = nx, nscale

    print(
        f"\nchain_persist_resident_mh: {PT - fails}/{PT} tokens PASS  "
        f"(on-chip feedback + RESIDENT GROWING KV + per-position rope, weights-only host)"
    )
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
