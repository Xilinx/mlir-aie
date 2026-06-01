"""Phase 6c.5b.3: attention-half bit-exact test with per-channel weight
quant + per-token dynamic activation scales.

Random gamma + fp32 wq/wo (per-channel-quantized) + cs + host-pre-filled
K/V cache. Per-seed calibration computes every dynamic scale from a
numpy forward pass; scales are packed into the wblob slot prefixes and
kvblob headers; xclbin is built ONCE and dispatched for each seed.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_attn_half import (
    D, HEAD_D, N_HEADS, N_KV, QD, KVD, T, N_TILE,
    WQ_SLOT, WO_SLOT, N_TILES_Q, N_TILES_O,
    WQ_TOTAL, WO_TOTAL,
    OFF_GAMMA, OFF_WQ, OFF_CS, OFF_WO,
    GAMMA_BYTES, CS_BYTES,
    KCACHE_BYTES, VCACHE_BYTES,
    KCACHE_PADDED, VCACHE_PADDED, KV_HEADER,
    OFF_K, OFF_V, KV_BYTES, WEIGHTS_BYTES,
    ACT_SCALE, INV_ACT_SCALE,
)
from test_rmsnorm_int8 import numpy_rmsnorm_int8
from test_rope_int8 import numpy_rope
from test_flowkv import numpy_attention, EXP_QUANT_SCALE, quant_shifted, sw_recip
from gen_exp_lut import exp_lut
from gen_llama_data import quant_int8_perchan_absmax
from test_ffn_half import pack_perchan_slots, fp32_bytes, numpy_gemm_perchan, i8_add_wrap


def compute_sv_fp(qr, kcache, vcache, head_dim, t, q_scale, k_scale, v_scale, lut):
    """Replicate numpy_attention's fp32 sv computation up to the final
    (* inv_out_scale + round) — returns the unrounded fp32 sv vector so
    the caller can derive sv_out_scale = max(|sv|)/127."""
    qs = np.float32(q_scale); ks = np.float32(k_scale)
    inv_sqrt = np.float32(0.125)
    qk_scale = ((qs * ks).astype(np.float32) * inv_sqrt).astype(np.float32)
    k_mat = kcache.astype(np.int32).reshape(t, head_dim)
    dots = (k_mat @ qr.astype(np.int32))
    scores = (dots.astype(np.float32) * qk_scale)
    max_s = np.float32(scores.max())
    shifted = (scores - max_s).astype(np.float32)
    q = quant_shifted(shifted)
    exp_v = lut[q + 128].astype(np.float32)
    sum_e = np.float32(0.0)
    for i in range(t):
        sum_e = (sum_e + exp_v[i]).astype(np.float32)
    inv_sum = sw_recip(sum_e)
    probs = (exp_v * inv_sum).astype(np.float32)
    v_mat = vcache.astype(np.int32).reshape(t, head_dim).astype(np.float32)
    acc = np.zeros(head_dim, dtype=np.float32)
    for i in range(t):
        acc = (acc + (probs[i] * v_mat[i])).astype(np.float32)
    return (acc * np.float32(v_scale)).astype(np.float32)


def run_one_seed(seed: int, opts, lut_bf, npu_kernel) -> int:
    rng = np.random.default_rng(seed)
    active = os.environ.get("LLAMA_ATTN_ACTIVE", "0") == "1"

    # --- Inputs / parameters ---
    x_in     = rng.integers(-32, 33, size=D, dtype=np.int8)
    gamma_in = (1.0 + 0.1 * rng.standard_normal(D).astype(np.float32)).astype(bfloat16)

    # Random fp32 weights with heterogeneous per-row magnitudes (gives a
    # non-trivial per-channel scale distribution).
    def random_w(out_dim, in_dim):
        base = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.05
        row_scale = rng.uniform(0.1, 1.0, size=out_dim).astype(np.float32)
        return base * row_scale[:, None]
    wq_fp = random_w(QD, D);  bq = np.zeros(QD, dtype=np.int32)
    wo_fp = random_w(D,  QD); bo = np.zeros(D,  dtype=np.int32)
    wq_i8, wq_sc = quant_int8_perchan_absmax(wq_fp)
    wo_i8, wo_sc = quant_int8_perchan_absmax(wo_fp)

    # RoPE cos/sin: doubled-halves over a random angle per pair (Llama-3).
    half = HEAD_D // 2
    ang = rng.uniform(0, 2 * np.pi, size=half).astype(np.float32)
    cos_half = np.cos(ang).astype(bfloat16)
    sin_half = np.sin(ang).astype(bfloat16)
    cos = np.concatenate([cos_half, cos_half])
    sin = np.concatenate([sin_half, sin_half])

    # Random pre-filled KV cache (T x KVD bytes each).
    if active:
        kcache_lim, vcache_lim = 33, 33
    else:
        kcache_lim, vcache_lim = 33, 33
    kcache = rng.integers(-kcache_lim, kcache_lim, size=T * KVD, dtype=np.int8)
    vcache = rng.integers(-vcache_lim, vcache_lim, size=T * KVD, dtype=np.int8)

    # K/V scales: keep fixed at 0.05 (mirrors 6c.3b.2; what matters here
    # is that the kernel now READS them from the kvblob header instead of
    # closure baking).
    k_scale = 0.05
    v_scale = 0.05

    # --- Numpy reference: forward pass + per-stage scale calibration ---
    lut_exp = exp_lut(EXP_QUANT_SCALE).astype(np.float32)
    h1 = numpy_rmsnorm_int8(x_in, gamma_in, ACT_SCALE, INV_ACT_SCALE)

    fp_q = (wq_i8.astype(np.int32) @ h1.astype(np.int32) + bq).astype(np.float32) \
           * np.float32(ACT_SCALE) * wq_sc.astype(np.float32)
    q_out_scale = float(np.maximum(np.abs(fp_q).max(), 1e-12)) / 127.0
    q_inv_out   = float(np.float32(1.0) / np.float32(q_out_scale))
    def requant(fp, inv):
        s = fp * np.float32(inv)
        r = np.where(s >= 0, np.floor(s + np.float32(0.5)),
                              np.ceil(s - np.float32(0.5)))
        return r.clip(-128, 127).astype(np.int8)
    qf = requant(fp_q, q_inv_out)

    qr = numpy_rope(qf, cos, sin, N_HEADS, HEAD_D, q_out_scale)

    sv_fp = compute_sv_fp(qr, kcache, vcache, HEAD_D, T,
                          q_out_scale, k_scale, v_scale, lut_exp)
    sv_out_scale       = float(np.maximum(np.abs(sv_fp).max(), 1e-12)) / 127.0
    sv_inv_out_scale   = float(np.float32(1.0) / np.float32(sv_out_scale))
    af = numpy_attention(qr, kcache, vcache, HEAD_D, T,
                         q_out_scale, k_scale, v_scale, sv_inv_out_scale,
                         lut_exp)

    op = numpy_gemm_perchan(af, sv_out_scale, wo_i8, wo_sc, bo, INV_ACT_SCALE)
    expected = i8_add_wrap(op, x_in)

    # --- Pack weights blob ---
    wblob = np.zeros(WEIGHTS_BYTES, dtype=np.int8)
    wblob[OFF_GAMMA:OFF_GAMMA + GAMMA_BYTES] = gamma_in.view(np.int8)
    # WQ slot prefix (64 B): (ACT_SCALE, q_inv_out, q_out_scale, spare) + 48 B pad
    wq_prefix = (fp32_bytes(ACT_SCALE, q_inv_out, q_out_scale, 0.0)
                 + b"\x00" * 48)
    wq_packed = pack_perchan_slots(wq_i8, wq_sc, bq, N_TILE, prefix_bytes=wq_prefix)
    assert wq_packed.size == N_TILES_Q * WQ_SLOT, (wq_packed.size, N_TILES_Q * WQ_SLOT)
    wblob[OFF_WQ:OFF_WQ + wq_packed.size] = wq_packed

    cs_packed = np.concatenate([cos, sin])
    wblob[OFF_CS:OFF_CS + CS_BYTES] = cs_packed.view(np.int8)

    # WO slot prefix (64 B): (sv_out_scale, INV_ACT_SCALE) + 56 B pad
    wo_prefix = fp32_bytes(sv_out_scale, INV_ACT_SCALE) + b"\x00" * 56
    wo_packed = pack_perchan_slots(wo_i8, wo_sc, bo, N_TILE, prefix_bytes=wo_prefix)
    assert wo_packed.size == N_TILES_O * WO_SLOT, (wo_packed.size, N_TILES_O * WO_SLOT)
    wblob[OFF_WO:OFF_WO + wo_packed.size] = wo_packed

    # --- Pack KV cache blob with 8 B headers ---
    kvblob = np.zeros(KV_BYTES, dtype=np.int8)
    # kcache header = (k_scale, spare)
    k_header = fp32_bytes(k_scale, 0.0)
    kvblob[OFF_K:OFF_K + KV_HEADER] = np.frombuffer(k_header, dtype=np.int8)
    kvblob[OFF_K + KV_HEADER:OFF_K + KV_HEADER + KCACHE_BYTES] = kcache
    # vcache header = (v_scale, sv_inv_out_scale)
    v_header = fp32_bytes(v_scale, sv_inv_out_scale)
    kvblob[OFF_V:OFF_V + KV_HEADER] = np.frombuffer(v_header, dtype=np.int8)
    kvblob[OFF_V + KV_HEADER:OFF_V + KV_HEADER + VCACHE_BYTES] = vcache

    # --- NPU dispatch ---
    x_t  = iron.tensor(x_in, dtype=np.int8)
    w_t  = iron.tensor(wblob, dtype=np.int8)
    kv_t = iron.tensor(kvblob, dtype=np.int8)
    o_t  = iron.zeros([D], dtype=np.int8)

    rc = DefaultNPURuntime.run_test(
        npu_kernel, [x_t, w_t, kv_t, o_t],
        {}, verify=False, verbosity=opts.verbosity,
    )
    if rc != 0:
        return rc
    o_t.to("cpu")
    actual = o_t.numpy()

    diff = actual.astype(np.int16) - expected.astype(np.int16)
    n_diff = int((diff != 0).sum())
    max_abs = int(np.abs(diff).max()) if n_diff else 0
    sat = int((expected == 127).sum() + (expected == -128).sum())
    print(f"  seed={seed:>3}  mismatches={n_diff}/{D}  max|diff|={max_abs}  "
          f"sat={sat}/{D}  q_inv={q_inv_out:.3f}  "
          f"sv_inv={sv_inv_out_scale:.3f}")
    if n_diff != 0:
        for i in np.argwhere(diff != 0).flatten()[:8]:
            print(f"    i={i}: NPU={actual[i]}  expected={expected[i]}  x={x_in[i]}")
    return n_diff


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated seed list (default: LLAMA_ATTN_SEED env or 0,1,7,42)")
    opts = p.parse_args()

    npu_kernel = test_utils.create_npu_kernel(opts).npu_kernel
    lut_bf = exp_lut(EXP_QUANT_SCALE).astype(np.float32)

    if opts.seeds is not None:
        seeds = [int(s) for s in opts.seeds.split(",") if s.strip()]
    elif "LLAMA_ATTN_SEED" in os.environ:
        seeds = [int(os.environ["LLAMA_ATTN_SEED"])]
    else:
        seeds = [0, 1, 7, 42]

    print(f"attn-half (perchan + dynamic-scale runtime) D={D} HEAD_D={HEAD_D} "
          f"T={T}  seeds={seeds}")
    print("  (same xclbin for every seed -- no rebuild)")

    n_fail = 0
    for s in seeds:
        rc = run_one_seed(s, opts, lut_bf, npu_kernel)
        if rc != 0:
            n_fail += 1
    if n_fail == 0:
        print(f"BIT-EXACT PASS x {len(seeds)}  (perchan + dynamic-scale attn-half)")
        return 0
    print(f"FAIL: {n_fail}/{len(seeds)} seeds had mismatches")
    return 1


if __name__ == "__main__":
    sys.exit(main())
