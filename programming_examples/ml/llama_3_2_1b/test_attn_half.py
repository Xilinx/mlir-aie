"""Phase 6c.3b.2: attention-half bit-exact test (D=2048, single-head, T=128).

Random gamma + wq + cs + wo + host-pre-filled K/V cache. Single-shot
xclbin dispatch runs rmsnorm + q_proj + rope_q + flowkv pair + o_proj +
add1. Bit-exact compare against the numpy attention replay used in
test_chain_real.numpy_single_layer (attention block only).
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
    GAMMA_BYTES, CS_BYTES, KCACHE_BYTES, VCACHE_BYTES,
    OFF_K, OFF_V, KV_BYTES, WEIGHTS_BYTES,
    RIGHT_SHIFT, ACT_SCALE, INV_ACT_SCALE,
    Q_SCALE, K_SCALE, V_SCALE, INV_OUT_SCALE_AT,
)
from test_rmsnorm_int8 import numpy_rmsnorm_int8
from test_gemm_int8_srs_real import banker_srs
from test_rope_int8 import numpy_rope
from test_flowkv import numpy_attention, EXP_QUANT_SCALE
from gen_exp_lut import exp_lut


def i8_add_wrap(a, b):
    return (a.astype(np.int32) + b.astype(np.int32)).astype(np.int8)


def numpy_gemm_srs(act, weights, bias, K, N, right_shift):
    acc = weights.astype(np.int32) @ act.astype(np.int32) + bias
    return banker_srs(acc, right_shift).clip(-128, 127).astype(np.int8)


def pack_tiled_weights(buf, off, weights, bias, n_tiles, n_tile, K):
    slot = n_tile * K + n_tile * 4
    for t in range(n_tiles):
        base = off + t * slot
        w_slice = weights[t * n_tile:(t + 1) * n_tile].flatten()
        b_slice = bias[t * n_tile:(t + 1) * n_tile].view(np.int8).flatten()
        buf[base:base + w_slice.size] = w_slice
        buf[base + w_slice.size:base + w_slice.size + b_slice.size] = b_slice


def main():
    p = test_utils.create_default_argparser()
    opts = p.parse_args()

    seed = int(os.environ.get("LLAMA_ATTN_SEED", "0"))
    rng = np.random.default_rng(seed)

    active = os.environ.get("LLAMA_ATTN_ACTIVE", "0") == "1"
    if active:
        wlim, blim = 33, 1000
    else:
        wlim, blim = 8, 100

    # --- Inputs / parameters ---
    x_in       = rng.integers(-32, 33, size=D, dtype=np.int8)
    gamma_in   = (1.0 + 0.1 * rng.standard_normal(D).astype(np.float32)).astype(bfloat16)

    wq = rng.integers(-wlim, wlim + 1, size=(QD, D), dtype=np.int8)
    bq = rng.integers(-blim, blim,     size=QD,      dtype=np.int32)
    wo = rng.integers(-wlim, wlim + 1, size=(D, QD), dtype=np.int8)
    bo = rng.integers(-blim, blim,     size=D,       dtype=np.int32)

    # RoPE cos/sin: doubled-halves over a random angle per pair (Llama-3).
    half = HEAD_D // 2
    ang = rng.uniform(0, 2 * np.pi, size=half).astype(np.float32)
    cos_half = np.cos(ang).astype(bfloat16)
    sin_half = np.sin(ang).astype(bfloat16)
    cos = np.concatenate([cos_half, cos_half])
    sin = np.concatenate([sin_half, sin_half])

    # Random pre-filled KV cache (T x KVD bytes each).
    kcache = rng.integers(-32, 33, size=T * KVD, dtype=np.int8)
    vcache = rng.integers(-32, 33, size=T * KVD, dtype=np.int8)

    # --- Numpy reference ---
    h1 = numpy_rmsnorm_int8(x_in, gamma_in, ACT_SCALE, INV_ACT_SCALE)
    qf = numpy_gemm_srs(h1, wq, bq, D, QD, RIGHT_SHIFT)
    qr = numpy_rope(qf, cos, sin, N_HEADS, HEAD_D, ACT_SCALE)
    lut = exp_lut(EXP_QUANT_SCALE).astype(np.float32)
    af = numpy_attention(qr, kcache, vcache, HEAD_D, T,
                         Q_SCALE, K_SCALE, V_SCALE, INV_OUT_SCALE_AT, lut)
    op = numpy_gemm_srs(af, wo, bo, QD, D, RIGHT_SHIFT)
    expected = i8_add_wrap(op, x_in)

    # --- Pack weights blob ---
    wblob = np.zeros(WEIGHTS_BYTES, dtype=np.int8)
    wblob[OFF_GAMMA:OFF_GAMMA + GAMMA_BYTES] = gamma_in.view(np.int8)
    pack_tiled_weights(wblob, OFF_WQ, wq, bq, N_TILES_Q, N_TILE, D)
    cs_packed = np.concatenate([cos, sin])
    wblob[OFF_CS:OFF_CS + CS_BYTES] = cs_packed.view(np.int8)
    pack_tiled_weights(wblob, OFF_WO, wo, bo, N_TILES_O, N_TILE, QD)

    # --- Pack KV cache blob ---
    kvblob = np.zeros(KV_BYTES, dtype=np.int8)
    kvblob[OFF_K:OFF_K + KCACHE_BYTES] = kcache
    kvblob[OFF_V:OFF_V + VCACHE_BYTES] = vcache

    # --- NPU dispatch ---
    x_t  = iron.tensor(x_in, dtype=np.int8)
    w_t  = iron.tensor(wblob, dtype=np.int8)
    kv_t = iron.tensor(kvblob, dtype=np.int8)
    o_t  = iron.zeros([D], dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    rc = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [x_t, w_t, kv_t, o_t],
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
    print(f"attn-half NPU vs numpy: D={D} HEAD_DIM={HEAD_D} N_HEADS={N_HEADS} "
          f"T={T}  mismatches={n_diff}/{D}  max|diff|={max_abs}  "
          f"saturated={sat}/{D}")

    if n_diff == 0:
        print("BIT-EXACT PASS  (attn-half: rmsnorm + q + rope + flowkv + o + add)")
        return 0
    if active and max_abs <= 1:
        print(f"PASS (within 1-LSB active-data modeling gap: {n_diff} diffs)")
        return 0
    print("FAIL")
    for i in np.argwhere(diff != 0).flatten()[:8]:
        print(f"  i={i}: NPU={actual[i]}  expected={expected[i]}  x_in={x_in[i]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
