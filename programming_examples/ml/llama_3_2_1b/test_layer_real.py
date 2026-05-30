"""Phase 3: full single-layer real-kernel integration test.

Generates random weights/biases/gammas/cs/KV-cache, packs them per
aie2_layer_real.py's blob layout, runs the kernel on NPU, and bit-
exact compares against a chained numpy reference that uses each
kernel's own numpy reference function. v0 drops k_proj/v_proj/rope_k
(current-token KV-write); cache is supplied wholly from runtime.
"""

from __future__ import annotations

import sys

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_layer_real import (
    D, QD, KVD, HD, HEAD_D, N_HEADS, N_KV, T,
    OFF_GAMMA_IN, OFF_GAMMA_POST,
    OFF_WQ, OFF_WO, OFF_WG, OFF_WU, OFF_WD, OFF_CS,
    GAMMA_BYTES, WQ_BYTES, WO_BYTES, WG_BYTES, WU_BYTES, WD_BYTES, CS_BYTES,
    KCACHE_BYTES, VCACHE_BYTES, TOTAL_W, KV_TOTAL, OFF_K, OFF_V,
    RIGHT_SHIFT, ACT_SCALE, INV_ACT_SCALE,
    UP_SCALE, INV_OUT_SCALE_SI,
    Q_SCALE, K_SCALE, V_SCALE, INV_OUT_SCALE_AT,
)
from test_rmsnorm_int8 import numpy_rmsnorm_int8
from test_gemm_int8_srs_real import banker_srs
from test_rope_int8 import numpy_rope
from test_silu_mul_int8 import numpy_silu_mul
from test_flowkv import numpy_attention, EXP_QUANT_SCALE
from gen_silu_lut import silu_lut
from gen_exp_lut import exp_lut


def round_to_i8(v):
    r = np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5)).astype(np.int32)
    return np.clip(r, -128, 127).astype(np.int8)


def i8_add_wrap(a, b):
    return (a.astype(np.int32) + b.astype(np.int32)).astype(np.int8)


def numpy_gemm(act, weights, bias, K, N, right_shift):
    acc = weights.astype(np.int32) @ act.astype(np.int32)
    sums = acc + bias
    return banker_srs(sums, right_shift).clip(-128, 127).astype(np.int8)


def main():
    p = test_utils.create_default_argparser()
    opts = p.parse_args()

    rng = np.random.default_rng(0)

    # Random inputs (small magnitudes for activations so the chain
    # stays in int8 range after several saturating SRS rounds).
    x_in = rng.integers(-32, 33, size=D, dtype=np.int8)

    # Weights: small to keep accumulators in range.
    def randw(*shape):
        return rng.integers(-32, 33, size=shape, dtype=np.int8)
    def randb(n):
        return rng.integers(-100, 100, size=n, dtype=np.int32)
    def randg(n):
        return (1.0 + 0.1 * rng.standard_normal(n).astype(np.float32)).astype(bfloat16)

    gamma_in   = randg(D)
    gamma_post = randg(D)
    wq = randw(QD, D);  bq = randb(QD)
    wo = randw(D, QD);  bo = randb(D)
    wg = randw(HD, D);  bg = randb(HD)
    wu = randw(HD, D);  bu = randb(HD)
    wd = randw(D, HD);  bd = randb(D)

    # cos/sin (random angles).
    ang = rng.uniform(0, 2 * np.pi, size=HEAD_D // 2).astype(np.float32)
    cos_half = np.cos(ang).astype(bfloat16); sin_half = np.sin(ang).astype(bfloat16)
    cos = np.concatenate([cos_half, cos_half]); sin = np.concatenate([sin_half, sin_half])
    cs_packed = np.concatenate([cos, sin])    # 2*head_dim bf16

    # KV cache.
    kcache = rng.integers(-32, 33, size=T * KVD, dtype=np.int8)
    vcache = rng.integers(-32, 33, size=T * KVD, dtype=np.int8)

    # --- Pack weights blob (matches aie2_layer_real.py layout). ---
    wblob = np.zeros(TOTAL_W, dtype=np.int8)

    def put_bf16(buf, off, arr):
        # bf16 view as 2-byte
        as_bytes = arr.view(np.int8)
        buf[off:off + as_bytes.size] = as_bytes

    def put_packed(buf, off, w_i8, b_i32):
        # weights row-major i8, then bias i32 raw bytes
        buf[off:off + w_i8.size] = w_i8.flatten()
        bias_bytes = b_i32.view(np.int8)
        buf[off + w_i8.size:off + w_i8.size + bias_bytes.size] = bias_bytes.flatten()

    put_bf16(wblob, OFF_GAMMA_IN,   gamma_in)
    put_bf16(wblob, OFF_GAMMA_POST, gamma_post)
    put_packed(wblob, OFF_WQ, wq, bq)
    put_packed(wblob, OFF_WO, wo, bo)
    put_packed(wblob, OFF_WG, wg, bg)
    put_packed(wblob, OFF_WU, wu, bu)
    put_packed(wblob, OFF_WD, wd, bd)
    put_bf16(wblob, OFF_CS, cs_packed)

    # KV cache buffer.
    kvbuf = np.zeros(KV_TOTAL, dtype=np.int8)
    kvbuf[OFF_K:OFF_K + KCACHE_BYTES] = kcache
    kvbuf[OFF_V:OFF_V + VCACHE_BYTES] = vcache

    # --- Numpy reference: chain the per-kernel refs. ---
    # 1. rmsnorm1
    h1 = numpy_rmsnorm_int8(x_in, gamma_in, ACT_SCALE, INV_ACT_SCALE)
    # 2. q_proj
    qf = numpy_gemm(h1, wq, bq, D, QD, RIGHT_SHIFT)
    # 3. rope_q
    qr = numpy_rope(qf, cos, sin, N_HEADS, HEAD_D, ACT_SCALE)
    # 4. flowkv (qk + sv)
    lut_exp = exp_lut(EXP_QUANT_SCALE)
    af = numpy_attention(qr, kcache, vcache, HEAD_D, T,
                         Q_SCALE, K_SCALE, V_SCALE, INV_OUT_SCALE_AT, lut_exp)
    # 5. o_proj
    op = numpy_gemm(af, wo, bo, QD, D, RIGHT_SHIFT)
    # 6. add1
    x1 = i8_add_wrap(op, x_in)
    # 7. rmsnorm2
    h2 = numpy_rmsnorm_int8(x1, gamma_post, ACT_SCALE, INV_ACT_SCALE)
    # 8. gate_proj
    gf = numpy_gemm(h2, wg, bg, D, HD, RIGHT_SHIFT)
    # 9. up_proj
    uf = numpy_gemm(h2, wu, bu, D, HD, RIGHT_SHIFT)
    # 10. silu_mul
    silu_lut_bf = silu_lut(0.05)   # MUST match SILU_GATE_SCALE in Makefile
    sf = numpy_silu_mul(gf, uf, silu_lut_bf, UP_SCALE, INV_OUT_SCALE_SI)
    # 11. down_proj
    df = numpy_gemm(sf, wd, bd, HD, D, RIGHT_SHIFT)
    # 12. add2
    expected = i8_add_wrap(df, x1)

    # --- Run NPU. ---
    x_t  = iron.tensor(x_in.copy(),       dtype=np.int8)
    w_t  = iron.tensor(wblob,             dtype=np.int8)
    kv_t = iron.tensor(kvbuf,             dtype=np.int8)
    o_t  = iron.zeros([D],                dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    rc = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [x_t, w_t, kv_t, o_t],
        {},
        verify=False,
        verbosity=opts.verbosity,
    )
    if rc != 0:
        return rc

    o_t.to("cpu")
    actual = o_t.numpy()

    diff = actual.astype(np.int16) - expected.astype(np.int16)
    n_diff = int((diff != 0).sum())
    max_abs = int(np.abs(diff).max()) if n_diff else 0
    print(f"layer_real NPU vs chained-numpy: D={D}  mismatches={n_diff}/{D}  max|diff|={max_abs}")

    if n_diff == 0:
        print("BIT-EXACT PASS  (full single-layer end-to-end)")
        return 0
    print("FAIL")
    for i in np.argwhere(diff != 0).flatten()[:8]:
        print(f"  i={i}: NPU={actual[i]}  expected={expected[i]}  x_in={x_in[i]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
