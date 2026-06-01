"""Phase 6c.5b.4: full single-layer Llama 3.2 1B decoder at production
D=2048, single-head, T=128, with per-channel weight quant + per-token
dynamic activation scales. Single xclbin, multi-seed bit-exact.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_layer_d2048 import (
    D, HD, HEAD_D, N_HEADS, N_KV, QD, KVD, T, N_TILE,
    WQ_SLOT, WO_SLOT, WG_SLOT, WU_SLOT, WD_SLOT,
    N_TILES_Q, N_TILES_O, N_TILES_G, N_TILES_U, N_TILES_D,
    OFF_GAMMA_IN, OFF_WQ, OFF_CS, OFF_WO,
    OFF_GAMMA_POST, OFF_WG, OFF_WU, OFF_WD,
    GAMMA_BYTES, CS_BYTES, KCACHE_BYTES, VCACHE_BYTES,
    KCACHE_PADDED, VCACHE_PADDED, KV_HEADER,
    OFF_K, OFF_V, KV_BYTES, WEIGHTS_BYTES,
    ACT_SCALE, INV_ACT_SCALE, SILU_GATE_SCALE, GATE_INV_OUT_SCALE,
)
from test_rmsnorm_int8 import numpy_rmsnorm_int8
from test_rope_int8 import numpy_rope, numpy_rope_dyn
from test_flowkv import numpy_attention, EXP_QUANT_SCALE
from test_silu_mul_int8 import numpy_silu_mul
from test_attn_half import compute_sv_fp
from test_ffn_half import (
    pack_perchan_slots, fp32_bytes, numpy_gemm_perchan, i8_add_wrap,
)
from gen_exp_lut import exp_lut
from gen_silu_lut import silu_lut
from gen_llama_data import quant_int8_perchan_absmax


def run_one_seed(seed: int, opts, npu_kernel) -> int:
    rng = np.random.default_rng(seed)

    # --- Inputs / parameters ---
    x_in       = rng.integers(-32, 33, size=D, dtype=np.int8)
    gamma_in   = (1.0 + 0.1 * rng.standard_normal(D).astype(np.float32)).astype(bfloat16)
    gamma_post = (1.0 + 0.1 * rng.standard_normal(D).astype(np.float32)).astype(bfloat16)

    def random_w(out_dim, in_dim):
        base = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.05
        row_scale = rng.uniform(0.1, 1.0, size=out_dim).astype(np.float32)
        return base * row_scale[:, None]

    wq_fp = random_w(QD, D);   bq = np.zeros(QD, dtype=np.int32)
    wo_fp = random_w(D,  QD);  bo = np.zeros(D,  dtype=np.int32)
    wg_fp = random_w(HD, D);   bg = np.zeros(HD, dtype=np.int32)
    wu_fp = random_w(HD, D);   bu = np.zeros(HD, dtype=np.int32)
    wd_fp = random_w(D,  HD);  bd = np.zeros(D,  dtype=np.int32)

    wq_i8, wq_sc = quant_int8_perchan_absmax(wq_fp)
    wo_i8, wo_sc = quant_int8_perchan_absmax(wo_fp)
    wg_i8, wg_sc = quant_int8_perchan_absmax(wg_fp)
    wu_i8, wu_sc = quant_int8_perchan_absmax(wu_fp)
    wd_i8, wd_sc = quant_int8_perchan_absmax(wd_fp)

    # RoPE
    half = HEAD_D // 2
    ang = rng.uniform(0, 2 * np.pi, size=half).astype(np.float32)
    cos_half = np.cos(ang).astype(bfloat16)
    sin_half = np.sin(ang).astype(bfloat16)
    cos = np.concatenate([cos_half, cos_half])
    sin = np.concatenate([sin_half, sin_half])

    kcache = rng.integers(-32, 33, size=T * KVD, dtype=np.int8)
    vcache = rng.integers(-32, 33, size=T * KVD, dtype=np.int8)

    k_scale = 0.05
    v_scale = 0.05

    # --- Numpy reference: attention + FFN with per-stage scale calibration ---
    lut_exp = exp_lut(EXP_QUANT_SCALE).astype(np.float32)
    lut_silu = silu_lut(SILU_GATE_SCALE)

    # 1. rmsnorm1
    h1 = numpy_rmsnorm_int8(x_in, gamma_in, ACT_SCALE, INV_ACT_SCALE)

    # 2. q_proj (dynamic out scale)
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

    # 3. rope
    qr = numpy_rope_dyn(qf, cos, sin, N_HEADS, HEAD_D)

    # 4. flowkv (calibrate sv output scale)
    sv_fp = compute_sv_fp(qr, kcache, vcache, HEAD_D, T,
                          q_out_scale, k_scale, v_scale, lut_exp)
    sv_out_scale     = float(np.maximum(np.abs(sv_fp).max(), 1e-12)) / 127.0
    sv_inv_out_scale = float(np.float32(1.0) / np.float32(sv_out_scale))
    af = numpy_attention(qr, kcache, vcache, HEAD_D, T,
                         q_out_scale, k_scale, v_scale, sv_inv_out_scale,
                         lut_exp)

    # 5. o_proj (dynamic act scale = sv_out_scale; out -> baked INV_ACT_SCALE)
    op = numpy_gemm_perchan(af, sv_out_scale, wo_i8, wo_sc, bo, INV_ACT_SCALE)

    # 6. add1
    x1 = i8_add_wrap(op, x_in)

    # 7. rmsnorm2
    h2 = numpy_rmsnorm_int8(x1, gamma_post, ACT_SCALE, INV_ACT_SCALE)

    # 8. gate (closure-baked GATE_INV_OUT_SCALE = 1/SILU_GATE_SCALE)
    fp_gate = (wg_i8.astype(np.int32) @ h2.astype(np.int32) + bg).astype(np.float32) \
              * np.float32(ACT_SCALE) * wg_sc.astype(np.float32)
    gf = requant(fp_gate, GATE_INV_OUT_SCALE)

    # 9. up (dynamic out scale)
    fp_up = (wu_i8.astype(np.int32) @ h2.astype(np.int32) + bu).astype(np.float32) \
            * np.float32(ACT_SCALE) * wu_sc.astype(np.float32)
    up_out_scale = float(np.maximum(np.abs(fp_up).max(), 1e-12)) / 127.0
    up_inv_out   = float(np.float32(1.0) / np.float32(up_out_scale))
    uf = requant(fp_up, up_inv_out)

    # 10. silu_mul (dynamic up_scale + silu_inv_out)
    silu_up_scale = up_out_scale
    s_gate_fp = lut_silu[gf.astype(np.int32) + 128].astype(np.float32)
    s_up_fp = uf.astype(np.float32) * np.float32(silu_up_scale)
    sf_fp = s_gate_fp * s_up_fp
    silu_out_scale     = float(np.maximum(np.abs(sf_fp).max(), 1e-12)) / 127.0
    silu_inv_out_scale = float(np.float32(1.0) / np.float32(silu_out_scale))
    sf = numpy_silu_mul(gf, uf, lut_silu, silu_up_scale, silu_inv_out_scale)

    # 11. down (dynamic act scale = silu_out_scale; out -> baked INV_ACT_SCALE)
    down_act_scale = silu_out_scale
    df = numpy_gemm_perchan(sf, down_act_scale, wd_i8, wd_sc, bd, INV_ACT_SCALE)

    # 12. add2
    expected = i8_add_wrap(df, x1)

    # --- Pack weights blob ---
    wblob = np.zeros(WEIGHTS_BYTES, dtype=np.int8)
    wblob[OFF_GAMMA_IN:OFF_GAMMA_IN + GAMMA_BYTES] = gamma_in.view(np.int8)
    wblob[OFF_GAMMA_POST:OFF_GAMMA_POST + GAMMA_BYTES] = gamma_post.view(np.int8)
    cs_packed = np.concatenate([cos, sin])
    wblob[OFF_CS:OFF_CS + CS_BYTES] = cs_packed.view(np.int8)

    # WQ: 64 B prefix (ACT_SCALE, q_inv_out, q_out_scale, spare) + 48 B pad.
    wq_prefix = (fp32_bytes(ACT_SCALE, q_inv_out, q_out_scale, 0.0)
                 + b"\x00" * 48)
    wq_packed = pack_perchan_slots(wq_i8, wq_sc, bq, N_TILE, prefix_bytes=wq_prefix)
    assert wq_packed.size == N_TILES_Q * WQ_SLOT
    wblob[OFF_WQ:OFF_WQ + wq_packed.size] = wq_packed

    # WO: 64 B prefix (sv_out_scale, INV_ACT_SCALE) + 56 B pad.
    wo_prefix = fp32_bytes(sv_out_scale, INV_ACT_SCALE) + b"\x00" * 56
    wo_packed = pack_perchan_slots(wo_i8, wo_sc, bo, N_TILE, prefix_bytes=wo_prefix)
    assert wo_packed.size == N_TILES_O * WO_SLOT
    wblob[OFF_WO:OFF_WO + wo_packed.size] = wo_packed

    # WG: no prefix (closure-baked).
    wg_packed = pack_perchan_slots(wg_i8, wg_sc, bg, N_TILE)
    assert wg_packed.size == N_TILES_G * WG_SLOT
    wblob[OFF_WG:OFF_WG + wg_packed.size] = wg_packed

    # WU: 64 B prefix (ACT_SCALE, up_inv_out, silu_up_scale, silu_inv_out) + 48 B pad.
    wu_prefix = (fp32_bytes(ACT_SCALE, up_inv_out, silu_up_scale, silu_inv_out_scale)
                 + b"\x00" * 48)
    wu_packed = pack_perchan_slots(wu_i8, wu_sc, bu, N_TILE, prefix_bytes=wu_prefix)
    assert wu_packed.size == N_TILES_U * WU_SLOT
    wblob[OFF_WU:OFF_WU + wu_packed.size] = wu_packed

    # WD: 64 B prefix (down_act_scale, INV_ACT_SCALE) + 56 B pad.
    wd_prefix = fp32_bytes(down_act_scale, INV_ACT_SCALE) + b"\x00" * 56
    wd_packed = pack_perchan_slots(wd_i8, wd_sc, bd, N_TILE, prefix_bytes=wd_prefix)
    assert wd_packed.size == N_TILES_D * WD_SLOT
    wblob[OFF_WD:OFF_WD + wd_packed.size] = wd_packed

    # --- Pack KV cache blob ---
    kvblob = np.zeros(KV_BYTES, dtype=np.int8)
    k_header = fp32_bytes(k_scale, 0.0)
    kvblob[OFF_K:OFF_K + KV_HEADER] = np.frombuffer(k_header, dtype=np.int8)
    kvblob[OFF_K + KV_HEADER:OFF_K + KV_HEADER + KCACHE_BYTES] = kcache
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
          f"sat={sat}/{D}  q_inv={q_inv_out:.3f}  sv_inv={sv_inv_out_scale:.3f}  "
          f"up_inv={up_inv_out:.3f}  silu_inv={silu_inv_out_scale:.3f}  "
          f"down_act={down_act_scale:.3f}")
    if n_diff != 0:
        for i in np.argwhere(diff != 0).flatten()[:8]:
            print(f"    i={i}: NPU={actual[i]}  expected={expected[i]}  x={x_in[i]}")
    return n_diff


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated seed list (default: LLAMA_LAYER_SEED env or 0,1,7,42)")
    opts = p.parse_args()

    npu_kernel = test_utils.create_npu_kernel(opts).npu_kernel

    if opts.seeds is not None:
        seeds = [int(s) for s in opts.seeds.split(",") if s.strip()]
    elif "LLAMA_LAYER_SEED" in os.environ:
        seeds = [int(os.environ["LLAMA_LAYER_SEED"])]
    else:
        seeds = [0, 1, 7, 42]

    print(f"layer-d2048 (perchan + dynamic-scale runtime) D={D} HD={HD} "
          f"HEAD_D={HEAD_D} T={T}  SILU_GATE_SCALE={SILU_GATE_SCALE}  seeds={seeds}")
    print("  (same xclbin for every seed -- no rebuild)")

    n_fail = 0
    for s in seeds:
        rc = run_one_seed(s, opts, npu_kernel)
        if rc != 0:
            n_fail += 1
    if n_fail == 0:
        print(f"BIT-EXACT PASS x {len(seeds)}  (perchan + dynamic-scale single-layer D=2048)")
        return 0
    print(f"FAIL: {n_fail}/{len(seeds)} seeds had mismatches")
    return 1


if __name__ == "__main__":
    sys.exit(main())
