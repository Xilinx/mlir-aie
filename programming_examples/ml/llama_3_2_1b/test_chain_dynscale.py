"""Phase 6c.5b.5: N-layer decode chain at production D=2048/HD=8192
with per-channel weight quant + per-token dynamic activation scales.
Single xclbin handles N seeds (each layer's per-layer scales packed
into the wblob slot prefixes / kvblob headers).
"""

from __future__ import annotations

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_chain_dynscale import (
    D, HD, HEAD_D, N_HEADS, N_KV, QD, KVD, T, N_TILE, N_LAYERS,
    WQ_SLOT, WO_SLOT, WG_SLOT, WU_SLOT, WD_SLOT,
    N_TILES_Q, N_TILES_O, N_TILES_G, N_TILES_U, N_TILES_D,
    TOT_GAMMA, TOT_CS, TOT_WQ, TOT_WO, TOT_WG, TOT_WU, TOT_WD,
    OFF_GAMMA_IN, OFF_WQ, OFF_CS, OFF_WO,
    OFF_GAMMA_POST, OFF_WG, OFF_WU, OFF_WD,
    GAMMA_BYTES, CS_BYTES, KCACHE_BYTES, VCACHE_BYTES,
    KCACHE_PADDED, VCACHE_PADDED, KV_HEADER, PER_LAYER_KV,
    OFF_K, OFF_V, KV_BYTES, WEIGHTS_BYTES,
    ACT_SCALE, INV_ACT_SCALE, SILU_GATE_SCALE, GATE_INV_OUT_SCALE,
)
from test_rmsnorm_int8 import numpy_rmsnorm_int8
from test_rope_int8 import numpy_rope
from test_flowkv import numpy_attention, EXP_QUANT_SCALE
from test_silu_mul_int8 import numpy_silu_mul
from test_attn_half import compute_sv_fp
from test_ffn_half import (
    pack_perchan_slots, fp32_bytes, numpy_gemm_perchan, i8_add_wrap,
)
from gen_exp_lut import exp_lut
from gen_silu_lut import silu_lut
from gen_llama_data import quant_int8_perchan_absmax


def requant(fp, inv):
    s = fp * np.float32(inv)
    r = np.where(s >= 0, np.floor(s + np.float32(0.5)),
                          np.ceil(s - np.float32(0.5)))
    return r.clip(-128, 127).astype(np.int8)


def numpy_layer_forward(x, layer):
    """Single-layer forward with per-stage scale calibration. Returns
    (x_out, layer_scales) where layer_scales is a dict of all dynamic
    scales for this layer.
    """
    lut_exp = layer["lut_exp"]; lut_silu = layer["lut_silu"]
    gamma_in = layer["gamma_in"]; gamma_post = layer["gamma_post"]
    wq_i8 = layer["wq_i8"]; wq_sc = layer["wq_sc"]; bq = layer["bq"]
    wo_i8 = layer["wo_i8"]; wo_sc = layer["wo_sc"]; bo = layer["bo"]
    wg_i8 = layer["wg_i8"]; wg_sc = layer["wg_sc"]; bg = layer["bg"]
    wu_i8 = layer["wu_i8"]; wu_sc = layer["wu_sc"]; bu = layer["bu"]
    wd_i8 = layer["wd_i8"]; wd_sc = layer["wd_sc"]; bd = layer["bd"]
    cos = layer["cos"]; sin = layer["sin"]
    kcache = layer["kcache"]; vcache = layer["vcache"]
    k_scale = layer["k_scale"]; v_scale = layer["v_scale"]

    h1 = numpy_rmsnorm_int8(x, gamma_in, ACT_SCALE, INV_ACT_SCALE)

    fp_q = (wq_i8.astype(np.int32) @ h1.astype(np.int32) + bq).astype(np.float32) \
           * np.float32(ACT_SCALE) * wq_sc.astype(np.float32)
    q_out_scale = float(np.maximum(np.abs(fp_q).max(), 1e-12)) / 127.0
    q_inv_out   = float(np.float32(1.0) / np.float32(q_out_scale))
    qf = requant(fp_q, q_inv_out)

    qr = numpy_rope(qf, cos, sin, N_HEADS, HEAD_D, q_out_scale)

    sv_fp = compute_sv_fp(qr, kcache, vcache, HEAD_D, T,
                          q_out_scale, k_scale, v_scale, lut_exp)
    sv_out_scale     = float(np.maximum(np.abs(sv_fp).max(), 1e-12)) / 127.0
    sv_inv_out_scale = float(np.float32(1.0) / np.float32(sv_out_scale))
    af = numpy_attention(qr, kcache, vcache, HEAD_D, T,
                         q_out_scale, k_scale, v_scale, sv_inv_out_scale, lut_exp)

    op = numpy_gemm_perchan(af, sv_out_scale, wo_i8, wo_sc, bo, INV_ACT_SCALE)
    x1 = i8_add_wrap(op, x)

    h2 = numpy_rmsnorm_int8(x1, gamma_post, ACT_SCALE, INV_ACT_SCALE)

    fp_gate = (wg_i8.astype(np.int32) @ h2.astype(np.int32) + bg).astype(np.float32) \
              * np.float32(ACT_SCALE) * wg_sc.astype(np.float32)
    gf = requant(fp_gate, GATE_INV_OUT_SCALE)

    fp_up = (wu_i8.astype(np.int32) @ h2.astype(np.int32) + bu).astype(np.float32) \
            * np.float32(ACT_SCALE) * wu_sc.astype(np.float32)
    up_out_scale = float(np.maximum(np.abs(fp_up).max(), 1e-12)) / 127.0
    up_inv_out   = float(np.float32(1.0) / np.float32(up_out_scale))
    uf = requant(fp_up, up_inv_out)

    silu_up_scale = up_out_scale
    s_gate_fp = lut_silu[gf.astype(np.int32) + 128].astype(np.float32)
    s_up_fp = uf.astype(np.float32) * np.float32(silu_up_scale)
    sf_fp = s_gate_fp * s_up_fp
    silu_out_scale     = float(np.maximum(np.abs(sf_fp).max(), 1e-12)) / 127.0
    silu_inv_out_scale = float(np.float32(1.0) / np.float32(silu_out_scale))
    sf = numpy_silu_mul(gf, uf, lut_silu, silu_up_scale, silu_inv_out_scale)

    down_act_scale = silu_out_scale
    df = numpy_gemm_perchan(sf, down_act_scale, wd_i8, wd_sc, bd, INV_ACT_SCALE)

    x_out = i8_add_wrap(df, x1)

    scales = dict(
        q_inv_out=q_inv_out, q_out_scale=q_out_scale,
        sv_inv_out_scale=sv_inv_out_scale, sv_out_scale=sv_out_scale,
        up_inv_out=up_inv_out, up_out_scale=up_out_scale,
        silu_up_scale=silu_up_scale, silu_inv_out_scale=silu_inv_out_scale,
        down_act_scale=down_act_scale,
        k_scale=k_scale, v_scale=v_scale,
    )
    return x_out, scales


def gen_layer(rng):
    """Per-layer random weights / cache / gammas / cs."""
    def random_w(out_dim, in_dim):
        base = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.05
        row_scale = rng.uniform(0.1, 1.0, size=out_dim).astype(np.float32)
        return base * row_scale[:, None]

    wq_fp = random_w(QD, D);   bq = np.zeros(QD, np.int32)
    wo_fp = random_w(D,  QD);  bo = np.zeros(D,  np.int32)
    wg_fp = random_w(HD, D);   bg = np.zeros(HD, np.int32)
    wu_fp = random_w(HD, D);   bu = np.zeros(HD, np.int32)
    wd_fp = random_w(D,  HD);  bd = np.zeros(D,  np.int32)
    wq_i8, wq_sc = quant_int8_perchan_absmax(wq_fp)
    wo_i8, wo_sc = quant_int8_perchan_absmax(wo_fp)
    wg_i8, wg_sc = quant_int8_perchan_absmax(wg_fp)
    wu_i8, wu_sc = quant_int8_perchan_absmax(wu_fp)
    wd_i8, wd_sc = quant_int8_perchan_absmax(wd_fp)

    half = HEAD_D // 2
    ang = rng.uniform(0, 2 * np.pi, size=half).astype(np.float32)
    cos_half = np.cos(ang).astype(bfloat16)
    sin_half = np.sin(ang).astype(bfloat16)
    cos = np.concatenate([cos_half, cos_half])
    sin = np.concatenate([sin_half, sin_half])

    kcache = rng.integers(-32, 33, size=T * KVD, dtype=np.int8)
    vcache = rng.integers(-32, 33, size=T * KVD, dtype=np.int8)

    gamma_in   = (1.0 + 0.1 * rng.standard_normal(D).astype(np.float32)).astype(bfloat16)
    gamma_post = (1.0 + 0.1 * rng.standard_normal(D).astype(np.float32)).astype(bfloat16)

    return dict(
        wq_i8=wq_i8, wq_sc=wq_sc, bq=bq,
        wo_i8=wo_i8, wo_sc=wo_sc, bo=bo,
        wg_i8=wg_i8, wg_sc=wg_sc, bg=bg,
        wu_i8=wu_i8, wu_sc=wu_sc, bu=bu,
        wd_i8=wd_i8, wd_sc=wd_sc, bd=bd,
        cos=cos, sin=sin,
        kcache=kcache, vcache=vcache,
        gamma_in=gamma_in, gamma_post=gamma_post,
        k_scale=0.05, v_scale=0.05,
    )


def run_one_seed(seed: int, opts, npu_kernel) -> int:
    rng = np.random.default_rng(seed)

    # Generate per-layer data + LUTs (LUTs shared across layers).
    lut_exp  = exp_lut(EXP_QUANT_SCALE).astype(np.float32)
    lut_silu = silu_lut(SILU_GATE_SCALE)
    layers = []
    for L in range(N_LAYERS):
        layer = gen_layer(rng)
        layer["lut_exp"] = lut_exp
        layer["lut_silu"] = lut_silu
        layers.append(layer)

    # Initial activation (residual seed).
    x = rng.integers(-32, 33, size=D, dtype=np.int8)
    x_in_original = x.copy()

    # Numpy forward through N_LAYERS.
    for L in range(N_LAYERS):
        x, scales = numpy_layer_forward(x, layers[L])
        layers[L]["scales"] = scales
    expected = x

    # --- Pack wblob ---
    wblob = np.zeros(WEIGHTS_BYTES, dtype=np.int8)

    # gam_in / gam_post: N_LAYERS slots of GAMMA_BYTES each, contiguous.
    for L in range(N_LAYERS):
        off = OFF_GAMMA_IN + L * GAMMA_BYTES
        wblob[off:off + GAMMA_BYTES] = layers[L]["gamma_in"].view(np.int8)
        off = OFF_GAMMA_POST + L * GAMMA_BYTES
        wblob[off:off + GAMMA_BYTES] = layers[L]["gamma_post"].view(np.int8)
        off = OFF_CS + L * CS_BYTES
        cs_packed = np.concatenate([layers[L]["cos"], layers[L]["sin"]])
        wblob[off:off + CS_BYTES] = cs_packed.view(np.int8)

    # WQ / WO / WG / WU / WD: each layer packs its perchan slots; the
    # per-layer packed blobs are concatenated across layers.
    def pack_layer_slot(off, w_i8, w_sc, bias, n_tile, n_tiles_per_layer,
                        prefix_bytes):
        slot_bytes = (len(prefix_bytes) + n_tile *
                      (w_i8.shape[1]) + n_tile * 4 + n_tile * 4)
        per_layer_total = n_tiles_per_layer * slot_bytes
        packed = pack_perchan_slots(w_i8, w_sc, bias, n_tile,
                                    prefix_bytes=prefix_bytes)
        assert packed.size == per_layer_total
        wblob[off:off + per_layer_total] = packed

    for L in range(N_LAYERS):
        sc = layers[L]["scales"]
        # WQ
        wq_prefix = (fp32_bytes(ACT_SCALE, sc["q_inv_out"],
                                sc["q_out_scale"], 0.0)
                     + b"\x00" * 48)
        pack_layer_slot(OFF_WQ + L * N_TILES_Q * WQ_SLOT,
                        layers[L]["wq_i8"], layers[L]["wq_sc"], layers[L]["bq"],
                        N_TILE, N_TILES_Q, wq_prefix)
        # WO
        wo_prefix = fp32_bytes(sc["sv_out_scale"], INV_ACT_SCALE) + b"\x00" * 56
        pack_layer_slot(OFF_WO + L * N_TILES_O * WO_SLOT,
                        layers[L]["wo_i8"], layers[L]["wo_sc"], layers[L]["bo"],
                        N_TILE, N_TILES_O, wo_prefix)
        # WG (no prefix; closure-baked)
        pack_layer_slot(OFF_WG + L * N_TILES_G * WG_SLOT,
                        layers[L]["wg_i8"], layers[L]["wg_sc"], layers[L]["bg"],
                        N_TILE, N_TILES_G, b"")
        # WU
        wu_prefix = (fp32_bytes(ACT_SCALE, sc["up_inv_out"],
                                sc["silu_up_scale"], sc["silu_inv_out_scale"])
                     + b"\x00" * 48)
        pack_layer_slot(OFF_WU + L * N_TILES_U * WU_SLOT,
                        layers[L]["wu_i8"], layers[L]["wu_sc"], layers[L]["bu"],
                        N_TILE, N_TILES_U, wu_prefix)
        # WD
        wd_prefix = fp32_bytes(sc["down_act_scale"], INV_ACT_SCALE) + b"\x00" * 56
        pack_layer_slot(OFF_WD + L * N_TILES_D * WD_SLOT,
                        layers[L]["wd_i8"], layers[L]["wd_sc"], layers[L]["bd"],
                        N_TILE, N_TILES_D, wd_prefix)

    # --- Pack kvblob ---
    kvblob = np.zeros(KV_BYTES, dtype=np.int8)
    for L in range(N_LAYERS):
        sc = layers[L]["scales"]
        k_off = L * PER_LAYER_KV + OFF_K
        v_off = L * PER_LAYER_KV + OFF_V
        k_header = fp32_bytes(sc["k_scale"], 0.0)
        kvblob[k_off:k_off + KV_HEADER] = np.frombuffer(k_header, dtype=np.int8)
        kvblob[k_off + KV_HEADER:k_off + KV_HEADER + KCACHE_BYTES] = layers[L]["kcache"]
        v_header = fp32_bytes(sc["v_scale"], sc["sv_inv_out_scale"])
        kvblob[v_off:v_off + KV_HEADER] = np.frombuffer(v_header, dtype=np.int8)
        kvblob[v_off + KV_HEADER:v_off + KV_HEADER + VCACHE_BYTES] = layers[L]["vcache"]

    # --- NPU dispatch ---
    x_t  = iron.tensor(x_in_original, dtype=np.int8)
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
          f"sat={sat}/{D}")
    if n_diff != 0:
        for i in np.argwhere(diff != 0).flatten()[:8]:
            print(f"    i={i}: NPU={actual[i]}  expected={expected[i]}  x={x_in_original[i]}")
    return n_diff


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated seed list (default: LLAMA_CHAIN_SEED env or 0,1,7,42)")
    opts = p.parse_args()

    npu_kernel = test_utils.create_npu_kernel(opts).npu_kernel

    if opts.seeds is not None:
        seeds = [int(s) for s in opts.seeds.split(",") if s.strip()]
    elif "LLAMA_CHAIN_SEED" in os.environ:
        seeds = [int(os.environ["LLAMA_CHAIN_SEED"])]
    else:
        seeds = [0, 1, 7, 42]

    print(f"chain-dynscale (N_LAYERS={N_LAYERS}, D={D}, HD={HD}, T={T})  seeds={seeds}")
    print("  (same xclbin for every seed -- no rebuild)")

    n_fail = 0
    for s in seeds:
        rc = run_one_seed(s, opts, npu_kernel)
        if rc != 0:
            n_fail += 1
    if n_fail == 0:
        print(f"BIT-EXACT PASS x {len(seeds)}  (N={N_LAYERS} dynscale chain)")
        return 0
    print(f"FAIL: {n_fail}/{len(seeds)} seeds had mismatches")
    return 1


if __name__ == "__main__":
    sys.exit(main())
