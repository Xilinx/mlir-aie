"""Debug: drain per-layer x1 (post-attn, pre-FFN) from chain_dynscale.
Compare to numpy per-layer x1 to localize where chain corrupts.
Build with LLAMA_CHAIN_TRACE_X1=1; runtime gets a 5th arg of size
N_LAYERS*D bytes."""

import os, sys
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_chain_dynscale import (
    D, HD, HEAD_D, N_HEADS, N_KV, QD, KVD, T, N_TILE, N_LAYERS,
    WQ_SLOT, WO_SLOT, WG_SLOT, WU_SLOT, WD_SLOT,
    N_TILES_Q, N_TILES_O, N_TILES_G, N_TILES_U, N_TILES_D,
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
from test_attn_half import compute_sv_fp
from test_ffn_half import pack_perchan_slots, fp32_bytes, numpy_gemm_perchan, i8_add_wrap
from gen_exp_lut import exp_lut
from gen_silu_lut import silu_lut
from gen_llama_data import quant_int8_perchan_absmax
from test_chain_dynscale import numpy_layer_forward, gen_layer, requant


def numpy_per_layer_x1(x_in, layers):
    """Reproduce numpy_layer_forward but capture x1 (post-attn pre-FFN)
    per layer."""
    x1s = []
    x = x_in
    for L in range(N_LAYERS):
        # Replicate the attn-half portion of numpy_layer_forward inline,
        # capturing x1 = i8_add_wrap(op, x).
        layer = layers[L]
        lut_exp = layer["lut_exp"]
        gamma_in = layer["gamma_in"]
        wq_i8 = layer["wq_i8"]; wq_sc = layer["wq_sc"]; bq = layer["bq"]
        wo_i8 = layer["wo_i8"]; wo_sc = layer["wo_sc"]; bo = layer["bo"]
        cos = layer["cos"]; sin = layer["sin"]
        kcache = layer["kcache"]; vcache = layer["vcache"]
        k_scale = layer["k_scale"]; v_scale = layer["v_scale"]
        h1 = numpy_rmsnorm_int8(x, gamma_in, ACT_SCALE, INV_ACT_SCALE)
        fp_q = (wq_i8.astype(np.int32) @ h1.astype(np.int32) + bq).astype(np.float32) \
               * np.float32(ACT_SCALE) * wq_sc.astype(np.float32)
        q_out_scale = float(np.maximum(np.abs(fp_q).max(), 1e-12)) / 127.0
        q_inv_out = float(np.float32(1.0) / np.float32(q_out_scale))
        qf = requant(fp_q, q_inv_out)
        qr = numpy_rope(qf, cos, sin, N_HEADS, HEAD_D, q_out_scale)
        sv_fp = compute_sv_fp(qr, kcache, vcache, HEAD_D, T,
                              q_out_scale, k_scale, v_scale, lut_exp)
        sv_out_scale = float(np.maximum(np.abs(sv_fp).max(), 1e-12)) / 127.0
        sv_inv_out = float(np.float32(1.0) / np.float32(sv_out_scale))
        af = numpy_attention(qr, kcache, vcache, HEAD_D, T,
                             q_out_scale, k_scale, v_scale, sv_inv_out, lut_exp)
        op = numpy_gemm_perchan(af, sv_out_scale, wo_i8, wo_sc, bo, INV_ACT_SCALE)
        x1 = i8_add_wrap(op, x)
        x1s.append(x1.copy())
        # Continue to layer L's output (chain output) for next iteration.
        x_out, _ = numpy_layer_forward(x, layer)
        x = x_out
    return x1s


def main():
    opts = test_utils.create_default_argparser().parse_args()
    seed = int(os.environ.get("LLAMA_CHAIN_SEED", "0"))
    rng = np.random.default_rng(seed)

    lut_exp = exp_lut(EXP_QUANT_SCALE).astype(np.float32)
    lut_silu = silu_lut(SILU_GATE_SCALE)
    layers = []
    for L in range(N_LAYERS):
        layer = gen_layer(rng)
        layer["lut_exp"] = lut_exp
        layer["lut_silu"] = lut_silu
        layers.append(layer)

    x_in = rng.integers(-32, 33, size=D, dtype=np.int8)
    x_in_orig = x_in.copy()

    # Numpy per-layer x1
    ref_x1s = numpy_per_layer_x1(x_in_orig, layers)

    # Pack wblob + kvblob (same as test_chain_dynscale)
    # We need scales per layer; reuse numpy_layer_forward to populate.
    x = x_in_orig.copy()
    for L in range(N_LAYERS):
        x, scales = numpy_layer_forward(x, layers[L])
        layers[L]["scales"] = scales

    wblob = np.zeros(WEIGHTS_BYTES, dtype=np.int8)
    for L in range(N_LAYERS):
        off = OFF_GAMMA_IN + L * GAMMA_BYTES
        wblob[off:off + GAMMA_BYTES] = layers[L]["gamma_in"].view(np.int8)
        off = OFF_GAMMA_POST + L * GAMMA_BYTES
        wblob[off:off + GAMMA_BYTES] = layers[L]["gamma_post"].view(np.int8)
        off = OFF_CS + L * CS_BYTES
        cs_packed = np.concatenate([layers[L]["cos"], layers[L]["sin"]])
        wblob[off:off + CS_BYTES] = cs_packed.view(np.int8)
    def pack_layer_slot(off, w_i8, w_sc, bias, n_tile, n_tiles_per_layer, prefix_bytes):
        packed = pack_perchan_slots(w_i8, w_sc, bias, n_tile, prefix_bytes=prefix_bytes)
        wblob[off:off + packed.size] = packed
    for L in range(N_LAYERS):
        sc = layers[L]["scales"]
        wq_prefix = (fp32_bytes(ACT_SCALE, sc["q_inv_out"], sc["q_out_scale"], 0.0) + b"\x00" * 48)
        pack_layer_slot(OFF_WQ + L * N_TILES_Q * WQ_SLOT, layers[L]["wq_i8"], layers[L]["wq_sc"], layers[L]["bq"], N_TILE, N_TILES_Q, wq_prefix)
        wo_prefix = fp32_bytes(sc["sv_out_scale"], INV_ACT_SCALE) + b"\x00" * 56
        pack_layer_slot(OFF_WO + L * N_TILES_O * WO_SLOT, layers[L]["wo_i8"], layers[L]["wo_sc"], layers[L]["bo"], N_TILE, N_TILES_O, wo_prefix)
        pack_layer_slot(OFF_WG + L * N_TILES_G * WG_SLOT, layers[L]["wg_i8"], layers[L]["wg_sc"], layers[L]["bg"], N_TILE, N_TILES_G, b"")
        wu_prefix = (fp32_bytes(ACT_SCALE, sc["up_inv_out"], sc["silu_up_scale"], sc["silu_inv_out_scale"]) + b"\x00" * 48)
        pack_layer_slot(OFF_WU + L * N_TILES_U * WU_SLOT, layers[L]["wu_i8"], layers[L]["wu_sc"], layers[L]["bu"], N_TILE, N_TILES_U, wu_prefix)
        wd_prefix = fp32_bytes(sc["down_act_scale"], INV_ACT_SCALE) + b"\x00" * 56
        pack_layer_slot(OFF_WD + L * N_TILES_D * WD_SLOT, layers[L]["wd_i8"], layers[L]["wd_sc"], layers[L]["bd"], N_TILE, N_TILES_D, wd_prefix)

    kvblob = np.zeros(KV_BYTES, dtype=np.int8)
    for L in range(N_LAYERS):
        sc = layers[L]["scales"]
        k_off = L * PER_LAYER_KV + OFF_K
        v_off = L * PER_LAYER_KV + OFF_V
        kvblob[k_off:k_off + KV_HEADER] = np.frombuffer(fp32_bytes(sc["k_scale"], 0.0), np.int8)
        kvblob[k_off + KV_HEADER:k_off + KV_HEADER + KCACHE_BYTES] = layers[L]["kcache"]
        kvblob[v_off:v_off + KV_HEADER] = np.frombuffer(fp32_bytes(sc["v_scale"], sc["sv_inv_out_scale"]), np.int8)
        kvblob[v_off + KV_HEADER:v_off + KV_HEADER + VCACHE_BYTES] = layers[L]["vcache"]

    trace_af = os.environ.get("LLAMA_CHAIN_TRACE_AF", "0") == "1"
    trace_x1 = os.environ.get("LLAMA_CHAIN_TRACE_X1", "0") == "1"
    x_t  = iron.tensor(x_in_orig, dtype=np.int8)
    w_t  = iron.tensor(wblob, dtype=np.int8)
    kv_t = iron.tensor(kvblob, dtype=np.int8)
    o_t  = iron.zeros([D], dtype=np.int8)
    tr_t = iron.zeros([N_LAYERS * D], dtype=np.int8)
    ta_t = iron.zeros([N_LAYERS * QD], dtype=np.int8)

    # XRT 5-arg ceiling: can have at most 5 runtime args. Always include
    # xin, w, kv, out (4). The 5th is whichever trace is enabled.
    args = [x_t, w_t, kv_t, o_t]
    if trace_x1: args.append(tr_t)
    if trace_af: args.append(ta_t)
    assert len(args) <= 5, "XRT 5-arg ceiling"

    npu_kernel = test_utils.create_npu_kernel(opts).npu_kernel
    rc = DefaultNPURuntime.run_test(npu_kernel, args, {}, verify=False, verbosity=opts.verbosity)
    if rc != 0:
        return rc
    if trace_x1:
        tr_t.to("cpu")
        trace = tr_t.numpy()
        print(f"PER-LAYER X1 DRAIN (seed={seed}, N_LAYERS={N_LAYERS})")
        for L in range(N_LAYERS):
            actual = trace[L*D:(L+1)*D]
            ref = ref_x1s[L]
            diff = actual.astype(np.int16) - ref.astype(np.int16)
            n_diff = int((diff != 0).sum())
            max_abs = int(np.abs(diff).max()) if n_diff else 0
            mark = "  ✓" if n_diff == 0 else "  ✗"
            print(f"  L={L}: x1 mismatches={n_diff:>5}/{D}  max|diff|={max_abs}{mark}")

    if trace_af:
        ta_t.to("cpu")
        af_trace = ta_t.numpy()
        # Compute numpy AF per-layer (need to track x across layers).
        print(f"PER-LAYER AF DRAIN (seed={seed})")
        x = x_in_orig.copy()
        for L in range(N_LAYERS):
            layer = layers[L]
            sc = layer["scales"]  # use cached scales (same as wblob)
            h1 = numpy_rmsnorm_int8(x, layer["gamma_in"], ACT_SCALE, INV_ACT_SCALE)
            fp_q = (layer["wq_i8"].astype(np.int32) @ h1.astype(np.int32) + layer["bq"]).astype(np.float32) \
                   * np.float32(ACT_SCALE) * layer["wq_sc"].astype(np.float32)
            qf = requant(fp_q, sc["q_inv_out"])
            qr = numpy_rope(qf, layer["cos"], layer["sin"], N_HEADS, HEAD_D, sc["q_out_scale"])
            af_ref = numpy_attention(qr, layer["kcache"], layer["vcache"], HEAD_D, T,
                                     sc["q_out_scale"], sc["k_scale"], sc["v_scale"],
                                     sc["sv_inv_out_scale"], lut_exp)
            af_actual = af_trace[L*QD:(L+1)*QD]
            diff = af_actual.astype(np.int16) - af_ref.astype(np.int16)
            n_diff = int((diff != 0).sum())
            max_abs = int(np.abs(diff).max()) if n_diff else 0
            mark = "  ✓" if n_diff == 0 else "  ✗"
            print(f"  L={L}: af mismatches={n_diff:>4}/{QD}  max|diff|={max_abs}{mark}")
            x, _ = numpy_layer_forward(x, layer)


if __name__ == "__main__":
    sys.exit(main())
