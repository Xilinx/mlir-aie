"""Phase 7b: bit-exact test for the N-layer multi-head GQA chain xclbin.

Per seed:
  1. Generate N_LAYERS independent multi-head fixtures.
  2. Run numpy_layer_mh_forward through all layers to compute golden y +
     per-layer dynamic scales (q_out, sv_out, sv_inv, o_act, etc.).
  3. Pack wblob (per-fifo contiguous across layers) + kvblob (per-layer
     8-KV-head contiguous block).
  4. Dispatch xclbin, compare bit-exact.

Run:
  make chain_mh CHAIN_MH_N=2
  python test_chain_mh.py -x build/final_chain_mh_N2_T128.xclbin \\
                          -i build/insts_chain_mh_N2_T128.bin \\
                          -k MLIR_AIE --seeds 0,1,7,42
"""

from __future__ import annotations

import os
import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_chain_dynscale_mh import (
    D, HD, HEAD_D, N_HEADS_Q, N_HEADS_KV, REP, QD, T, N_TILE, N_LAYERS,
    WQ_SLOT, WO_SLOT, WG_SLOT, WU_SLOT, WD_SLOT, WQ_PREFIX,
    N_TILES_Q, N_TILES_O, N_TILES_G, N_TILES_U, N_TILES_D,
    OFF_GAMMA_IN, OFF_WQ, OFF_CS, OFF_WO, OFF_AF_SCALES,
    OFF_GAMMA_POST, OFF_WG, OFF_WU, OFF_WD,
    GAMMA_BYTES, CS_BYTES, AF_SCALES_BYTES,
    KCACHE_BYTES, VCACHE_BYTES, KCACHE_PADDED, VCACHE_PADDED, KV_HEADER,
    PER_KV_HEAD_BYTES, PER_LAYER_KV, KV_BYTES, WEIGHTS_BYTES,
    ACT_SCALE, INV_ACT_SCALE, SILU_GATE_SCALE,
)
from numpy_layer_mh import gen_layer_mh, numpy_layer_mh_forward
from test_layer_mh import wq_mh_prefix, af_scales_bytes
from test_ffn_half import pack_perchan_slots, fp32_bytes
from test_flowkv import EXP_QUANT_SCALE
from gen_exp_lut import exp_lut
from gen_silu_lut import silu_lut


def pack_blobs(layers):
    wblob = np.zeros(WEIGHTS_BYTES, dtype=np.int8)
    for L in range(N_LAYERS):
        layer = layers[L]
        sc = layer["scales"]

        # gamma_in (per-fifo contiguous: layer L at offset L * GAMMA_BYTES)
        off = OFF_GAMMA_IN + L * GAMMA_BYTES
        wblob[off:off + GAMMA_BYTES] = layer["gamma_in"].view(np.int8)

        # wq-mh: each layer is N_TILES_Q * WQ_SLOT bytes
        off = OFF_WQ + L * N_TILES_Q * WQ_SLOT
        wq_pre = wq_mh_prefix(ACT_SCALE, sc["q_inv_outs"],
                              sc["q_out_scales"], sc["sv_inv_out_scales"])
        wq_packed = pack_perchan_slots(layer["wq_i8"], layer["wq_sc"], layer["bq"],
                                       N_TILE, prefix_bytes=wq_pre)
        assert wq_packed.size == N_TILES_Q * WQ_SLOT
        wblob[off:off + wq_packed.size] = wq_packed

        # cs
        off = OFF_CS + L * CS_BYTES
        cs_packed = np.concatenate([layer["cos"], layer["sin"]])
        wblob[off:off + CS_BYTES] = cs_packed.view(np.int8)

        # wo-mh
        off = OFF_WO + L * N_TILES_O * WO_SLOT
        wo_pre = fp32_bytes(float(sc["o_act_scale"]), INV_ACT_SCALE) + b"\x00" * 56
        wo_packed = pack_perchan_slots(layer["wo_i8"], layer["wo_sc"], layer["bo"],
                                       N_TILE, prefix_bytes=wo_pre)
        assert wo_packed.size == N_TILES_O * WO_SLOT
        wblob[off:off + wo_packed.size] = wo_packed

        # af_scales
        off = OFF_AF_SCALES + L * AF_SCALES_BYTES
        af_sc = af_scales_bytes(sc["sv_out_scales"], float(sc["o_inv_act_scale"]))
        wblob[off:off + AF_SCALES_BYTES] = np.frombuffer(af_sc, dtype=np.int8)

        # gamma_post
        off = OFF_GAMMA_POST + L * GAMMA_BYTES
        wblob[off:off + GAMMA_BYTES] = layer["gamma_post"].view(np.int8)

        # wg
        off = OFF_WG + L * N_TILES_G * WG_SLOT
        wg_packed = pack_perchan_slots(layer["wg_i8"], layer["wg_sc"], layer["bg"],
                                       N_TILE, prefix_bytes=b"")
        assert wg_packed.size == N_TILES_G * WG_SLOT
        wblob[off:off + wg_packed.size] = wg_packed

        # wu
        off = OFF_WU + L * N_TILES_U * WU_SLOT
        wu_pre = (fp32_bytes(ACT_SCALE, float(sc["up_inv_out"]),
                             float(sc["silu_up_scale"]),
                             float(sc["silu_inv_out_scale"]))
                  + b"\x00" * 48)
        wu_packed = pack_perchan_slots(layer["wu_i8"], layer["wu_sc"], layer["bu"],
                                       N_TILE, prefix_bytes=wu_pre)
        assert wu_packed.size == N_TILES_U * WU_SLOT
        wblob[off:off + wu_packed.size] = wu_packed

        # wd
        off = OFF_WD + L * N_TILES_D * WD_SLOT
        wd_pre = fp32_bytes(float(sc["down_act_scale"]), INV_ACT_SCALE) + b"\x00" * 56
        wd_packed = pack_perchan_slots(layer["wd_i8"], layer["wd_sc"], layer["bd"],
                                       N_TILE, prefix_bytes=wd_pre)
        assert wd_packed.size == N_TILES_D * WD_SLOT
        wblob[off:off + wd_packed.size] = wd_packed

    # kvblob: per-layer contiguous block of 8 KV heads.
    kvblob = np.zeros(KV_BYTES, dtype=np.int8)
    for L in range(N_LAYERS):
        layer = layers[L]
        layer_base = L * PER_LAYER_KV
        for h in range(N_HEADS_KV):
            k_off = layer_base + h * PER_KV_HEAD_BYTES
            v_off = k_off + KCACHE_PADDED
            kvblob[k_off:k_off + 4] = np.frombuffer(
                fp32_bytes(float(layer["k_scales"][h])), dtype=np.int8)
            kvblob[k_off + KV_HEADER:k_off + KV_HEADER + KCACHE_BYTES] = \
                layer["kcaches"][h]
            kvblob[v_off:v_off + 4] = np.frombuffer(
                fp32_bytes(float(layer["v_scales"][h])), dtype=np.int8)
            kvblob[v_off + KV_HEADER:v_off + KV_HEADER + VCACHE_BYTES] = \
                layer["vcaches"][h]
    return wblob, kvblob


def run_one_seed(seed: int, opts, lut_exp, lut_silu, npu_kernel) -> int:
    rng = np.random.default_rng(seed)

    # x is N-independent: generate before per-layer fixtures.
    x = rng.integers(-32, 33, size=D, dtype=np.int8)

    layers = []
    for L in range(N_LAYERS):
        layer = gen_layer_mh(rng)
        layer["lut_exp"]  = lut_exp
        layer["lut_silu"] = lut_silu
        layers.append(layer)

    # Numpy forward through all layers, capturing per-layer dynamic scales.
    x_cur = x.copy()
    for L in range(N_LAYERS):
        x_cur, scales = numpy_layer_mh_forward(x_cur, layers[L])
        layers[L]["scales"] = scales
    y_ref = x_cur

    wblob, kvblob = pack_blobs(layers)

    x_t  = iron.tensor(x,     dtype=np.int8)
    w_t  = iron.tensor(wblob, dtype=np.int8)
    kv_t = iron.tensor(kvblob, dtype=np.int8)
    o_t  = iron.zeros([D], dtype=np.int8)
    rc = DefaultNPURuntime.run_test(
        npu_kernel, [x_t, w_t, kv_t, o_t],
        {}, verify=False, verbosity=opts.verbosity,
    )
    if rc != 0:
        print(f"seed {seed}: NPU dispatch returned {rc}", file=sys.stderr)
        return rc
    o_t.to("cpu")
    y_dev = o_t.numpy()

    diff = y_dev.astype(np.int32) - y_ref.astype(np.int32)
    n_mismatch = int((diff != 0).sum())
    if n_mismatch == 0:
        sat = int((y_ref == 127).sum() + (y_ref == -128).sum())
        print(f"seed {seed}: BIT-EXACT  (sat={sat}/{D})")
        return 0
    max_abs = int(np.abs(diff).max())
    idx = np.where(diff != 0)[0][:10]
    print(f"seed {seed}: FAIL  mismatch={n_mismatch}/{D}  max|d|={max_abs}")
    for i in idx:
        print(f"  i={i}: dev={int(y_dev[i])} ref={int(y_ref[i])}")
    return 1


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--seeds", type=str, default="0,1,7,42",
                   help="comma-separated seed list")
    opts = p.parse_args()
    seeds = [int(s) for s in opts.seeds.split(",")]

    npu_kernel = test_utils.create_npu_kernel(opts).npu_kernel
    lut_exp  = exp_lut(EXP_QUANT_SCALE).astype(np.float32)
    lut_silu = silu_lut(SILU_GATE_SCALE)

    print(f"chain_mh N_LAYERS={N_LAYERS}  T={T}  seeds={seeds}")
    fails = 0
    for s in seeds:
        fails += (run_one_seed(s, opts, lut_exp, lut_silu, npu_kernel) != 0)
    print(f"\nchain_mh: {len(seeds) - fails}/{len(seeds)} seeds BIT-EXACT")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
