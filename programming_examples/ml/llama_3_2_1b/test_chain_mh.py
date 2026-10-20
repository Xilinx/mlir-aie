#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Phase 7b: bit-exact test for the N-layer multi-head GQA chain xclbin.

Per seed:
  1. Generate N_LAYERS independent multi-head fixtures.
  2. Run numpy_layer_mh_forward through all layers to compute golden y +
     per-layer dynamic scales (q_out, sv_out, sv_inv, o_act, etc.).
  3. Pack wblob (per-fifo contiguous across layers) + kvblob (per-layer
     8-KV-head contiguous block).
  4. Dispatch xclbin, compare bit-exact.

Run (compiles the design, or reuses its JIT cache entry):
  python test_chain_mh.py --seeds 0,1,7,42
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

import aie.iron as iron
import numpy as np
from aie.utils import DefaultNPURuntime
from aie2_chain_dynscale_mh import (
    ACT_SCALE,
    AF_SCALES_BYTES,
    CS_BYTES,
    GAMMA_BYTES,
    HEAD_D,
    INV_ACT_SCALE,
    KV_DIM,
    N_HEADS_KV,
    N_TILE,
    N_TILES_D,
    N_TILES_G,
    N_TILES_K,
    N_TILES_O,
    N_TILES_Q,
    N_TILES_U,
    N_TILES_V,
    SILU_GATE_SCALE,
    T_USED_BYTES,
    WD_SLOT,
    WG_SLOT,
    WK_SLOT,
    WO_SLOT,
    WQ_SLOT,
    WU_SLOT,
    WV_SLOT,
    D,
    build,
    derived_sizes,
)
from gen_exp_lut import exp_lut
from gen_silu_lut import silu_lut
from llama_npu import add_artifact_args, design_kernel
from numpy_layer_mh import gen_layer_mh, numpy_layer_mh_forward, requant
from test_flowkv import EXP_QUANT_SCALE
from test_layer_mh import (
    af_scales_bytes,
    fp32_bytes,
    pack_perchan_slots,
    wq_mh_prefix,
)

T = int(os.environ.get("LLAMA_CHAIN_T", "128"))
N_LAYERS = int(os.environ.get("LLAMA_CHAIN_N", "2"))
_sizes = derived_sizes(T=T, N_LAYERS=N_LAYERS)
OFF_GAMMA_IN = _sizes.OFF_GAMMA_IN
OFF_WQ = _sizes.OFF_WQ
OFF_CS = _sizes.OFF_CS
OFF_WO = _sizes.OFF_WO
OFF_AF_SCALES = _sizes.OFF_AF_SCALES
OFF_GAMMA_POST = _sizes.OFF_GAMMA_POST
OFF_WG = _sizes.OFF_WG
OFF_WU = _sizes.OFF_WU
OFF_WD = _sizes.OFF_WD
OFF_WK = _sizes.OFF_WK
OFF_WV = _sizes.OFF_WV
KCACHE_BYTES = _sizes.KCACHE_BYTES
VCACHE_BYTES = _sizes.VCACHE_BYTES
KCACHE_PADDED = _sizes.KCACHE_PADDED
VCACHE_PADDED = _sizes.VCACHE_PADDED
KV_HEADER = _sizes.KV_HEADER
PER_KV_HEAD_BYTES = _sizes.PER_KV_HEAD_BYTES
PER_LAYER_KV = _sizes.PER_LAYER_KV
KV_BYTES = _sizes.KV_BYTES
WEIGHTS_BYTES = _sizes.WEIGHTS_BYTES


def pack_blobs(layers, weights_bytes=WEIGHTS_BYTES):
    """Pack `layers` into the chain's wblob and kvblob.

    Args:
        layers: Per-layer fixtures, as numpy_layer_mh builds them.
        weights_bytes: The wblob size; a PERSIST_GROW chain's is larger, its
            per-position cos/sin block following the packed weights.

    Returns:
        `(wblob, kvblob)`.
    """
    wblob = np.zeros(weights_bytes, dtype=np.int8)
    for L in range(N_LAYERS):
        layer = layers[L]
        sc = layer["scales"]

        # gamma_in (per-fifo contiguous: layer L at offset L * GAMMA_BYTES)
        off = OFF_GAMMA_IN + L * GAMMA_BYTES
        wblob[off : off + GAMMA_BYTES] = layer["gamma_in"].view(np.int8)

        # wq-mh: each layer is N_TILES_Q * WQ_SLOT bytes
        off = OFF_WQ + L * N_TILES_Q * WQ_SLOT
        wq_pre = wq_mh_prefix(
            ACT_SCALE, sc["q_inv_outs"], sc["q_out_scales"], sc["sv_inv_out_scales"]
        )
        wq_packed = pack_perchan_slots(
            layer["wq_i8"], layer["wq_sc"], layer["bq"], N_TILE, prefix_bytes=wq_pre
        )
        assert wq_packed.size == N_TILES_Q * WQ_SLOT
        wblob[off : off + wq_packed.size] = wq_packed

        # cs
        off = OFF_CS + L * CS_BYTES
        cs_packed = np.concatenate([layer["cos"], layer["sin"]])
        wblob[off : off + CS_BYTES] = cs_packed.view(np.int8)

        # wo-mh
        off = OFF_WO + L * N_TILES_O * WO_SLOT
        wo_pre = fp32_bytes(float(sc["o_act_scale"]), INV_ACT_SCALE) + b"\x00" * 56
        wo_packed = pack_perchan_slots(
            layer["wo_i8"], layer["wo_sc"], layer["bo"], N_TILE, prefix_bytes=wo_pre
        )
        assert wo_packed.size == N_TILES_O * WO_SLOT
        wblob[off : off + wo_packed.size] = wo_packed

        # af_scales
        off = OFF_AF_SCALES + L * AF_SCALES_BYTES
        af_sc = af_scales_bytes(sc["sv_out_scales"], float(sc["o_inv_act_scale"]))
        wblob[off : off + AF_SCALES_BYTES] = np.frombuffer(af_sc, dtype=np.int8)

        # gamma_post
        off = OFF_GAMMA_POST + L * GAMMA_BYTES
        wblob[off : off + GAMMA_BYTES] = layer["gamma_post"].view(np.int8)

        # wg
        off = OFF_WG + L * N_TILES_G * WG_SLOT
        wg_packed = pack_perchan_slots(
            layer["wg_i8"], layer["wg_sc"], layer["bg"], N_TILE, prefix_bytes=b""
        )
        assert wg_packed.size == N_TILES_G * WG_SLOT
        wblob[off : off + wg_packed.size] = wg_packed

        # wu
        off = OFF_WU + L * N_TILES_U * WU_SLOT
        wu_pre = (
            fp32_bytes(
                ACT_SCALE,
                float(sc["up_inv_out"]),
                float(sc["silu_up_scale"]),
                float(sc["silu_inv_out_scale"]),
            )
            + b"\x00" * 48
        )
        wu_packed = pack_perchan_slots(
            layer["wu_i8"], layer["wu_sc"], layer["bu"], N_TILE, prefix_bytes=wu_pre
        )
        assert wu_packed.size == N_TILES_U * WU_SLOT
        wblob[off : off + wu_packed.size] = wu_packed

        # wd
        off = OFF_WD + L * N_TILES_D * WD_SLOT
        wd_pre = fp32_bytes(float(sc["down_act_scale"]), INV_ACT_SCALE) + b"\x00" * 56
        wd_packed = pack_perchan_slots(
            layer["wd_i8"], layer["wd_sc"], layer["bd"], N_TILE, prefix_bytes=wd_pre
        )
        assert wd_packed.size == N_TILES_D * WD_SLOT
        wblob[off : off + wd_packed.size] = wd_packed

        # wk / wv (on-chip KV append): 64 B zero prefix (act_scale from h1 tail).
        off = OFF_WK + L * N_TILES_K * WK_SLOT
        wk_packed = pack_perchan_slots(
            layer["wk_i8"],
            layer["wk_sc"],
            np.zeros(KV_DIM, np.int32),
            N_TILE,
            prefix_bytes=b"\x00" * 64,
        )
        assert wk_packed.size == N_TILES_K * WK_SLOT
        wblob[off : off + wk_packed.size] = wk_packed
        off = OFF_WV + L * N_TILES_V * WV_SLOT
        wv_packed = pack_perchan_slots(
            layer["wv_i8"],
            layer["wv_sc"],
            np.zeros(KV_DIM, np.int32),
            N_TILE,
            prefix_bytes=b"\x00" * 64,
        )
        assert wv_packed.size == N_TILES_V * WV_SLOT
        wblob[off : off + wv_packed.size] = wv_packed

    # kvblob: per-layer contiguous block of 8 KV heads. Per-slot KV layout:
    # [T_used i32 | pad | k_slot_scales (T fp32) | k body | v_slot_scales (T
    # fp32) | v body]. Each cached position carries its OWN k/v scale.
    kvblob = np.zeros(KV_BYTES, dtype=np.int8)
    for L in range(N_LAYERS):
        layer = layers[L]
        t_used = layer.get("t_used", T)
        k_slot = layer["k_scales_slot"]  # (N_HEADS_KV, T)
        v_slot = layer["v_scales_slot"]
        layer_base = L * PER_LAYER_KV
        for h in range(N_HEADS_KV):
            tu_off = layer_base + h * PER_KV_HEAD_BYTES
            k_off = tu_off + T_USED_BYTES
            v_off = k_off + KCACHE_PADDED
            kvblob[tu_off : tu_off + 4] = np.frombuffer(
                np.int32(t_used).tobytes(), dtype=np.int8
            )
            kvblob[k_off : k_off + KV_HEADER] = np.frombuffer(
                k_slot[h].astype(np.float32).tobytes(), dtype=np.int8
            )
            kvblob[k_off + KV_HEADER : k_off + KV_HEADER + KCACHE_BYTES] = layer[
                "kcaches"
            ][h]
            kvblob[v_off : v_off + KV_HEADER] = np.frombuffer(
                v_slot[h].astype(np.float32).tobytes(), dtype=np.int8
            )
            kvblob[v_off + KV_HEADER : v_off + KV_HEADER + VCACHE_BYTES] = layer[
                "vcaches"
            ][h]
    return wblob, kvblob


def run_one_seed(seed: int, opts, lut_exp, lut_silu, npu_kernel) -> int:
    rng = np.random.default_rng(seed)

    # x is N-independent: generate before per-layer fixtures. Per-token
    # residual: int8[D] body + fp32 scale (Phase B residual_dyn).
    x_fp = rng.uniform(-1.6, 1.6, size=D).astype(np.float32)
    res_scale = np.float32(np.maximum(np.abs(x_fp).max(), 1e-12) / 127.0)
    x_i8 = requant(x_fp, np.float32(1.0) / res_scale)

    # On-chip KV append: the device computes slot P (k_proj/v_proj/rope_k/
    # quant) for every layer. Pre-fill slots 0..P-1, zero slot P, pass
    # position=P so numpy does the same append.
    P = T // 2
    layers = []
    for L in range(N_LAYERS):
        layer = gen_layer_mh(rng)
        layer["lut_exp"] = lut_exp
        layer["lut_silu"] = lut_silu
        k_slot = rng.uniform(0.02, 0.08, size=(N_HEADS_KV, T)).astype(np.float32)
        v_slot = rng.uniform(0.02, 0.08, size=(N_HEADS_KV, T)).astype(np.float32)
        for h in range(N_HEADS_KV):
            layer["kcaches"][h][P * HEAD_D : (P + 1) * HEAD_D] = 0
            layer["vcaches"][h][P * HEAD_D : (P + 1) * HEAD_D] = 0
            k_slot[h, P] = 0.0
            v_slot[h, P] = 0.0
        layer["k_scales_slot"] = k_slot
        layer["v_scales_slot"] = v_slot
        layer["t_used"] = P + 1
        layers.append(layer)

    # Snapshot pre-append caches (host streams these in); pack from them.
    kin = [[c.copy() for c in lyr["kcaches"]] for lyr in layers]
    vin = [[c.copy() for c in lyr["vcaches"]] for lyr in layers]
    kslot_in = [lyr["k_scales_slot"].copy() for lyr in layers]
    vslot_in = [lyr["v_scales_slot"].copy() for lyr in layers]

    # Numpy forward through all layers; position=P makes numpy do the real
    # k/v_proj + rope_k + append at slot P each layer (in place).
    x_cur = (x_i8.copy(), float(res_scale))
    for L in range(N_LAYERS):
        x_cur, scales = numpy_layer_mh_forward(
            x_cur,
            layers[L],
            position=P,
            residual_dyn=True,
            attn_perslot=True,
            attn_lut=True,
        )
        layers[L]["scales"] = scales
    xo_i8, xo_scale = x_cur
    y_ref = xo_i8
    # numpy post-append caches = golden for the drained device cache slot P.
    kref = [lyr["kcaches"] for lyr in layers]
    vref = [lyr["vcaches"] for lyr in layers]
    ksref = [lyr["k_scales_slot"] for lyr in layers]
    vsref = [lyr["v_scales_slot"] for lyr in layers]

    # Restore pre-append caches so the device starts from slots 0..P-1 + zeroed.
    for L in range(N_LAYERS):
        layers[L]["kcaches"] = kin[L]
        layers[L]["vcaches"] = vin[L]
        layers[L]["k_scales_slot"] = kslot_in[L]
        layers[L]["v_scales_slot"] = vslot_in[L]
    wblob, kvblob = pack_blobs(layers)

    # xin buffer is int8[D+8]: body + 4 B res_scale + 4 B pad.
    xin = np.zeros(D + 8, dtype=np.int8)
    xin[:D] = x_i8
    xin[D : D + 4] = np.frombuffer(np.float32(res_scale).tobytes(), dtype=np.int8)

    x_t = iron.tensor(xin, dtype=np.int8)
    w_t = iron.tensor(wblob, dtype=np.int8)
    kv_t = iron.tensor(kvblob, dtype=np.int8)
    o_t = iron.zeros([D + 8], dtype=np.int8)
    rc = DefaultNPURuntime.run_test(
        npu_kernel,
        [x_t, w_t, kv_t, o_t],
        {},
        verify=False,
        verbosity=opts.verbosity,
    )
    if rc != 0:
        print(f"seed {seed}: NPU dispatch returned {rc}", file=sys.stderr)
        return rc
    o_t.to("cpu")
    out_full = o_t.numpy()
    y_dev = out_full[:D]
    dev_scale = struct.unpack("<f", out_full[D : D + 4].tobytes())[0]

    # Each layer's appended slot P (body and per-slot scale) must match byte for
    # byte, every KV head.
    kv_t.to("cpu")
    kv_dev = kv_t.numpy()
    append_fails = 0
    append_fails_per_layer = [0] * N_LAYERS
    for L in range(N_LAYERS):
        base = L * PER_LAYER_KV
        for h in range(N_HEADS_KV):
            tu = base + h * PER_KV_HEAD_BYTES
            k_off = tu + T_USED_BYTES
            v_off = k_off + KCACHE_PADDED
            kb = kv_dev[
                k_off + KV_HEADER + P * HEAD_D : k_off + KV_HEADER + (P + 1) * HEAD_D
            ]
            vb = kv_dev[
                v_off + KV_HEADER + P * HEAD_D : v_off + KV_HEADER + (P + 1) * HEAD_D
            ]
            kd = int(
                np.abs(
                    kb.astype(np.int32)
                    - kref[L][h][P * HEAD_D : (P + 1) * HEAD_D].astype(np.int32)
                ).max()
            )
            vd = int(
                np.abs(
                    vb.astype(np.int32)
                    - vref[L][h][P * HEAD_D : (P + 1) * HEAD_D].astype(np.int32)
                ).max()
            )
            ks_dev = kv_dev[k_off + P * 4 : k_off + P * 4 + 4].tobytes()
            vs_dev = kv_dev[v_off + P * 4 : v_off + P * 4 + 4].tobytes()
            scales_match = ks_dev == np.float32(ksref[L][h, P]).tobytes() and (
                vs_dev == np.float32(vsref[L][h, P]).tobytes()
            )
            if kd or vd or not scales_match:
                append_fails += 1
                append_fails_per_layer[L] += 1
                if os.environ.get("LLAMA_APPEND_DIAG"):
                    print(
                        f"    L{L} h{h}: kd={kd} vd={vd} "
                        f"scales_match={scales_match}"
                    )
    if append_fails:
        print(f"  append_fails_per_layer={append_fails_per_layer}")

    diff = y_dev.astype(np.int32) - y_ref.astype(np.int32)
    n_mismatch = int((diff != 0).sum())
    max_abs = int(np.abs(diff).max()) if n_mismatch else 0
    scale_match = struct.pack("<f", dev_scale) == struct.pack(
        "<f", np.float32(xo_scale)
    )

    if n_mismatch == 0 and scale_match and append_fails == 0:
        sat = int((y_ref == 127).sum() + (y_ref == -128).sum())
        print(
            f"seed {seed}: BIT-EXACT  (mismatch={n_mismatch}/{D} max|d|={max_abs} "
            f"sat={sat}/{D} scale={dev_scale:.10g}  "
            f"append=OK[{N_LAYERS * N_HEADS_KV}/{N_LAYERS * N_HEADS_KV}])"
        )
        return 0
    idx = np.where(diff != 0)[0][:10]
    print(
        f"seed {seed}: FAIL  mismatch={n_mismatch}/{D}  max|d|={max_abs}  "
        f"scale_match={scale_match} append_fails={append_fails}/{N_LAYERS * N_HEADS_KV} "
        f"dev_scale={dev_scale:.10g} ref_scale={float(xo_scale):.10g}"
    )
    for i in idx:
        print(f"  i={i}: dev={int(y_dev[i])} ref={int(y_ref[i])}")
    return 1


def main():
    p = argparse.ArgumentParser()
    add_artifact_args(p)
    p.add_argument(
        "--seeds", type=str, default="0,1,7,42", help="comma-separated seed list"
    )
    opts = p.parse_args()
    seeds = [int(s) for s in opts.seeds.split(",")]

    npu_kernel = design_kernel(build, opts, N_LAYERS=N_LAYERS, T=T)
    lut_exp = exp_lut(EXP_QUANT_SCALE).astype(np.float32)
    lut_silu = silu_lut(SILU_GATE_SCALE)

    print(f"chain_mh N_LAYERS={N_LAYERS}  T={T}  seeds={seeds}")
    fails = 0
    for s in seeds:
        fails += run_one_seed(s, opts, lut_exp, lut_silu, npu_kernel) != 0
    print(
        f"\nchain_mh: {len(seeds) - fails}/{len(seeds)} seeds PASS "
        f"(residual and KV appends byte-exact)"
    )
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
