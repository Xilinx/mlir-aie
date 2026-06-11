"""Phase 6c.5b.2: FFN-half with per-token activation scales as runtime args.

Builds the xclbin ONCE (via `make ffn_half`). Each test invocation packs
calibrated per-stage scales into the wblob's slot prefixes and dispatches
the same xclbin — no rebuild between seeds.

Slot prefix layout:
  - gate slot: no prefix (gate uses _perchan kernel with closure-baked
    scales; gate_inv_out_scale is locked to 1/SILU_GATE_SCALE).
  - up slot:   16 B prefix = (up_act_scale fp32, up_inv_out_scale fp32,
                              silu_up_scale fp32, silu_inv_out_scale fp32)
  - down slot:  8 B prefix = (down_act_scale fp32, down_inv_out_scale fp32)

Up's worker mirrors the silu 8 B into of_uf's tail (HD+8 buffer).
Silu reads those 8 B from up[HD..HD+8].
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_ffn_half import (
    D,
    HD,
    N_TILE,
    N_TILES_GATE,
    N_TILES_UP,
    N_TILES_DOWN,
    OFF_GAMMA,
    OFF_WG,
    OFF_WU,
    OFF_WD,
    GAMMA_BYTES,
    ACT_SCALE,
    SILU_GATE_SCALE,
    GATE_INV_OUT_SCALE,
    WG_SLOT,
    WU_SLOT,
    WD_SLOT,
    WEIGHTS_BYTES,
)
from test_rmsnorm_int8 import numpy_rmsnorm_int8
from test_silu_mul_int8 import numpy_silu_mul
from gen_silu_lut import silu_lut
from gen_llama_data import quant_int8_perchan_absmax


def i8_add_wrap(a, b):
    return (a.astype(np.int32) + b.astype(np.int32)).astype(np.int8)


def numpy_gemm_perchan(act_i8, x_scale, w_i8, w_scales, bias_i32, inv_out_scale):
    """Bit-exact with the perchan kernel.
    acc = w@a (int32) + bias
    fp = float(acc) * x_scale * w_scales[n]
    out = round_half_away_from_zero(fp * inv_out_scale).clip(-128,127)
    """
    acc = w_i8.astype(np.int32) @ act_i8.astype(np.int32) + bias_i32
    fp = acc.astype(np.float32) * np.float32(x_scale) * w_scales.astype(np.float32)
    scaled = fp * np.float32(inv_out_scale)
    rounded = np.where(
        scaled >= 0,
        np.floor(scaled + np.float32(0.5)),
        np.ceil(scaled - np.float32(0.5)),
    )
    return rounded.clip(-128, 127).astype(np.int8)


def pack_perchan_slots(w_i8, w_scales, bias_i32, n_tile, prefix_bytes: bytes = b""):
    """Pack per-tile slots:
       [prefix_bytes | N_TILE*K i8 | N_TILE i32 bias | N_TILE fp32 w_scales]
    prefix_bytes is a per-slot constant (repeated in every tile slot).
    """
    N, K = w_i8.shape
    n_tiles = N // n_tile
    pre = len(prefix_bytes)
    slot = pre + n_tile * K + n_tile * 4 + n_tile * 4
    blob = np.zeros(n_tiles * slot, dtype=np.int8)
    pre_arr = np.frombuffer(prefix_bytes, dtype=np.int8) if pre else None
    for t in range(n_tiles):
        base = t * slot
        off = base
        if pre:
            blob[off : off + pre] = pre_arr
            off += pre
        rows = w_i8[t * n_tile : (t + 1) * n_tile].flatten()
        bs = bias_i32[t * n_tile : (t + 1) * n_tile].view(np.int8).flatten()
        ws = w_scales[t * n_tile : (t + 1) * n_tile].view(np.int8).flatten()
        blob[off : off + rows.size] = rows
        off += rows.size
        blob[off : off + bs.size] = bs
        off += bs.size
        blob[off : off + ws.size] = ws
    return blob


def fp32_bytes(*vals):
    return np.array(vals, dtype=np.float32).tobytes()


def run_one_seed(seed: int, opts, lut_bf, npu_kernel) -> int:
    """One end-to-end calibrate-pack-dispatch-verify. Returns 0 on bit-exact."""
    rng = np.random.default_rng(seed)

    # --- Inputs / weights ---
    x1_in = rng.integers(-32, 33, size=D, dtype=np.int8)
    gamma_post = (1.0 + 0.1 * rng.standard_normal(D).astype(np.float32)).astype(
        bfloat16
    )

    def random_w(out_dim, in_dim):
        base = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.05
        row_scale = rng.uniform(0.1, 1.0, size=out_dim).astype(np.float32)
        return base * row_scale[:, None]

    wg_fp = random_w(HD, D)
    bg = np.zeros(HD, dtype=np.int32)
    wu_fp = random_w(HD, D)
    bu = np.zeros(HD, dtype=np.int32)
    wd_fp = random_w(D, HD)
    bd = np.zeros(D, dtype=np.int32)

    wg_i8, wg_sc = quant_int8_perchan_absmax(wg_fp)
    wu_i8, wu_sc = quant_int8_perchan_absmax(wu_fp)
    wd_i8, wd_sc = quant_int8_perchan_absmax(wd_fp)

    # --- Numpy forward pass to calibrate per-stage scales ---
    h2 = numpy_rmsnorm_int8(x1_in, gamma_post, ACT_SCALE, 1.0 / ACT_SCALE)

    # Gate's output scale is LOCKED to SILU_GATE_SCALE (baked LUT). Up's
    # output scale is free — calibrate it independently.
    gate_inv_out_scale = float(np.float32(GATE_INV_OUT_SCALE))
    gate_out_scale = float(np.float32(SILU_GATE_SCALE))

    def gemm_fp(act_i8, x_scale, w_i8, w_sc, bias):
        acc = w_i8.astype(np.int32) @ act_i8.astype(np.int32) + bias
        return acc.astype(np.float32) * np.float32(x_scale) * w_sc.astype(np.float32)

    fp_gate = gemm_fp(h2, ACT_SCALE, wg_i8, wg_sc, bg)
    fp_up = gemm_fp(h2, ACT_SCALE, wu_i8, wu_sc, bu)
    up_out_scale = float(np.maximum(np.abs(fp_up).max(), 1e-12)) / 127.0
    up_inv_out_scale = float(np.float32(1.0) / np.float32(up_out_scale))

    def requant(fp, inv):
        scaled = fp * np.float32(inv)
        rounded = np.where(
            scaled >= 0,
            np.floor(scaled + np.float32(0.5)),
            np.ceil(scaled - np.float32(0.5)),
        )
        return rounded.clip(-128, 127).astype(np.int8)

    gf = requant(fp_gate, gate_inv_out_scale)
    uf = requant(fp_up, up_inv_out_scale)

    # Silu: LUT keyed by gate_scale (= SILU_GATE_SCALE, baked). up_scale =
    # up_out_scale (dynamic). Calibrate silu's output scale.
    silu_up_scale = up_out_scale
    silu_gate_fp = lut_bf[gf.astype(np.int32) + 128].astype(np.float32)
    silu_up_fp = uf.astype(np.float32) * np.float32(silu_up_scale)
    sf_fp = silu_gate_fp * silu_up_fp
    sf_out_scale = float(np.maximum(np.abs(sf_fp).max(), 1e-12)) / 127.0
    silu_inv_out_scale = float(np.float32(1.0) / np.float32(sf_out_scale))
    sf = numpy_silu_mul(gf, uf, lut_bf, silu_up_scale, silu_inv_out_scale)

    # Down: sf @ wd. Calibrate output scale.
    down_act_scale = sf_out_scale
    fp_down = gemm_fp(sf, down_act_scale, wd_i8, wd_sc, bd)
    df_out_scale = float(np.maximum(np.abs(fp_down).max(), 1e-12)) / 127.0
    down_inv_out_scale = float(np.float32(1.0) / np.float32(df_out_scale))
    df = requant(fp_down, down_inv_out_scale)

    expected = i8_add_wrap(df, x1_in)

    # --- Pack weights blob with per-stage scale prefixes ---
    wblob = np.zeros(WEIGHTS_BYTES, dtype=np.int8)
    wblob[OFF_GAMMA : OFF_GAMMA + GAMMA_BYTES] = gamma_post.view(np.int8)

    # Gate: no prefix (closure-baked).
    wg_packed = pack_perchan_slots(wg_i8, wg_sc, bg, N_TILE)
    # Up: 64 B prefix = (act_scale, inv_out_scale, silu_up_scale, silu_inv_out)
    # + 48 B pad (alignment for aie::load_v<64> on weights).
    up_prefix = (
        fp32_bytes(ACT_SCALE, up_inv_out_scale, silu_up_scale, silu_inv_out_scale)
        + b"\x00" * 48
    )
    wu_packed = pack_perchan_slots(wu_i8, wu_sc, bu, N_TILE, prefix_bytes=up_prefix)
    # Down: 64 B prefix = (down_act_scale, down_inv_out_scale) + 56 B pad.
    down_prefix = fp32_bytes(down_act_scale, down_inv_out_scale) + b"\x00" * 56
    wd_packed = pack_perchan_slots(wd_i8, wd_sc, bd, N_TILE, prefix_bytes=down_prefix)

    assert wg_packed.size == N_TILES_GATE * WG_SLOT, (
        wg_packed.size,
        N_TILES_GATE * WG_SLOT,
    )
    assert wu_packed.size == N_TILES_UP * WU_SLOT, (
        wu_packed.size,
        N_TILES_UP * WU_SLOT,
    )
    assert wd_packed.size == N_TILES_DOWN * WD_SLOT, (
        wd_packed.size,
        N_TILES_DOWN * WD_SLOT,
    )
    wblob[OFF_WG : OFF_WG + wg_packed.size] = wg_packed
    wblob[OFF_WU : OFF_WU + wu_packed.size] = wu_packed
    wblob[OFF_WD : OFF_WD + wd_packed.size] = wd_packed

    # --- NPU dispatch ---
    x_t = iron.tensor(x1_in, dtype=np.int8)
    w_t = iron.tensor(wblob, dtype=np.int8)
    o_t = iron.zeros([D], dtype=np.int8)

    rc = DefaultNPURuntime.run_test(
        npu_kernel,
        [x_t, w_t, o_t],
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
    sat = int((expected == 127).sum() + (expected == -128).sum())
    print(
        f"  seed={seed:>3}  mismatches={n_diff}/{D}  max|diff|={max_abs}  "
        f"saturated={sat}/{D}  "
        f"up_inv={up_inv_out_scale:.3f}  silu_inv={silu_inv_out_scale:.3f}  "
        f"down_inv={down_inv_out_scale:.3f}"
    )
    if n_diff != 0:
        for i in np.argwhere(diff != 0).flatten()[:8]:
            print(f"    i={i}: NPU={actual[i]}  expected={expected[i]}  x={x1_in[i]}")
    return n_diff


def main():
    p = test_utils.create_default_argparser()
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed list (default: LLAMA_FFN_SEED env or 0,1,7,42)",
    )
    opts = p.parse_args()

    # LUT is build-time fixed; precompute the bf16 array once for numpy ref.
    lut_bf = silu_lut(SILU_GATE_SCALE)
    npu_kernel = test_utils.create_npu_kernel(opts).npu_kernel

    if opts.seeds is not None:
        seeds = [int(s) for s in opts.seeds.split(",") if s.strip()]
    elif "LLAMA_FFN_SEED" in os.environ:
        seeds = [int(os.environ["LLAMA_FFN_SEED"])]
    else:
        seeds = [0, 1, 7, 42]

    print(
        f"FFN-half (perchan + dynamic-scale runtime) D={D} HD={HD} "
        f"N_TILE={N_TILE}  SILU_GATE_SCALE={SILU_GATE_SCALE}  "
        f"seeds={seeds}"
    )
    print("  (same xclbin for every seed — no rebuild)")

    n_fail = 0
    for s in seeds:
        rc = run_one_seed(s, opts, lut_bf, npu_kernel)
        if rc != 0:
            n_fail += 1
    if n_fail == 0:
        print(f"BIT-EXACT PASS x {len(seeds)}  (per-channel + dynamic-scale FFN-half)")
        return 0
    print(f"FAIL: {n_fail}/{len(seeds)} seeds had mismatches")
    return 1


if __name__ == "__main__":
    sys.exit(main())
