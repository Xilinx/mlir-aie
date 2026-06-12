"""Phase 2 int8 silu_mul test (LUT-based, bit-exact by construction).

Uses the same SiLU LUT as the kernel (generated from gen_silu_lut.py
with the same gate_scale). The reference replicates the kernel's
arithmetic exactly:
  silu_bf = silu_lut[gate_i8 + 128]   # bf16
  out_f   = silu_bf * (up_i8 * up_scale) * inv_out_scale
  out_i8  = clamp(round(out_f), -128, 127)
"""

from __future__ import annotations

import os
import struct
import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_silu_mul_int8 import GATE_SCALE, UP_SCALE, INV_OUT_SCALE, SELFCAL
from gen_silu_lut import silu_lut


def round_to_i8(v):
    r = np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5)).astype(np.int32)
    return np.clip(r, -128, 127).astype(np.int8)


def numpy_silu_mul(gate, up, lut, up_scale, inv_out_scale):
    # silu_bf via LUT; matches the kernel byte for byte.
    silu_bf = lut[(gate.astype(np.int32) + 128)]  # bf16
    s_f = silu_bf.astype(np.float32)
    u_f = up.astype(np.float32) * up_scale
    out_f = s_f * u_f * inv_out_scale
    return round_to_i8(out_f)


def main():
    from test_rmsnorm_int8_dyn import sw_recip

    p = test_utils.create_default_argparser()
    p.add_argument("-D", type=int, default=8192)
    opts = p.parse_args()

    D = opts.D
    rng = np.random.default_rng(0)
    gate = rng.integers(-128, 128, size=D, dtype=np.int8)
    up = rng.integers(-128, 128, size=D, dtype=np.int8)

    lut = silu_lut(GATE_SCALE)  # exact same arithmetic as the kernel's baked LUT

    # up buffer carries up_scale in its tail (dyn/selfcal both read up[D]).
    up_buf = np.zeros(D + 8, dtype=np.int8)
    up_buf[:D] = up
    up_buf[D : D + 4] = np.frombuffer(np.float32(UP_SCALE).tobytes(), dtype=np.int8)
    if not SELFCAL:
        up_buf[D + 4 : D + 8] = np.frombuffer(
            np.float32(INV_OUT_SCALE).tobytes(), dtype=np.int8
        )

    out_n = D + 8 if SELFCAL else D + 8  # both paths now HD+8 out in this harness
    g_t = iron.tensor(gate, dtype=np.int8)
    u_t = iron.tensor(up_buf, dtype=np.int8)
    o_t = iron.zeros([out_n], dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    rc = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [g_t, u_t, o_t],
        {},
        verify=False,
        verbosity=opts.verbosity,
    )
    if rc != 0:
        return rc

    o_t.to("cpu")
    out_full = o_t.numpy()
    actual = out_full[:D]

    # Expected: selfcal computes its own inv_out from absmax(s_f*u_f); dyn uses
    # the host INV_OUT_SCALE.
    silu_bf = lut[(gate.astype(np.int32) + 128)]
    s_f = silu_bf.astype(np.float32)
    u_f = up.astype(np.float32) * np.float32(UP_SCALE)
    sf_fp = s_f * u_f
    if SELFCAL:
        absmax = max(float(np.abs(sf_fp).max()), 1e-12)
        out_scale = np.float32(absmax * np.float32(1.0 / 127.0))
        inv = np.float32(sw_recip(out_scale))
        expected = round_to_i8(sf_fp * inv)
        dev_scale = struct.unpack("<f", out_full[D : D + 4].tobytes())[0]
        print(
            f"selfcal: dev_scale={dev_scale:.8g} numpy_out_scale={float(out_scale):.8g} "
            f"rel={abs(dev_scale-float(out_scale))/float(out_scale):.2e}"
        )
    else:
        expected = numpy_silu_mul(gate, up, lut, UP_SCALE, INV_OUT_SCALE)

    diff = actual.astype(np.int16) - expected.astype(np.int16)
    n_diff = int((diff != 0).sum())
    max_abs = int(np.abs(diff).max()) if n_diff else 0

    print(
        f"silu_mul NPU vs numpy: D={D} selfcal={SELFCAL} "
        f"mismatches={n_diff}/{D}  max|int8 diff|={max_abs}"
    )

    if n_diff == 0:
        print("BIT-EXACT PASS")
        return 0
    print("FAIL")
    for i in np.argwhere(diff != 0).flatten()[:8]:
        print(
            f"  i={i}: NPU={actual[i]}  expected={expected[i]}  "
            f"gate={gate[i]} (lut={float(lut[gate[i]+128]):.4f})  up={up[i]}"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
