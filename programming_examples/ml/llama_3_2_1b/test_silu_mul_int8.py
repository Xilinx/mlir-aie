"""Phase 2 int8 silu_mul test (LUT-based, bit-exact by construction).

Uses the same SiLU LUT as the kernel (generated from gen_silu_lut.py
with the same gate_scale). The reference replicates the kernel's
arithmetic exactly:
  silu_bf = silu_lut[gate_i8 + 128]   # bf16
  out_f   = silu_bf * (up_i8 * up_scale) * inv_out_scale
  out_i8  = clamp(round(out_f), -128, 127)
"""

from __future__ import annotations

import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_silu_mul_int8 import GATE_SCALE, UP_SCALE, INV_OUT_SCALE
from gen_silu_lut import silu_lut


def round_to_i8(v):
    r = np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5)).astype(np.int32)
    return np.clip(r, -128, 127).astype(np.int8)


def numpy_silu_mul(gate, up, lut, up_scale, inv_out_scale):
    # silu_bf via LUT; matches the kernel byte for byte.
    silu_bf = lut[(gate.astype(np.int32) + 128)]      # bf16
    s_f = silu_bf.astype(np.float32)
    u_f = up.astype(np.float32) * up_scale
    out_f = s_f * u_f * inv_out_scale
    return round_to_i8(out_f)


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("-D", type=int, default=8192)
    opts = p.parse_args()

    D = opts.D
    rng = np.random.default_rng(0)
    gate = rng.integers(-128, 128, size=D, dtype=np.int8)
    up   = rng.integers(-128, 128, size=D, dtype=np.int8)

    lut = silu_lut(GATE_SCALE)  # exact same arithmetic as the kernel's baked LUT

    g_t = iron.tensor(gate, dtype=np.int8)
    u_t = iron.tensor(up,   dtype=np.int8)
    o_t = iron.zeros([D], dtype=np.int8)

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
    actual = o_t.numpy()
    expected = numpy_silu_mul(gate, up, lut, UP_SCALE, INV_OUT_SCALE)

    diff = actual.astype(np.int16) - expected.astype(np.int16)
    n_diff = int((diff != 0).sum())
    max_abs = int(np.abs(diff).max()) if n_diff else 0

    print(f"silu_mul_int8 NPU vs numpy (LUT-matched): D={D}  "
          f"mismatches={n_diff}/{D}  max|int8 diff|={max_abs}")

    if n_diff == 0:
        print("BIT-EXACT PASS")
        return 0
    print("FAIL")
    for i in np.argwhere(diff != 0).flatten()[:8]:
        print(f"  i={i}: NPU={actual[i]}  expected={expected[i]}  "
              f"gate={gate[i]} (lut={float(lut[gate[i]+128]):.4f})  up={up[i]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
