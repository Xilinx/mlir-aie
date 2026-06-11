"""Bit-exact test for llama_rmsnorm_int8_dyn: int8 output + fp32 scale tail.

The kernel computes per-token absmax of its fp32 output, requants with
inv_dyn = 127/absmax, and writes the dynamic scale (absmax/127) into the
last 4 bytes of the output buffer. The numpy reference here recomputes
the exact same chain (two-pass, no intermediate fp32 array, sw_invsqrt
+ sw_recip) and compares both halves byte-for-byte.
"""

from __future__ import annotations

import struct
import sys

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_rmsnorm_int8_dyn import ACT_SCALE_IN
from test_rmsnorm_int8 import round_to_i8, sw_invsqrt


def sw_recip(a: float) -> float:
    """Bit-exact mirror of the kernel sw_recip (NR over magic-constant init)."""
    a32 = np.float32(a)
    bits = a32.view(np.int32)
    new_bits = np.int32(np.uint32(0x7EF477D5)) - bits
    x = new_bits.view(np.float32)
    two = np.float32(2.0)
    x = np.float32(x * (two - a32 * x))
    x = np.float32(x * (two - a32 * x))
    x = np.float32(x * (two - a32 * x))
    x = np.float32(x * (two - a32 * x))
    return float(x)


def numpy_rmsnorm_int8_dyn(
    x_i8: np.ndarray, gamma_bf16: np.ndarray, act_scale_in: float, eps: float = 1e-5
):
    """Returns (y_i8: int8[D], scale_dyn: fp32)."""
    sum_sq_i32 = int((x_i8.astype(np.int32) ** 2).sum())
    var = np.float32(
        np.float32(sum_sq_i32)
        * np.float32(act_scale_in)
        * np.float32(act_scale_in)
        / np.float32(len(x_i8))
    )
    inv_rms = sw_invsqrt(np.float32(var + np.float32(eps)))
    pre = np.float32(np.float32(act_scale_in) * np.float32(inv_rms))

    g_f = gamma_bf16.astype(np.float32)
    x_f = x_i8.astype(np.float32)
    # Pass A: scan fp32 y_f to find absmax.
    y_f = (x_f * pre * g_f).astype(np.float32)
    absmax = float(np.abs(y_f).max())
    if absmax < 1e-12:
        absmax = 1e-12
    scale_dyn = np.float32(absmax * np.float32(1.0 / 127.0))
    inv_dyn = np.float32(sw_recip(scale_dyn))
    combined = np.float32(pre * inv_dyn)

    # Pass B: recompute (matches the kernel's two-pass-no-intermediate).
    out_f = (x_f * combined * g_f).astype(np.float32)
    y_i8 = round_to_i8(out_f)
    return y_i8, float(scale_dyn)


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("-D", type=int, default=2048)
    p.add_argument("--max-mismatches", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    opts = p.parse_args()

    D = opts.D
    rng = np.random.default_rng(opts.seed)
    x = rng.integers(-128, 128, size=D, dtype=np.int8)
    g = (1.0 + 0.1 * rng.standard_normal(D).astype(np.float32)).astype(bfloat16)

    x_t = iron.tensor(x, dtype=np.int8)
    g_t = iron.tensor(g, dtype=bfloat16)
    y_t = iron.zeros([D + 8], dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    rc = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [x_t, g_t, y_t],
        {},
        verify=False,
        verbosity=opts.verbosity,
    )
    if rc != 0:
        return rc

    y_t.to("cpu")
    actual_full = y_t.numpy()
    actual_y = actual_full[:D]
    actual_scale = struct.unpack("<f", actual_full[D : D + 4].tobytes())[0]

    expected_y, expected_scale = numpy_rmsnorm_int8_dyn(x, g, ACT_SCALE_IN)

    diff = actual_y.astype(np.int16) - expected_y.astype(np.int16)
    n_diff = int((diff != 0).sum())
    max_abs = int(np.abs(diff).max()) if n_diff else 0
    scale_match = struct.pack("<f", actual_scale) == struct.pack("<f", expected_scale)

    print(
        f"RMSNorm-int8-dyn: D={D}  int8 mismatches={n_diff}/{D}  "
        f"max|int8 diff|={max_abs}  "
        f"scale NPU={actual_scale:.10g}  ref={expected_scale:.10g}  "
        f"scale {'EXACT' if scale_match else 'DIFFER'}"
    )

    if n_diff <= opts.max_mismatches and scale_match:
        print(
            "BIT-EXACT PASS"
            if n_diff == 0
            else f"PASS (<={opts.max_mismatches} 1-LSB diffs)"
        )
        return 0
    print("FAIL")
    for i in np.argwhere(diff != 0).flatten()[:8]:
        print(
            f"  i={i}: NPU={actual_y[i]}  expected={expected_y[i]}  "
            f"diff={diff[i]}  x={x[i]}  g={float(g[i]):.4f}"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
