"""Phase 2 int8 RMSNorm test.

Bit-exact-ish vs a numpy modeled reference that mimics the kernel:
bf16 internal precision (gamma is bf16), aie::invsqrt is approximated
by 1/sqrt(float) (close enough for our tol), final int8 cast uses
the same round-half-away-from-zero + saturating clamp as the kernel.
"""

from __future__ import annotations

import sys

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_rmsnorm_int8 import ACT_SCALE_IN, INV_ACT_SCALE_OUT


def round_to_i8(v: np.ndarray) -> np.ndarray:
    r = np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5)).astype(np.int32)
    return np.clip(r, -128, 127).astype(np.int8)


def sw_invsqrt(a: float) -> float:
    """Quake-III + 2 NR; matches the kernel's sw_invsqrt byte-for-byte."""
    a32 = np.float32(a)
    bits = a32.view(np.int32)
    new_bits = np.int32(np.uint32(0x5f3759df)) - (bits >> np.int32(1))
    x = new_bits.view(np.float32)
    half = np.float32(0.5)
    onehalf = np.float32(1.5)
    x = np.float32(x * (onehalf - half * a32 * x * x))
    x = np.float32(x * (onehalf - half * a32 * x * x))
    return float(x)


def numpy_rmsnorm_int8(x_i8: np.ndarray, gamma_bf16: np.ndarray,
                       act_scale_in: float, inv_act_scale_out: float,
                       eps: float = 1e-5) -> np.ndarray:
    # Match the kernel exactly: int32 sum-of-squares, fp32 scalar var/
    # invsqrt/combined, then bf16 mul chain on activations.
    sum_sq_i32 = int((x_i8.astype(np.int32) ** 2).sum())   # exact int sum
    var = np.float32(np.float32(sum_sq_i32) * np.float32(act_scale_in) *
                     np.float32(act_scale_in) / np.float32(len(x_i8)))
    inv_rms = sw_invsqrt(np.float32(var + np.float32(eps)))
    combined = np.float32(np.float32(act_scale_in) * np.float32(inv_rms) *
                          np.float32(inv_act_scale_out))
    # Match the kernel exactly: single fp32 multiplication chain per
    # element, no bf16 truncation in pass 2.
    g_f = gamma_bf16.astype(np.float32)  # bf16 -> fp32 (exact)
    x_f = x_i8.astype(np.float32)        # int8 -> fp32 (exact)
    out_f = (x_f * combined * g_f).astype(np.float32)
    return round_to_i8(out_f)


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("-D", type=int, default=2048)
    # With fp32-scalar pass 2 + sw_invsqrt + exact-int sum-of-squares,
    # the kernel and reference compute byte-identical fp32 values --
    # true bit-exact, zero tolerance.
    p.add_argument("--max-mismatches", type=int, default=0)
    opts = p.parse_args()

    D = opts.D
    rng = np.random.default_rng(0)

    # Random int8 activations and bf16 gamma roughly around 1.
    x = rng.integers(-128, 128, size=D, dtype=np.int8)
    g = (1.0 + 0.1 * rng.standard_normal(D).astype(np.float32)).astype(bfloat16)

    x_t = iron.tensor(x, dtype=np.int8)
    g_t = iron.tensor(g, dtype=bfloat16)
    y_t = iron.zeros([D], dtype=np.int8)

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
    actual = y_t.numpy()
    expected = numpy_rmsnorm_int8(x, g, ACT_SCALE_IN, INV_ACT_SCALE_OUT)

    diff = (actual.astype(np.int16) - expected.astype(np.int16))
    n_diff = int((diff != 0).sum())
    max_abs = int(np.abs(diff).max()) if n_diff else 0

    print(
        f"RMSNorm-int8 NPU vs modeled-numpy: D={D}  mismatches={n_diff}/{D}  "
        f"max|int8 diff|={max_abs}"
    )

    if n_diff <= opts.max_mismatches:
        if n_diff == 0:
            print("BIT-EXACT PASS")
        else:
            print(f"PASS (<={opts.max_mismatches} 1-LSB diffs)")
        return 0
    print("FAIL")
    # Show some examples.
    for i in np.argwhere(diff != 0).flatten()[:8]:
        print(f"  i={i}: NPU={actual[i]}  expected={expected[i]}  diff={diff[i]}  x={x[i]}  g={float(g[i]):.4f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
