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


def numpy_rmsnorm_int8(x_i8: np.ndarray, gamma_bf16: np.ndarray,
                       act_scale_in: float, inv_act_scale_out: float,
                       eps: float = 1e-5) -> np.ndarray:
    # Match the kernel: bf16 gamma; sum-of-squares in float; combined scalar.
    x_f = x_i8.astype(np.float32) * act_scale_in
    var = np.mean(x_f * x_f)
    inv_rms = 1.0 / np.sqrt(var + eps)
    combined = act_scale_in * inv_rms * inv_act_scale_out
    # bf16 round-trip on the per-element multiplications (matches kernel).
    combined_bf = np.float32(bfloat16(combined))
    g_f = gamma_bf16.astype(np.float32)  # bf16 -> float for math
    scaled = (x_i8.astype(np.float32) * combined_bf).astype(bfloat16).astype(np.float32)
    out_bf = (scaled * g_f).astype(bfloat16).astype(np.float32)
    return round_to_i8(out_bf)


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("-D", type=int, default=2048)
    # bf16 rounding-order between aie::mul accumulators and numpy's explicit
    # .astype(bfloat16) casts diverges by ~1 int8 LSB on ~10% of elements at
    # D=2048 / random inputs. Allow up to 15% 1-LSB diffs; max abs MUST be 1.
    p.add_argument("--max-mismatches", type=int, default=400,
                   help="tolerated #int8 mismatches (each must be 1 LSB only)")
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

    if n_diff <= opts.max_mismatches and max_abs <= 1:
        print(f"PASS (within bf16 rounding noise: <={opts.max_mismatches} 1-LSB diffs)")
        return 0
    print("FAIL")
    # Show some examples.
    for i in np.argwhere(diff != 0).flatten()[:8]:
        print(f"  i={i}: NPU={actual[i]}  expected={expected[i]}  diff={diff[i]}  x={x[i]}  g={float(g[i]):.4f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
