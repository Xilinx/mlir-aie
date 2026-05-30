"""Phase 2 RMSNorm bit-exact-ish test.

NPU uses aie::invsqrt (HW intrinsic, ~bf16-precision approximation);
numpy uses fp64 1/sqrt(). Strict bit-exact match isn't reachable, so
we check max abs+rel error against a tolerance instead. Tolerance set
to match the matmul example's bf16 tolerances; tighten if needed.
"""

from __future__ import annotations

import sys

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime


def numpy_rmsnorm(x: np.ndarray, g: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    xf = x.astype(np.float32)
    gf = g.astype(np.float32)
    var = np.mean(xf * xf, axis=-1, keepdims=True)
    xn = xf / np.sqrt(var + eps)
    return (xn * gf).astype(bfloat16)


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("-D", type=int, default=2048)
    # bf16 mantissa is 7 bits, so 1 ULP at magnitude ~2 is 2^-6 ~= 0.015,
    # and a few ULPs of accumulated error is normal. 0.05 abs / 0.05 rel
    # leaves headroom; tighten once we know the typical NPU vs numpy gap.
    p.add_argument("--abs-tol", type=float, default=0.05)
    p.add_argument("--rel-tol", type=float, default=0.05)
    opts = p.parse_args()

    D = opts.D
    rng = np.random.default_rng(0)
    # Activations roughly normal scale (~unit variance, similar to a
    # post-embedding residual stream).
    x = rng.standard_normal(D).astype(np.float32).astype(bfloat16)
    # Gamma per llama init: ones (no learned shift initially; for testing
    # we slightly perturb so any "gamma dropped" bug shows up).
    g = (1.0 + 0.1 * rng.standard_normal(D).astype(np.float32)).astype(bfloat16)

    x_t = iron.tensor(x, dtype=bfloat16)
    g_t = iron.tensor(g, dtype=bfloat16)
    out_t = iron.zeros([D], dtype=bfloat16)

    npu_opts = test_utils.create_npu_kernel(opts)
    rc = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [x_t, g_t, out_t],
        {},
        verify=False,
        verbosity=opts.verbosity,
    )
    if rc != 0:
        return rc

    out_t.to("cpu")
    actual = out_t.numpy().astype(np.float32)
    expected = numpy_rmsnorm(x, g).astype(np.float32)

    abs_err = np.abs(actual - expected)
    rel_err = abs_err / (np.abs(expected) + 1e-7)
    max_abs = float(abs_err.max())
    max_rel = float(rel_err.max())
    cos = float(np.dot(actual, expected) /
                (np.linalg.norm(actual) * np.linalg.norm(expected) + 1e-12))

    print(
        f"RMSNorm NPU vs numpy: D={D}  max|abs|={max_abs:.4g}  "
        f"max|rel|={max_rel:.4g}  cos={cos:.6f}"
    )

    if max_abs <= opts.abs_tol and max_rel <= opts.rel_tol:
        print("PASS")
        return 0
    print(f"FAIL (abs_tol={opts.abs_tol}, rel_tol={opts.rel_tol})")
    # Show top-error positions.
    bad = np.argsort(abs_err)[-5:]
    for i in bad:
        print(f"  i={i}: NPU={actual[i]:.6g} expected={expected[i]:.6g} "
              f"abs={abs_err[i]:.4g}  rel={rel_err[i]:.4g}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
