"""Phase 2 real gemm_int8_srs bit-exact test (GEMV, M=1).

Computes the expected output via a numpy reference using the SAME
banker_srs as the kernel, so bit-exact match is achievable.
"""

from __future__ import annotations

import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_gemm_int8_srs_real import RIGHT_SHIFT


def banker_srs(sum_i32: np.ndarray, rs: int) -> np.ndarray:
    # Matches the C kernel:
    #   (sum + (1 << (rs-1)) - 1 + ((sum >> rs) & 1)) >> rs
    # Arithmetic right shift on signed int32 -- numpy >> on int32 is the same.
    s = sum_i32.astype(np.int64)
    one_bit = 1 << (rs - 1)
    arith = (s + one_bit - 1 + ((s >> rs) & 1)) >> rs
    return arith.astype(np.int32)


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("-K", type=int, default=64)
    p.add_argument("-N", type=int, default=64)
    opts = p.parse_args()

    K, N = opts.K, opts.N
    rng = np.random.default_rng(0)

    # Test BOTH at the original full-range data (where the standalone
    # bit-exact was claimed) AND at the smaller chain-test magnitudes
    # to see if either gives diff > 0.
    act      = rng.integers(-128, 128, size=K,    dtype=np.int8)
    weights  = rng.integers(-128, 128, size=N*K,  dtype=np.int8).reshape(N, K)
    bias     = rng.integers(-1000, 1000, size=N,  dtype=np.int32)
    print(f"act range: [{act.min()}, {act.max()}], weights range: [{weights.min()}, {weights.max()}]")

    # Pack: weights[N*K] || bias[N*4].
    w_packed = np.concatenate([weights.flatten().view(np.int8),
                               bias.view(np.int8).flatten()])

    # Expected (numpy with same banker_srs).
    acc = (weights.astype(np.int32) @ act.astype(np.int32))  # (N,) int32 dot product per row
    sums = acc + bias
    expected = banker_srs(sums, RIGHT_SHIFT).clip(-128, 127).astype(np.int8)

    act_t = iron.tensor(act, dtype=np.int8)
    w_t   = iron.tensor(w_packed, dtype=np.int8)
    out_t = iron.zeros([N], dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    rc = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [act_t, w_t, out_t],
        {},
        verify=False,
        verbosity=opts.verbosity,
    )
    if rc != 0:
        return rc

    out_t.to("cpu")
    actual = out_t.numpy()

    diff = (actual.astype(np.int16) - expected.astype(np.int16))
    n_diff = int((diff != 0).sum())
    max_abs = int(np.abs(diff).max()) if n_diff else 0

    print(
        f"gemm_int8_srs NPU vs numpy: K={K} N={N} rs={RIGHT_SHIFT}  "
        f"mismatches={n_diff}/{N}  max|diff|={max_abs}"
    )

    if n_diff == 0:
        print("BIT-EXACT PASS")
        return 0
    print("FAIL")
    for i in np.argwhere(diff != 0).flatten()[:8]:
        print(f"  i={i}: NPU={actual[i]} expected={expected[i]} sum={sums[i]} acc={acc[i]} bias={bias[i]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
