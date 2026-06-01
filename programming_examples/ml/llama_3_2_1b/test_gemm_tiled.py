"""Phase 6c.1 tiled-GEMM bit-exact test.

Same arithmetic as the legacy test_gemm_int8_srs_real.py (same banker_srs)
but the host packs weights+bias in N/N_TILE successive slots so the kernel
processes them per-iteration. Bit-exact == 0 mismatches across the full N
vector, regardless of how many tile iterations were used.
"""

from __future__ import annotations

import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_gemm_tiled import RIGHT_SHIFT
from test_gemm_int8_srs_real import banker_srs


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("-K", type=int, default=2048)
    p.add_argument("-N", type=int, default=64)
    p.add_argument("--n-tile", type=int, default=8)
    opts = p.parse_args()

    K, N, n_tile = opts.K, opts.N, opts.n_tile
    assert N % n_tile == 0, f"N={N} must be a multiple of N_TILE={n_tile}"
    n_tiles = N // n_tile

    rng = np.random.default_rng(0)
    act     = rng.integers(-128, 128, size=K,    dtype=np.int8)
    weights = rng.integers(-128, 128, size=(N, K), dtype=np.int8)
    # Bias range scaled so post-SRS results land mostly in i8 range
    # (not all-saturated). For random i8 act/w at K=2048, |sum| ~ K *
    # 128 * 128 / 3 in expectation ~ 10M; after >>12 ~ 2500; clamp to
    # [-128, 127]. Bias should be on the same order to be meaningful.
    bias    = rng.integers(-2**14, 2**14, size=N, dtype=np.int32)

    # --- Numpy reference (bit-exact same path as the kernel) ---
    sums = weights.astype(np.int32) @ act.astype(np.int32) + bias
    expected = banker_srs(sums, RIGHT_SHIFT).clip(-128, 127).astype(np.int8)

    # Pack inlined: [(w_tile0 i8 | b_tile0 i32) | (w_tile1 | b_tile1) | ...]
    slot_bytes = n_tile * K + n_tile * 4
    w_blob = np.zeros(n_tiles * slot_bytes, dtype=np.int8)
    for t in range(n_tiles):
        base = t * slot_bytes
        w_slice = weights[t * n_tile:(t + 1) * n_tile].flatten()
        b_slice = bias[t * n_tile:(t + 1) * n_tile].view(np.int8).flatten()
        w_blob[base:base + w_slice.size] = w_slice
        w_blob[base + w_slice.size:base + w_slice.size + b_slice.size] = b_slice

    # --- NPU dispatch ---
    a_t = iron.tensor(act,   dtype=np.int8)
    w_t = iron.tensor(w_blob, dtype=np.int8)
    o_t = iron.zeros([N],     dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    rc = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [a_t, w_t, o_t],
        {}, verify=False, verbosity=opts.verbosity,
    )
    if rc != 0:
        return rc
    o_t.to("cpu")
    actual = o_t.numpy()

    diff = (actual.astype(np.int16) - expected.astype(np.int16))
    n_diff = int((diff != 0).sum())
    max_abs = int(np.abs(diff).max()) if n_diff else 0
    sat = int((expected == 127).sum() + (expected == -128).sum())
    print(f"tiled_gemm NPU vs numpy: K={K} N={N} N_TILE={n_tile} "
          f"({n_tiles} tile iters)  mismatches={n_diff}/{N}  "
          f"max|diff|={max_abs}  saturated={sat}/{N}")

    if n_diff == 0:
        print("BIT-EXACT PASS")
        return 0
    print("FAIL")
    for i in np.argwhere(diff != 0).flatten()[:8]:
        print(f"  i={i}: NPU={actual[i]}  expected={expected[i]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
