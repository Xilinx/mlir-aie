"""Phase 1.5 dataflow sanity test: 2-CT single-column projection slice.

Both CTs run the stub (act -> out passthrough); the combined output
should equal act concatenated with itself. Bit-exact.
"""

from __future__ import annotations

import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("-M", type=int, default=8)
    p.add_argument("-K", type=int, default=64)
    p.add_argument("-N", type=int, default=64)
    opts = p.parse_args()

    M, K, N = opts.M, opts.K, opts.N
    w_blob_bytes = N * K + N * 4 + N * 4

    rng = np.random.default_rng(0)
    act = rng.integers(-128, 128, size=M * K,             dtype=np.int8)
    w   = rng.integers(-128, 128, size=2 * w_blob_bytes,  dtype=np.int8)

    act_t = iron.tensor(act, dtype=np.int8)
    w_t   = iron.tensor(w,   dtype=np.int8)
    out_t = iron.zeros([2 * M * N], dtype=np.int8)

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

    # Stub: each CT copies first M*N bytes of act -> out_slice.
    # Combined output should be [act, act].
    expected = np.concatenate([act[: M * N], act[: M * N]])

    n_diff = int((actual != expected).sum())
    if n_diff == 0:
        print(
            f"BIT-EXACT NPU 2-CT broadcast OK "
            f"(M={M}, K={K}, N={N}, combined out = 2 * {M*N} = {2*M*N} bytes)"
        )
        return 0

    print(f"MISMATCH: {n_diff}/{2*M*N} bytes differ")
    # Show top/bot breakdown.
    top_diff = int((actual[:M*N] != expected[:M*N]).sum())
    bot_diff = int((actual[M*N:] != expected[M*N:]).sum())
    print(f"  top-tile slice (bytes 0..{M*N}): {top_diff} diffs")
    print(f"  bot-tile slice (bytes {M*N}..{2*M*N}): {bot_diff} diffs")
    diffs = np.argwhere(actual != expected)[:8]
    for (i,) in diffs:
        print(f"  out[{i}]={actual[i]}  expected={expected[i]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
