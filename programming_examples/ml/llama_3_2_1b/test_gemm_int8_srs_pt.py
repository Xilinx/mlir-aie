"""Phase 1 dataflow sanity test: stub kernel copies act -> out.

Builds the xclbin (via Makefile) for `aie2_gemm_int8_srs.py` with the
passthrough stub, runs it on the NPU with random byte fills for both
input fifos, and checks that the output equals the first M*N bytes of
the act buffer. Bit-exact (memcmp) -- any deviation is a wiring bug.
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
    w_blob_bytes = N * K + N * 4 + N * 4  # matches aie2_gemm_int8_srs.py
    rng = np.random.default_rng(0)
    act    = rng.integers(-128, 128, size=M * K,        dtype=np.int8)
    w_blob = rng.integers(-128, 128, size=w_blob_bytes, dtype=np.int8)

    act_t    = iron.tensor(act,    dtype=np.int8)
    w_blob_t = iron.tensor(w_blob, dtype=np.int8)
    out_t    = iron.zeros([M * N], dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    rc = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [act_t, w_blob_t, out_t],
        {},
        verify=False,
        verbosity=opts.verbosity,
    )
    if rc != 0:
        return rc

    out_t.to("cpu")
    actual = out_t.numpy()

    # Stub copies first M*N bytes of act -> out. At M=8,K=64,N=64 that's
    # M*N == M*K == 512, i.e. the entire act buffer.
    expected = act[: M * N]

    n_diff = int((actual != expected).sum())
    if n_diff == 0:
        print(f"BIT-EXACT NPU passthrough OK (M={M}, K={K}, N={N}, {M*N} bytes)")
        return 0

    print(f"MISMATCH: {n_diff}/{M*N} bytes differ")
    diffs = np.argwhere(actual != expected)[:8]
    for (i,) in diffs:
        print(f"  out[{i}]={actual[i]}  expected={expected[i]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
