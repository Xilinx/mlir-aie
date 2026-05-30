"""Phase 1.6 dataflow sanity test: 16-CT projection fan-out (2 rows x 8 cols).

3 runtime buffers (1 act broadcast, 1 packed weights, 1 packed outputs).
All 16 CTs run the stub (act -> out passthrough). The combined output
should equal `act` concatenated 16 times (one act-sized slice per CT).
Bit-exact across the whole 16-CT projection.
"""

from __future__ import annotations

import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime


N_COLS = 8
N_ROWS = 2


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("-M", type=int, default=8)
    p.add_argument("-K", type=int, default=64)
    p.add_argument("-N", type=int, default=64)
    opts = p.parse_args()

    M, K, N = opts.M, opts.K, opts.N
    w_blob_bytes = N * K + N * 4 + N * 4
    total_w      = N_COLS * N_ROWS * w_blob_bytes
    total_out    = N_COLS * N_ROWS * M * N

    rng = np.random.default_rng(0)
    act_data = rng.integers(-128, 128, size=M * K,    dtype=np.int8)
    w_data   = rng.integers(-128, 128, size=total_w,  dtype=np.int8)

    act_t = iron.tensor(act_data, dtype=np.int8)
    w_t   = iron.tensor(w_data,   dtype=np.int8)
    out_t = iron.zeros([total_out], dtype=np.int8)

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

    # Each of the 16 CTs copies first M*N bytes of act -> out_slice.
    # With M*N == M*K, that's the entire act buffer per CT.
    expected = np.tile(act_data[: M * N], N_COLS * N_ROWS)

    n_diff = int((actual != expected).sum())
    if n_diff == 0:
        print(
            f"BIT-EXACT 16-CT projection fan-out OK "
            f"(M={M}, K={K}, N={N}, {N_COLS} cols x {N_ROWS} rows = "
            f"{N_COLS * N_ROWS} CTs, {total_out} bytes total)"
        )
        return 0

    print(f"MISMATCH: {n_diff}/{total_out} bytes differ")
    # Per-CT breakdown.
    per_tile = M * N
    for c in range(N_COLS):
        for r in range(N_ROWS):
            idx = c * N_ROWS + r
            lo, hi = idx * per_tile, (idx + 1) * per_tile
            d = int((actual[lo:hi] != expected[lo:hi]).sum())
            if d:
                print(f"  col {c} row {r}: {d}/{per_tile} diffs")
    return 1


if __name__ == "__main__":
    sys.exit(main())
