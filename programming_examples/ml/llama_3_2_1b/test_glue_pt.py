"""Phase 1.8 dataflow sanity test: 2-input glue.

Stub computes out = in1 ^ in2. Bit-exact check: actual == in1 ^ in2.
"""

from __future__ import annotations

import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--bytes", type=int, default=512)
    opts = p.parse_args()

    n = opts.bytes
    rng = np.random.default_rng(0)
    in1 = rng.integers(-128, 128, size=n, dtype=np.int8)
    in2 = rng.integers(-128, 128, size=n, dtype=np.int8)

    in1_t = iron.tensor(in1, dtype=np.int8)
    in2_t = iron.tensor(in2, dtype=np.int8)
    out_t = iron.zeros([n], dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    rc = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [in1_t, in2_t, out_t],
        {},
        verify=False,
        verbosity=opts.verbosity,
    )
    if rc != 0:
        return rc

    out_t.to("cpu")
    actual = out_t.numpy()
    expected = (in1 ^ in2).astype(np.int8)

    n_diff = int((actual != expected).sum())
    if n_diff == 0:
        print(f"BIT-EXACT NPU 2-input glue OK ({n} bytes; xor)")
        return 0

    # Diagnose: is the output one of the inputs (i.e., a fifo dropped)?
    if int((actual == in1).sum()) == n:
        print(f"MISMATCH: actual == in1. in2 fifo likely dropped. ({n} bytes)")
    elif int((actual == in2).sum()) == n:
        print(f"MISMATCH: actual == in2. in1 fifo likely dropped. ({n} bytes)")
    else:
        print(f"MISMATCH: {n_diff}/{n} bytes differ")
        diffs = np.argwhere(actual != expected)[:8]
        for (i,) in diffs:
            print(f"  out[{i}]={actual[i]}  expected={expected[i]} (in1={in1[i]} in2={in2[i]})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
