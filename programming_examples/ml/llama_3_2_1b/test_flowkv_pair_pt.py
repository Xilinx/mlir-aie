"""Phase 1.7 dataflow sanity test: FlowKV qk -> sv pair.

Each stub kernel bitwise-inverts its input. Composed CT0 -> CT1, the
final output should equal the original input. Bit-exact across the
whole 512-byte payload.
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
    in_data = rng.integers(-128, 128, size=n, dtype=np.int8)

    in_t  = iron.tensor(in_data, dtype=np.int8)
    out_t = iron.zeros([n], dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    rc = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [in_t, out_t],
        {},
        verify=False,
        verbosity=opts.verbosity,
    )
    if rc != 0:
        return rc

    out_t.to("cpu")
    actual = out_t.numpy()

    # qk: ~in;  sv: ~that = in. Composition is identity.
    expected = in_data

    n_diff = int((actual != expected).sum())
    if n_diff == 0:
        print(f"BIT-EXACT NPU FlowKV qk->sv pair OK ({n} bytes round-trip)")
        return 0

    # Helpful diagnosis: is the output the input or its complement?
    if int((actual == (~in_data).astype(np.int8)).sum()) == n:
        print(
            f"MISMATCH: actual = ~input. Only one of qk/sv ran (or "
            f"CT->CT stream dropped). {n} bytes."
        )
    else:
        print(f"MISMATCH: {n_diff}/{n} bytes differ")
        diffs = np.argwhere(actual != expected)[:8]
        for (i,) in diffs:
            print(f"  out[{i}]={actual[i]}  expected={expected[i]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
