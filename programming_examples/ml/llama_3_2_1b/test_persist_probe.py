"""Test the persistent on-device feedback loop: out[k] == seed + (k+1)."""

from __future__ import annotations

import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_persist_probe import N_ITER, LEN


def main():
    opts = test_utils.create_default_argparser().parse_args()
    seed = np.arange(LEN, dtype=np.int32)
    in_t = iron.tensor(seed, dtype=np.int32)
    out_t = iron.zeros([LEN * N_ITER], dtype=np.int32)
    npu = test_utils.create_npu_kernel(opts).npu_kernel
    rc = DefaultNPURuntime.run_test(
        npu, [in_t, out_t], {}, verify=False, verbosity=opts.verbosity
    )
    if rc != 0:
        print(f"dispatch returned {rc}", file=sys.stderr)
        return rc
    out_t.to("cpu")
    out = out_t.numpy()
    fails = 0
    for k in range(N_ITER):
        seg = out[k * LEN : (k + 1) * LEN]
        exp = seed + (k + 1)
        ok = np.array_equal(seg, exp)
        fails += not ok
        print(f"iter {k}: {'OK' if ok else f'MISMATCH (first={seg[0]} exp={exp[0]})'}")
    print(
        f"\npersist_probe: {'PASS' if fails == 0 else f'{fails} FAIL'}  (N={N_ITER} LEN={LEN})"
    )
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
