"""Test the memtile write-once / read-Rx probe: out[p*V:(p+1)*V] == in for all p."""

from __future__ import annotations

import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_memrepeat_probe import V, R


def main():
    opts = test_utils.create_default_argparser().parse_args()
    rng = np.random.default_rng(0)
    src = rng.uniform(-10, 10, size=V).astype(np.float32)

    in_t = iron.tensor(src, dtype=np.float32)
    out_t = iron.zeros([V * R], dtype=np.float32)
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
    for p in range(R):
        seg = out[p * V : (p + 1) * V]
        ok = np.array_equal(seg, src)
        fails += not ok
        print(f"pass {p}: {'OK' if ok else 'MISMATCH'} (first={seg[0]:.3f} ref={src[0]:.3f})")
    print(f"\nmemrepeat_probe: {'PASS' if fails == 0 else f'{fails} FAIL'}  (V={V} R={R})")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
