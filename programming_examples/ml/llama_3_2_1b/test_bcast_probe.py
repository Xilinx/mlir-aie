"""Test the broadcast-to-two-consumers probe: one stream feeds reduce + select."""

from __future__ import annotations

import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_bcast_probe import NTILES, TILE, SELECT


def main():
    opts = test_utils.create_default_argparser().parse_args()
    rng = np.random.default_rng(0)
    tbl = rng.integers(-1000, 1000, size=NTILES * TILE, dtype=np.int32)

    tbl_t = iron.tensor(tbl, dtype=np.int32)
    suma_t = iron.zeros([1], dtype=np.int32)
    selb_t = iron.zeros([TILE], dtype=np.int32)
    npu = test_utils.create_npu_kernel(opts).npu_kernel
    rc = DefaultNPURuntime.run_test(
        npu, [tbl_t, suma_t, selb_t], {}, verify=False, verbosity=opts.verbosity
    )
    if rc != 0:
        print(f"dispatch returned {rc}", file=sys.stderr)
        return rc
    suma_t.to("cpu")
    selb_t.to("cpu")
    dev_sum = int(suma_t.numpy()[0])
    dev_sel = selb_t.numpy()

    ref_sum = int(tbl.sum())
    ref_sel = tbl[SELECT * TILE : (SELECT + 1) * TILE]
    sum_ok = dev_sum == ref_sum
    sel_ok = np.array_equal(dev_sel, ref_sel)
    print(
        f"reduce (consumer A): dev_sum={dev_sum} ref={ref_sum} {'OK' if sum_ok else 'DIFF'}"
    )
    print(
        f"select (consumer B): tile {SELECT} {'OK' if sel_ok else 'DIFF'} "
        f"(first dev={dev_sel[0]} ref={ref_sel[0]})"
    )
    ok = sum_ok and sel_ok
    print(f"\nbcast_probe: {'PASS' if ok else 'FAIL'}  (one stream -> two consumers)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
