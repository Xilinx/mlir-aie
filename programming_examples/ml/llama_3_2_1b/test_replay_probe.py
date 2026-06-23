"""Test the replay+chunk probe: every replayed chunk == the resident HALF."""

from __future__ import annotations

import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_replay_probe import HALF, CHUNK, R, N_CHUNKS, HALF_DATA


def main():
    opts = test_utils.create_default_argparser().parse_args()
    out_t = iron.zeros([N_CHUNKS * R * CHUNK], dtype=np.int8)
    npu = test_utils.create_npu_kernel(opts).npu_kernel
    rc = DefaultNPURuntime.run_test(npu, [out_t], {}, verify=False, verbosity=opts.verbosity)
    if rc != 0:
        print(f"dispatch returned {rc}", file=sys.stderr)
        return rc
    out_t.to("cpu")
    out = out_t.numpy()
    # Expected: R copies of HALF_DATA, each as N_CHUNKS chunks.
    expected = np.tile(HALF_DATA, R)
    fails = 0
    for p in range(R):
        seg = out[p * HALF : (p + 1) * HALF]
        ok = np.array_equal(seg, HALF_DATA)
        fails += not ok
        bad = int((seg != HALF_DATA).sum())
        print(f"pass {p}: {'OK' if ok else f'MISMATCH ({bad}/{HALF})'}")
    print(f"\nreplay_probe: {'PASS' if fails == 0 else f'{fails} FAIL'}  (HALF={HALF} CHUNK={CHUNK} R={R})")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
