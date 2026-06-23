"""Test the full logits relay (GEMM-fill + replay): every replayed chunk == in."""

from __future__ import annotations

import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_fillreplay_probe import HALF, CHUNK, R, N_CHUNKS


def main():
    opts = test_utils.create_default_argparser().parse_args()
    rng = np.random.default_rng(0)
    src = rng.uniform(-10, 10, size=HALF).astype(np.float32)

    in_t = iron.tensor(src, dtype=np.float32)
    out_t = iron.zeros([N_CHUNKS * R * CHUNK], dtype=np.float32)
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
        seg = out[p * HALF : (p + 1) * HALF]
        ok = np.array_equal(seg, src)
        fails += not ok
        bad = int((seg != src).sum())
        print(f"pass {p}: {'OK' if ok else f'MISMATCH ({bad}/{HALF})'}")
    print(
        f"\nfillreplay_probe: {'PASS' if fails == 0 else f'{fails} FAIL'}  (HALF={HALF} CHUNK={CHUNK} R={R})"
    )
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
