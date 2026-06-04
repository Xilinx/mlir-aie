# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# (c) Copyright 2026 AMD Inc.
#
# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS
import sys
import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

N_LINES = 14
LINE_LEN = 8
FIFO_DEPTH = 3
IN_LEN = (N_LINES + FIFO_DEPTH - 1) * LINE_LEN
OUT_LEN = N_LINES * LINE_LEN


def main(opts):
    rng = np.random.default_rng(0)
    src = rng.integers(-32, 32, size=(IN_LEN,), dtype=np.int8)

    # Expected: per output line i (i in 0..N_LINES-1),
    #   out[i, b] = src[i, b] + src[i+1, b] + src[i+2, b]  (saturating in int8)
    src2d = src.reshape(N_LINES + FIFO_DEPTH - 1, LINE_LEN).astype(np.int32)
    ref = (src2d[:N_LINES] + src2d[1 : N_LINES + 1] + src2d[2 : N_LINES + 2]).astype(
        np.int8
    )
    ref = ref.reshape(-1)

    inA = iron.tensor(src, dtype=np.int8)
    out = iron.zeros(OUT_LEN, dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    if not DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [inA, out],
        {1: ref},
        verify=npu_opts.verify,
        verbosity=npu_opts.verbosity,
    ):
        print("PASS!")
    else:
        print("Failed.")


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    sys.exit(main(opts))
