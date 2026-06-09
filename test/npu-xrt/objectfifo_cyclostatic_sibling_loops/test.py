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

N_LINES = 6
LINE_LEN = 8
ACQ = 3
CARRY = ACQ - 1
IN_LINES = 2 * (N_LINES + CARRY)
OUT_LINES = 2 * N_LINES
IN_LEN = IN_LINES * LINE_LEN
OUT_LEN = OUT_LINES * LINE_LEN


def main(opts):
    rng = np.random.default_rng(0)
    src = rng.integers(-8, 8, size=(IN_LEN,), dtype=np.int8)

    src2d = src.reshape(IN_LINES, LINE_LEN).astype(np.int32)
    ref = np.empty((OUT_LINES, LINE_LEN), dtype=np.int32)
    for half in range(2):
        base_in = half * (N_LINES + CARRY)
        base_out = half * N_LINES
        for i in range(N_LINES):
            ref[base_out + i] = (
                src2d[base_in + i] + src2d[base_in + i + 1] + src2d[base_in + i + 2]
            )
    ref = ref.astype(np.int8).reshape(-1)

    inA = iron.tensor(src, dtype=np.int8)
    out = iron.zeros(OUT_LEN, dtype=np.int8)

    print("Running...\n")
    npu_opts = test_utils.create_npu_kernel(opts)
    res = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [inA, out],
        {1: ref},
        verify=npu_opts.verify,
        verbosity=npu_opts.verbosity,
    )
    if res == 0:
        print("\nPASS!\n")
    sys.exit(res)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
