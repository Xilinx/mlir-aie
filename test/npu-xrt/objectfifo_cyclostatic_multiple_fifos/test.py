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

N_LINES = 10
LINE_LEN = 8
X_ACQ = 3
Y_ACQ = 2
X_IN_LINES = N_LINES + (X_ACQ - 1)
Y_IN_LINES = N_LINES + (Y_ACQ - 1)
X_LEN = X_IN_LINES * LINE_LEN
Y_LEN = Y_IN_LINES * LINE_LEN
OUT_LEN = N_LINES * LINE_LEN


def main(opts):
    rng = np.random.default_rng(0)
    x = rng.integers(-8, 8, size=(X_LEN,), dtype=np.int8)
    y = rng.integers(-8, 8, size=(Y_LEN,), dtype=np.int8)

    x2d = x.reshape(X_IN_LINES, LINE_LEN).astype(np.int32)
    y2d = y.reshape(Y_IN_LINES, LINE_LEN).astype(np.int32)
    ref = np.empty((N_LINES, LINE_LEN), dtype=np.int32)
    for i in range(N_LINES):
        ref[i] = x2d[i] + x2d[i + 1] + x2d[i + 2] + y2d[i] + y2d[i + 1]
    ref = ref.astype(np.int8).reshape(-1)

    inX = iron.tensor(x, dtype=np.int8)
    inY = iron.tensor(y, dtype=np.int8)
    out = iron.zeros(OUT_LEN, dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    if not DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [inX, inY, out],
        {2: ref},
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
