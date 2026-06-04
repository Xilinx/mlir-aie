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

N_OUTER = 4
N_INNER = 7
LINE_LEN = 8
W_ACQ = 2
X_ACQ = 3
W_LINES = N_OUTER + (W_ACQ - 1)
X_LINES = (N_INNER + (X_ACQ - 1)) * N_OUTER
OUT_LINES = N_OUTER * N_INNER
W_LEN = W_LINES * LINE_LEN
X_LEN = X_LINES * LINE_LEN
OUT_LEN = OUT_LINES * LINE_LEN


def main(opts):
    rng = np.random.default_rng(0)
    w = rng.integers(-8, 8, size=(W_LEN,), dtype=np.int8)
    x = rng.integers(-8, 8, size=(X_LEN,), dtype=np.int8)

    # Reference: for each outer o in 0..N_OUTER-1, X slides over a fresh
    # (N_INNER + X_ACQ - 1) chunk; W slides by 1 per outer iter.
    w2d = w.reshape(W_LINES, LINE_LEN).astype(np.int32)
    x2d = x.reshape(N_OUTER, N_INNER + (X_ACQ - 1), LINE_LEN).astype(np.int32)
    ref = np.empty((N_OUTER, N_INNER, LINE_LEN), dtype=np.int32)
    for o in range(N_OUTER):
        for i in range(N_INNER):
            ref[o, i] = w2d[o] + w2d[o + 1] + x2d[o, i] + x2d[o, i + 1] + x2d[o, i + 2]
    ref = ref.astype(np.int8).reshape(-1)

    inW = iron.tensor(w, dtype=np.int8)
    inX = iron.tensor(x, dtype=np.int8)
    out = iron.zeros(OUT_LEN, dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    if not DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [inW, inX, out],
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
