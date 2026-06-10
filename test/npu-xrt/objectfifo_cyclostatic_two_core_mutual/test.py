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

N_LINES = 12
LINE_LEN = 8
WINDOW = 3
SEED_LINES = WINDOW


def simulate(seed_a, seed_b):
    # Mirror of the device-side execution. Each core's "input queue" is the
    # peer's output queue. Seed phase: A's output queue starts with seed_a,
    # B's with seed_b. Cyclostatic phase: each core consumes its 3-line
    # window from peer's queue, emits sum, appends sum to its own queue.
    a_to_b = list(seed_a.reshape(SEED_LINES, LINE_LEN).astype(np.int32))
    b_to_a = list(seed_b.reshape(SEED_LINES, LINE_LEN).astype(np.int32))

    a_out = np.zeros((N_LINES, LINE_LEN), dtype=np.int32)
    b_out = np.zeros((N_LINES, LINE_LEN), dtype=np.int32)

    for i in range(N_LINES):
        a_win = b_to_a[i : i + WINDOW]
        b_win = a_to_b[i : i + WINDOW]
        a_sum = a_win[0] + a_win[1] + a_win[2]
        b_sum = b_win[0] + b_win[1] + b_win[2]
        a_out[i] = a_sum
        b_out[i] = b_sum
        a_to_b.append(a_sum)
        b_to_a.append(b_sum)

    a_out_i8 = a_out.astype(np.int8).reshape(-1)
    b_out_i8 = b_out.astype(np.int8).reshape(-1)
    return np.concatenate([a_out_i8, b_out_i8])


def main(opts):
    rng = np.random.default_rng(0)
    seed_a = rng.integers(-4, 4, size=(SEED_LINES * LINE_LEN,), dtype=np.int8)
    seed_b = rng.integers(-4, 4, size=(SEED_LINES * LINE_LEN,), dtype=np.int8)
    ref = simulate(seed_a, seed_b)

    in_a = iron.tensor(seed_a, dtype=np.int8)
    in_b = iron.tensor(seed_b, dtype=np.int8)
    out = iron.zeros(2 * N_LINES * LINE_LEN, dtype=np.int8)

    print("Running...\n")
    npu_opts = test_utils.create_npu_kernel(opts)
    res = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [in_a, in_b, out],
        {2: ref},
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
