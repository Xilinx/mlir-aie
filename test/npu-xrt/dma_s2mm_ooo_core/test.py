# ===- test.py -------------------------------------------------*- Python -*-===#
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===#

# Launched from run.lit; disable standalone discovery.
# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS

# Host runner for the compute-core out-of-order merge designs. A compute core is
# the out-of-order S2MM consumer (all prior OoO merges consumed on a memtile).
# The merge places each packet by its header out-of-order id (id (k + shift) % n
# for source k, shift=1), so the drained buffer is the source order rotated by
# one -- a non-identity permutation an in-order channel could not produce. The
# "backpressure" design runs two generations through one reused buffer with a
# throttled (delay-loop) consumer; a correct result there depends on the
# ooo_prod credit holding back the receive side until the consumer drains.

import argparse
import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime
from aie.utils.hostruntime.argparse import add_runtime_args

N = 2  # sources merged per generation (= receive slots)
TW = 16  # int32 words per packet
SHIFT = 1  # id (k + SHIFT) % N -> a non-identity rotation


def _data_and_ref(design):
    m = 1 if design == "release_only" else 2  # generations
    size = m * N * TW
    a = np.arange(size, dtype=np.int32)
    # generation g, source k holds a[(g*N + k)*TW : +TW] and is placed into slot
    # (k + SHIFT) % N by its header id.
    ab = a.reshape(m, N, TW)
    ref = np.empty_like(ab)
    for g in range(m):
        for k in range(N):
            ref[g][(k + SHIFT) % N] = ab[g][k]
    return size, ref.reshape(-1)


def main(opts):
    size, ref = _data_and_ref(opts.design)
    a_in = iron.arange(size, dtype=np.int32)
    c_out = iron.zeros((size,), dtype=np.int32)

    npu_opts = test_utils.create_npu_kernel(opts)
    rc = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [a_in, c_out],
        {1: ref},  # verify c_out (arg 1) == the header-id permutation
        verify=npu_opts.verify,
        verbosity=npu_opts.verbosity,
    )
    if rc == 0:
        print("PASS!")
        return 0
    print("FAILED")
    return 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_runtime_args(p)
    p.add_argument(
        "--design",
        choices=("release_only", "backpressure"),
        required=True,
        help="which core-merge design's data/reference to use",
    )
    opts = p.parse_args(sys.argv[1:])
    sys.exit(main(opts))
