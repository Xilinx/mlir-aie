# ===- test.py -------------------------------------------------*- Python -*-===#
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===#

# This gets launched from run.lit, so disable it with a bogus requires line
# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS
import argparse
import sys
import numpy as np

import aie.utils.test as test_utils
import aie.iron as iron
from aie.utils import DefaultNPURuntime
from aie.utils.hostruntime.argparse import add_runtime_args

SIZE = 128


def main(opts):
    inA = iron.arange(1, SIZE + 1, dtype=np.int32)
    out = iron.zeros((SIZE,), dtype=np.int32)
    ref = inA.numpy() + 41

    npu_opts = test_utils.create_npu_kernel(opts)
    errors = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [inA, out],
        {1: ref},
        verify=npu_opts.verify,
        verbosity=npu_opts.verbosity,
    )
    if errors:
        print("Failed.")
        return 1
    print("PASS!")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_runtime_args(p)
    opts = p.parse_args(sys.argv[1:])
    sys.exit(main(opts))
