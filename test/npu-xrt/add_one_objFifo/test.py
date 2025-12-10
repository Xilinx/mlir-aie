# ===- test.py -------------------------------------------------*- Python -*-===#
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
#
# ===----------------------------------------------------------------------===#

# This gets launched from run.lit, so disable it with a bogus requires line
# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS
import sys
import numpy as np

import aie.utils.test as test_utils
import aie.iron.hostruntime.xrtruntime.xrt as xrt_utils
import aie.iron as iron

IN_SIZE = 64
OUT_SIZE = 64


def main(opts):
    ref_data = np.arange(1, IN_SIZE + 1, dtype=np.uint32)
    inA = iron.tensor(ref_data, dtype=np.uint32)
    inB = iron.tensor(ref_data, dtype=np.uint32)
    out = iron.zeros((OUT_SIZE,), dtype=np.uint32)
    ref_data = ref_data + 42

    xrt_utils.setup_and_run_aie(
        inA,
        inB,
        out,
        ref_data,
        opts,
    )


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    sys.exit(main(opts))
