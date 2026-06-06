#!/usr/bin/env python3
# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Python test harness for the event-trace vector_scalar_mul design.

Loads the pre-compiled xclbin/insts produced by the Makefile's
``jit_xclbin`` rule and runs it on the attached NPU via the shared
``DefaultNPURuntime`` + ``NPUKernel`` helpers (the same path as
sibling ``test.py`` files like ``vector_scalar_mul/test.py``).
"""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.utils import DefaultNPURuntime
from aie.utils.hostruntime.argparse import add_runtime_args
from aie.utils.test import create_npu_kernel


def main(opts):
    tensor_elems = 4096
    scale_factor = 3
    dtype = np.int32

    rng = np.random.default_rng(seed=42)
    in1_np = rng.integers(1, 100, size=tensor_elems, dtype=dtype)
    in1 = iron.tensor(in1_np, dtype=dtype, device="npu")
    in2 = iron.full((1,), scale_factor, dtype=dtype)
    out = iron.zeros(tensor_elems, dtype=dtype)
    ref = in1_np * scale_factor

    npu_opts = create_npu_kernel(opts)
    if npu_opts.npu_kernel.trace_config:
        npu_opts.npu_kernel.trace_config.enable_ctrl_pkts = True

    print("Running...\n")
    res = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [in1, in2, out],
        {2: ref},
        verify=npu_opts.verify,
        verbosity=npu_opts.verbosity,
    )
    if res == 0:
        print("\nPASS!\n")
    sys.exit(res)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_runtime_args(p)
    opts = p.parse_args(sys.argv[1:])
    main(opts)
