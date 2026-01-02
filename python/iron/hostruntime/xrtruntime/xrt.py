# xrt.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
from pathlib import Path
from .. import DEFAULT_IRON_RUNTIME


# This wrapper function abstracts the full set of functions to setup the aie and run
# the kernel program including check for functional correctness and reporting the
# run time. Under the hood, we call `setup_aie` to set up the AIE application before
# calling `execute` and checking results. The tensors for the inputs
# and output buffers are passed in as arguments, along with the gold reference data
# to compare it against. Trace buffers is also written out to a text file if trace is
# enabled.
def setup_and_run_aie(
    io_args,
    ref,
    opts=None,
):
    if opts is None:
        from aie.utils import test as test_utils

        opts = test_utils.parse_args(None)

    return DEFAULT_IRON_RUNTIME.run_test(
        io_args,
        ref,
        Path(opts.xclbin),
        Path(opts.instr),
        trace_config=opts.trace_config,
        verify=opts.verify,
        verbosity=opts.verbosity,
    )
