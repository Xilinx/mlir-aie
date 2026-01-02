# xrt.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
import argparse
import numpy as np
from pathlib import Path
from .tensor import XRTTensor
from .. import DEFAULT_IRON_RUNTIME
from ..hostruntime import HostRuntime, TraceConfig


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

    kernel_handle = DEFAULT_IRON_RUNTIME.load(Path(opts.xclbin), Path(opts.instr))

    # Ensure io_args is a list
    if not isinstance(io_args, list):
        io_args = [io_args] if io_args else []

    buffers = io_args

    trace_config = opts.trace_config
    last_out = buffers[-1] if buffers else None

    if trace_config:
        trace_config.last_tensor_shape = last_out.shape if last_out else None
        trace_config.last_tensor_dtype = last_out.dtype if last_out else None

        HostRuntime.prepare_args_for_trace(buffers, trace_config)

    # [b.to("npu") for b in buffers]
    ret = DEFAULT_IRON_RUNTIME.run(kernel_handle, buffers)
    # [b.to("cpu") for b in buffers]

    if opts.verbosity >= 1:
        print("npu_time: ", ret.npu_time / 1000.0, " us")

    trace_buffer = None
    ctrl_buffer = None
    if trace_config:
        trace_buffer, ctrl_buffer = HostRuntime.extract_trace_from_args(
            buffers, trace_config
        )

    if trace_config:
        if opts.verbosity >= 1:
            print("trace_buffer shape: ", trace_buffer.shape)
            print("trace_buffer dtype: ", trace_buffer.dtype)
        trace_config.write_trace(trace_buffer)

        if trace_config.enable_ctrl_pkts:
            if opts.verbosity >= 1:
                print("ctrl_buffer shape: ", ctrl_buffer.shape)
                print("ctrl_buffer dtype: ", ctrl_buffer.dtype)
                print("ctrl buffer: ", [hex(d) for d in ctrl_buffer])
            for i in range(ctrl_buffer.size // 2):
                col, row, pkt_type, pkt_id = TraceConfig.extract_tile(
                    ctrl_buffer[i * 2]
                )
                overflow = True if (ctrl_buffer[i * 2 + 1] >> 8) == 3 else False
                if overflow:
                    print(
                        f"WARNING: Trace overflow detected in tile({row},{col}). Trace results may be invalid."
                    )

    # Copy output results and verify they are correct
    errors = 0
    if opts.verify:
        if opts.verbosity >= 1:
            print("Verifying results ...")

        # Handle ref being list or single
        if not isinstance(ref, list):
            ref = [ref]

        for item in ref:
            if isinstance(item, tuple) and len(item) == 2:
                idx, r = item
                if idx >= len(io_args):
                    print(
                        f"Error: Reference index {idx} out of bounds for {len(io_args)} IO buffers"
                    )
                    return 1
                io_args[idx].to("cpu")
                o = io_args[idx].numpy()
                e = np.equal(r, o)
                errors += np.size(e) - np.count_nonzero(e)
            else:
                print("Error: Reference data must be a list of (index, data) tuples")
                return 1

    if not errors:
        return 0
    else:
        if opts.verbosity >= 1:
            print("\nError count: ", errors)
            print("\nFailed.\n")
        return 1
