# xrt.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
import numpy as np
from pathlib import Path
from .tensor import XRTTensor
from .. import DEFAULT_IRON_RUNTIME
from ..hostruntime import HostRuntime, TraceConfig


def extract_tile(data):
    col = (data >> 21) & 0x7F
    row = (data >> 16) & 0x1F
    pkt_type = (data >> 12) & 0x3
    pkt_id = data & 0x1F
    return (col, row, pkt_type, pkt_id)


# Wrapper function to write trace buffer values to a text file
def write_out_trace(trace, file_name):
    out_str = "\n".join(f"{i:0{8}x}" for i in trace if i != 0)
    with open(file_name, "w") as f:
        f.write(out_str)


# This wrapper function abstracts the full set of functions to setup the aie and run
# the kernel program including check for functional correctness and reporting the
# run time. Under the hood, we call `setup_aie` to set up the AIE application before
# calling `execute` and checking results. The tensors for the inputs
# and output buffers are passed in as arguments, along with the gold reference data
# to compare it against. Trace buffers is also written out to a text file if trace is
# enabled.
def setup_and_run_aie(
    inputs,
    outputs,
    ref,
    opts,
    trace_after_output=False,
    enable_ctrl_pkts=False,
):
    kernel_handle = DEFAULT_IRON_RUNTIME.load(Path(opts.xclbin), Path(opts.instr))

    # Ensure inputs and outputs are lists
    if not isinstance(inputs, list):
        inputs = [inputs] if inputs else []
    if not isinstance(outputs, list):
        outputs = [outputs] if outputs else []

    buffers = inputs + outputs

    trace_config = None
    last_out = outputs[-1] if outputs else None

    if opts.trace_size > 0:
        trace_config = TraceConfig(
            trace_size=opts.trace_size,
            trace_after_last_tensor=trace_after_output,
            enable_ctrl_pkts=enable_ctrl_pkts,
            last_tensor_shape=last_out.shape if last_out else None,
            last_tensor_dtype=last_out.dtype if last_out else None,
        )
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

    output_data = [b.numpy() for b in buffers[len(inputs) : len(inputs) + len(outputs)]]

    if trace_config:
        if opts.verbosity >= 1:
            print("trace_buffer shape: ", trace_buffer.shape)
            print("trace_buffer dtype: ", trace_buffer.dtype)
        write_out_trace(trace_buffer, str(opts.trace_file))

        if enable_ctrl_pkts:
            if opts.verbosity >= 1:
                print("ctrl_buffer shape: ", ctrl_buffer.shape)
                print("ctrl_buffer dtype: ", ctrl_buffer.dtype)
                print("ctrl buffer: ", [hex(d) for d in ctrl_buffer])
            for i in range(ctrl_buffer.size // 2):
                col, row, pkt_type, pkt_id = extract_tile(ctrl_buffer[i * 2])
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

        if len(ref) != len(output_data):
            print(
                f"Error: Number of reference outputs ({len(ref)}) does not match number of actual outputs ({len(output_data)})"
            )
            return 1

        for r, o in zip(ref, output_data):
            e = np.equal(r, o)
            errors += np.size(e) - np.count_nonzero(e)

    if not errors:
        return 0
    else:
        if opts.verbosity >= 1:
            print("\nError count: ", errors)
            print("\nFailed.\n")
        return 1
