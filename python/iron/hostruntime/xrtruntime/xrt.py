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


# checks # of bits. Odd number returns a 1. Even returns 0.
def parity(x):
    return x.bit_count() & 1


# create control packet
def create_ctrl_pkt(
    operation,
    beats,
    addr,
    ctrl_pkt_read_id=28,  # global id used for all ctrl packet reads
    # WARNING: this needs to match the packet id used in packetflow/.py
):
    header = (ctrl_pkt_read_id << 24) | (operation << 22) | (beats << 20) | addr
    header |= (0x1 ^ parity(header)) << 31
    return header


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
# calling `execute` and checking results. The datatypes and shape for the 2 inputs
# and 1 output buffers are passed in as arguments, along with the gold reference data
# to compare it against. Trace buffers is also written out to a text file if trace is
# enabled.
def setup_and_run_aie(
    in1,
    in2,
    out,
    ref,
    opts,
    trace_after_output=False,
    enable_ctrl_pkts=False,
):
    kernel_handle = DEFAULT_IRON_RUNTIME.load(Path(opts.xclbin), Path(opts.instr))

    buffers = []
    if in1:
        buffers.append(in1)
    if in2:
        buffers.append(in2)
    if out:
        buffers.append(out)

    trace_config = None
    if opts.trace_size > 0:
        trace_config = TraceConfig(
            trace_size=opts.trace_size,
            trace_after_last_tensor=trace_after_output,
            enable_ctrl_pkts=enable_ctrl_pkts,
            last_tensor_shape=out.shape if out else None,
            last_tensor_dtype=out.dtype if out else None,
        )
        HostRuntime.prepare_args_for_trace(buffers, trace_config)

    ret = DEFAULT_IRON_RUNTIME.run(kernel_handle, buffers)

    if opts.verbosity >= 1:
        print("npu_time: ", ret.npu_time / 1000.0, " us")

    trace_buffer = None
    ctrl_buffer = None
    if trace_config:
        trace_buffer, ctrl_buffer = HostRuntime.extract_trace_from_args(
            buffers, trace_config
        )

    output = out.numpy() if out else None

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
        e = np.equal(ref, output)
        errors = np.size(e) - np.count_nonzero(e)

    if not errors:
        return 0
    else:
        if opts.verbosity >= 1:
            print("\nError count: ", errors)
            print("\nFailed.\n")
        return 1
