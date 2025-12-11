# xrt.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
import numpy as np
from .tensor import XRTTensor
from .. import DEFAULT_IRON_RUNTIME


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


def return_buffer_results(
    input_one=None,
    input_two=None,
    output=None,
    trace_size=0,
    trace_after_output=False,
    enable_ctrl_pkts=False,
):
    trace_buff = None
    ctrl_buff = None
    if trace_size:
        if trace_after_output:
            out_buff, trace_buff = extract_prefix(out_buff, output.shape, output.dtype)
        else:
            trace_buff = app.buffers[-1].numpy()

        if enable_ctrl_pkts:
            trace_buff, ctrl_buff = extract_prefix(trace_buff, trace_size, np.uint8)
        trace_buff = trace_buff.view(np.uint32).reshape(trace_size // 4)

    return out_buff, trace_buff, ctrl_buff


# Wrapper function to separate output data and trace data from a single output buffer stream
def extract_prefix(out_buf, prefix_shape, prefix_dtype):
    out_buf_flat = out_buf.reshape((-1,)).view(np.uint8)
    prefix_bytes = np.prod(prefix_shape) * prefix_dtype.itemsize
    output_prefix = out_buf_flat[:prefix_bytes].view(prefix_dtype).reshape(prefix_shape)
    output_suffix = out_buf_flat[-prefix_bytes:]
    return output_prefix, output_suffix


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
    enable_trace = opts.trace_size > 0
    kernel_handle = DEFAULT_IRON_RUNTIME.load(opts.xclbin_path, opts.insts_path)

    buffers = []
    if in1:
        buffers.append(in1)
    if in2:
        buffers.append(in2)

    if enable_trace and trace_after_output:
        # Create a new, extended out tensor.
        out_size = opts.trace_size
        if out:
            out_size += out.nbytes
        out = XRTTensor(out_size, dtype=np.uint8)
        buffers.append(out)
    elif not out:
        # TODO out always needed so register buf 7 succeeds (not needed in C/C++ host code)
        out = XRTTensor((1,), dtype=np.uint32)
        buffers.append(out)
    else:
        buffers.append(out)

    if enable_trace and not trace_after_output:
        # This is a dummy buffer
        buffers.append(
            XRTTensor((8,), dtype=np.uint32),
        )  # TODO Needed so register buf 7 succeeds (not needed in C/C++ host code)

        trace_buff = XRTTensor((opts.trace_size,), dtype=np.uint8)
        buffers.append(trace_buff)

    ret = DEFAULT_IRON_RUNTIME.run(kernel_handle, buffers)

    if opts.verbosity >= 1:
        print("npu_time: ", ret.npu_time / 1000.0, " us")
    output, trace_buffer, ctrl_buffer = return_buffer_results(
        in1,
        in2,
        out,
        trace_size=opts.trace_size,
        trace_after_output=trace_after_output,
        enable_ctrl_pkts=enable_ctrl_pkts,
    )

    if enable_trace:
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
