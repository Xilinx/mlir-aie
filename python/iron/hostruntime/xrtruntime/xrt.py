# xrt.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
import numpy as np
import copy
import time
from pathlib import Path
import pyxrt as xrt
import os
from .tensor import XRTTensor
from ..hostruntime import DEFAULT_IRON_RUNTIME


#
# AI Engine Application class
#
# This class configures and invokes the XRT components needed to run an AIE
# Application. This includes xrt.device, xrt.kernel, xrt.hw_context and XRTTensors.
# You can use this class to simplify and reduce the amount of code needed to
# set up an AIE application.
#
class AIE_Application:
    # Registers xclbin to set up the device, hw context and kernel. This
    # also sets up the instruction stream
    def __init__(self, xclbin_path, insts_path, kernel_name="PP_FD_PRE"):
        self.device = None
        self.kernel = None
        self.buffers = [None] * 8
        self.device = xrt.device(0)

        # Find kernel by name in the xclbin
        self.xclbin = xrt.xclbin(xclbin_path)
        kernels = self.xclbin.get_kernels()
        try:
            xkernel = [k for k in kernels if kernel_name == k.get_name()][0]
        except KeyError:
            raise AIE_Application_Error("No such kernel: " + kernel_name)
        self.device.register_xclbin(self.xclbin)
        self.context = xrt.hw_context(self.device, self.xclbin.get_uuid())
        self.kernel = xrt.kernel(self.context, xkernel.get_name())

        ## Set up instruction stream
        insts = DEFAULT_IRON_RUNTIME.read_insts(Path(insts_path))
        self.n_insts = len(insts)
        self.insts_buffer = XRTTensor(
            insts, insts.dtype, flags=xrt.bo.cacheable, group_id=1
        )

    # Registers an XRTTensor object given group_id
    def register_buffer(self, group_id, tensor):
        self.buffers[group_id] = tensor

    # This syncs the instruction buffer to the device and then invokes the
    # `call` function before wait for the call to complete
    def run(self):
        self.insts_buffer._sync_to_device()
        h = self.call()
        r = h.wait()
        if r != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise Exception(f"Kernel returned {r}")
        return r

    # Wrapper for xrt.kernel function passing in opcode and XRTTensor objects
    def call(self):
        opcode = 3
        h = self.kernel(
            opcode,
            self.insts_buffer.buffer_object(),
            self.n_insts,
            *[b.buffer_object() for b in self.buffers if b is not None],
        )
        return h

    def __del__(self):
        if hasattr(self, "kernel"):
            del self.kernel
            self.kernel = None
        if hasattr(self, "device"):
            del self.device
            self.device = None


class AIE_Application_Error(Exception):
    pass


# Sets up the AIE application with support for up to 2 input buffers, 1 output
# buffer, and an optional trace buffer. Under the hood, we call declare an
# AIE_Application object and register the buffers used given the buffer datatype
# and shapes.
def setup_aie(
    xclbin_path,
    insts_path,
    in0,
    in1,
    out,
    enable_trace=False,
    kernel_name="MLIR_AIE",
    trace_size=16384,
    verbosity=0,
    trace_after_output=False,
):
    app = AIE_Application(xclbin_path, insts_path, kernel_name)

    if in0:
        if verbosity >= 1:
            print(f"register 1st input to group_id 3: {in0}")
        app.register_buffer(3, in0)
    if in1:
        if verbosity >= 1:
            print(f"register 2nd input to group_id 4: {in1}")
        app.register_buffer(4, in1)

    if enable_trace and trace_after_output:
        # Create a new, extended out tensor.
        out_size = trace_size
        if out:
            out_size += out.nbytes
        out = XRTTensor(out_size, dtype=np.uint8)
    if not out:
        # TODO out always needed so register buf 7 succeeds (not needed in C/C++ host code)
        out = XRTTensor((1,), dtype=np.uint32)

    if verbosity >= 1:
        print(
            f"register output to group_id 5: size: {out_buf_shape}, dtype: {out_buf_dtype}"
        )
    app.register_buffer(5, out)

    if enable_trace and not trace_after_output:
        trace_buff = XRTTensor((trace_size,), dtype=np.uint8)

        if verbosity >= 1:
            # print("register placeholder buffer (32b) to group_id 6")
            # print("register 2x 32b words for ctrl packets to group_id 6")
            print("register for ctrl packets to group_id 6: size: 8, dtype: uint32")
        app.register_buffer(
            # 6, shape=(1,), dtype=np.uint32
            # 6, shape=(2,), dtype=np.uint32
            6,
            XRTTensor((8,), dtype=np.uint32),
        )  # TODO Needed so register buf 7 succeeds (not needed in C/C++ host code)

        if verbosity >= 1:
            print(f"register trace on 7: {trace_buff}")
        app.register_buffer(7, trace_buff)

    return app


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
    app,
    input_one=None,
    input_two=None,
    output=None,
    trace_size=0,
    trace_after_output=False,
    enable_ctrl_pkts=False,
):
    trace_buff = None
    ctrl_buff = None
    if not (input_two is None):
        out_buff = app.buffers[5].numpy()
    else:
        out_buff = app.buffers[4].numpy()

    if trace_size:
        if trace_after_output:
            out_buff, trace_buff = extract_prefix(out_buf, output.shape, output.dtype)
        else:
            trace_buff = app.buffers[7].numpy()

        if enable_ctrl_pkts:
            trace_buff, ctrl_buff = extract_prefix(trace_buff, trace_size, np.uint8)
        trace_buff = trace_buff.view(np.uint32).reshape(trace_size // 4)

    return out_buff, trace_buff, ctrl_buff


# Wrapper for execute but we do the host time delta directly around the app.run() call
# so buffer init and read are not included
def execute_timed(
    app,
):
    start = time.time_ns()
    ret = app.run()
    stop = time.time_ns()
    npu_time = stop - start
    return (ret, npu_time)


# Wrapper function to separate output data and trace data from a single output buffer stream
def extract_prefix(out_buf, prefix_shape, prefix_dtype):
    out_buf_flat = out_buf.reshape((-1,)).view(np.uint8)
    prefix_bytes = np.prod(prefix_shape) * prefix.dtype.itemsize
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
    if opts.verbosity >= 1:
        print("trace size = ", str(opts.trace_size))
        print("enable_trace = ", str(enable_trace))

    app = setup_aie(
        opts.xclbin,
        opts.instr,
        in1,
        in2,
        out,
        enable_trace=enable_trace,
        trace_size=opts.trace_size,
        verbosity=opts.verbosity,
        trace_after_output=trace_after_output,
    )

    (ret, npu_time) = execute_timed(
        app,
    )

    print("npu_time: ", npu_time / 1000.0, " us")
    output, trace_buffer, ctrl_buffer = return_buffer_results(
        app,
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
        print("\nPASS!\n")
        return 0
    else:
        print("\nError count: ", errors)
        print("\nFailed.\n")
        return 1
