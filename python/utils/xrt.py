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
import pyxrt as xrt
import os


#
# AI Engine Application class
#
# This class configures and invokes the XRT components needed to run an AIE
# Application. This includes xrt.device, xrt.kernel, xrt.hw_context and XRT
# buffers as enacpuslated by the AIE_Buffer class. You can use this class to
# simplify and reduce the amount of code needed to set up an AIE application.
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
        insts = read_insts(insts_path)
        self.n_insts = len(insts)
        self.insts_buffer = AIE_Buffer(
            self, 1, insts.dtype, insts.shape, xrt.bo.cacheable
        )
        self.insts_buffer.write(insts)

    # Registers an AIE_Buffer class object given group_id, datatype and shape
    def register_buffer(self, group_id, *args, **kwargs):
        self.buffers[group_id] = AIE_Buffer(self, group_id, *args, **kwargs)

    # This syncs the instruction buffer to the device and then invokes the
    # `call` function before wait for the call to complete
    def run(self):
        self.insts_buffer.sync_to_device()
        h = self.call()
        r = h.wait()
        if r != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise Exception(f"Kernel returned {r}")

    # Wrapper for xrt.kernel function passing in opcode and buffers objects
    # class `AIE_Buffer`
    def call(self):
        opcode = 3
        h = self.kernel(
            opcode,
            self.insts_buffer.bo,
            self.n_insts,
            *[b.bo for b in self.buffers if b is not None],
        )
        return h

    def __del__(self):
        if hasattr(self, "kernel"):
            del self.kernel
            self.kernel = None
        if hasattr(self, "device"):
            del self.device
            self.device = None


# This class wraps up access to the xrt.bo buffer object where sync calls are added
# to read and write calls to ensure data is synchronized.
class AIE_Buffer:

    # Declare xrt.bo object given group_id, datatype, shape
    def __init__(self, application, group_id, dtype, shape, flags=xrt.bo.host_only):
        self.application = application
        self.dtype = dtype
        self.shape = shape
        self.len_bytes = np.prod(shape) * np.dtype(dtype).itemsize
        self.bo = xrt.bo(
            application.device,
            self.len_bytes,
            flags,
            application.kernel.group_id(group_id),
        )

    # Synchronize data from device before reading xrt.bo data
    def read(self):
        self.sync_from_device()
        return self.bo.read(self.len_bytes, 0).view(self.dtype).reshape(self.shape)

    # Write data to xrt.bo and synchronize data to device
    def write(self, v, offset=0):
        self.bo.write(v.view(np.uint8), offset)
        self.sync_to_device()

    # Wrapper for xrt.bo.sync call (to device)
    def sync_to_device(self):
        return self.bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # Wrapper for xrt.bo.sync call (from device)
    def sync_from_device(self):
        return self.bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    def __del__(self):
        if hasattr(self, "bo"):
            del self.bo
            self.bo = None


class AIE_Application_Error(Exception):
    pass


insts_cache = {}


# Read instruction stream from text file and reformat it to be passed into the
# instruction buffer for the xrt.kernel call
def read_insts_sequence(insts_path):
    """Reads instructions from a text file (hex numbers, one per line)."""
    global insts_cache
    if insts_path in insts_cache:
        # Speed up things if we re-configure the array a lot: Don't re-parse
        # the insts.txt each time
        return insts_cache[insts_path]
    with open(insts_path, "r") as f:
        insts_text = f.readlines()
        insts_text = [l for l in insts_text if l != ""]
        insts_v = np.array([int(c, 16) for c in insts_text], dtype=np.uint32)
        insts_cache[insts_path] = insts_v
    return insts_v


# Read instruction stream from bin file and reformat it to be passed into the
# instruction buffer for the xrt.kernel call
def read_insts_binary(insts_path):
    """Reads instructions from a binary file."""
    global insts_cache
    if insts_path in insts_cache:
        # Speed up things if we re-configure the array a lot: Don't re-parse
        # the insts.bin each time
        return insts_cache[insts_path]
    with open(insts_path, "rb") as f:
        data = f.read()
    # Interpret the binary data as an array of uint32 values.
    insts_v = np.frombuffer(data, dtype=np.uint32)
    insts_cache[insts_path] = insts_v
    return insts_v


def read_insts(insts_path):
    """
    Reads instructions from the given file.
    If the file extension is .bin, uses binary read.
    If the file extension is .txt, uses sequence (text) read.
    """
    _, ext = os.path.splitext(insts_path)
    ext = ext.lower()

    if ext == ".bin":
        return read_insts_binary(insts_path)
    elif ext == ".txt":
        return read_insts_sequence(insts_path)
    else:
        raise ValueError("Unsupported file extension: expected .bin or .txt")


# Sets up the AIE application with support for up to 2 input buffers, 1 output
# buffer, and an optional trace buffer. Under the hood, we call declare an
# AIE_Application object and register the buffers used given the buffer datatype
# and shapes.
def setup_aie(
    xclbin_path,
    insts_path,
    in_0_shape,
    in_0_dtype,
    in_1_shape,
    in_1_dtype,
    out_buf_shape,
    out_buf_dtype,
    enable_trace=False,
    kernel_name="MLIR_AIE",
    trace_size=16384,
    verbosity=0,
    trace_after_output=False,
):
    app = AIE_Application(xclbin_path, insts_path, kernel_name)

    if in_0_shape and in_0_dtype:
        if verbosity >= 1:
            print("register 1st input to group_id 3")
        app.register_buffer(3, shape=in_0_shape, dtype=in_0_dtype)
    if in_1_shape and in_1_dtype:
        if verbosity >= 1:
            print("register 2nd input to group_id 4")
        app.register_buffer(4, shape=in_1_shape, dtype=in_1_dtype)

    if enable_trace:
        if trace_after_output:
            out_buf_len_bytes = (
                np.prod(out_buf_shape) * np.dtype(out_buf_dtype).itemsize
            )
            out_buf_shape = (out_buf_len_bytes + trace_size,)
            out_buf_dtype = np.uint8

    if in_1_shape and in_1_dtype:
        if verbosity >= 1:
            print("register output to group_id 5")
        app.register_buffer(5, shape=out_buf_shape, dtype=out_buf_dtype)
    else:
        if verbosity >= 1:
            print("register output to group_id 4")
        app.register_buffer(4, shape=out_buf_shape, dtype=out_buf_dtype)
        if verbosity >= 1:
            print("register placeholder buffer (32b) to group_id 5")
        app.register_buffer(
            5, shape=(1,), dtype=np.uint32
        )  # TODO Needed so register buf 7 succeeds (not needed in C/C++ host code)

    if enable_trace:
        if not trace_after_output:
            trace_buf_shape = (
                trace_size * 4,
            )  # 4x as workaround to avoid driver corruption
            trace_buf_dtype = np.uint8
            if verbosity >= 1:
                print("register placeholder buffer (32b) to group_id 6")
            app.register_buffer(
                6, shape=(1,), dtype=np.uint32
            )  # TODO Needed so register buf 7 succeeds (not needed in C/C++ host code)
            if verbosity >= 1:
                print(
                    "register trace on 7: size: "
                    + str(trace_buf_shape)
                    + ", dtype:"
                    + str(trace_buf_dtype)
                )
            app.register_buffer(7, shape=trace_buf_shape, dtype=trace_buf_dtype)

    return app


# Wrapper function to write buffer arguments into registered input buffers, then call
# `run` function for AIE Application, and finally return the output buffer data.
def execute(
    app, input_one=None, input_two=None, enable_trace=False, trace_after_output=False
):
    if not (input_one is None):
        app.buffers[3].write(input_one)
    if not (input_two is None):
        app.buffers[4].write(input_two)

    app.run()

    if trace_after_output or not enable_trace:
        if not (input_two is None):
            # return app.buffers[5].read(), 0
            return app.buffers[5].read()
        else:
            # return app.buffers[4].read(), 0
            return app.buffers[4].read()
    else:
        if not (input_two is None):
            return app.buffers[5].read(), app.buffers[7].read()
        else:
            return app.buffers[4].read(), app.buffers[7].read()


# Wrapper function to separate output data and trace data from a single output buffer stream
def extract_trace(out_buf, out_buf_shape, out_buf_dtype, trace_size):
    trace_size_words = trace_size // 4
    out_buf_flat = out_buf.reshape((-1,)).view(np.uint32)
    output_prefix = (
        out_buf_flat[:-trace_size_words].view(out_buf_dtype).reshape(out_buf_shape)
    )
    trace_suffix = out_buf_flat[-trace_size_words:]
    return output_prefix, trace_suffix


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
    in1_dtype,
    in2_dtype,
    out_dtype,
    in1_data,
    in2_data,
    out_data,
    in1_volume,
    in2_volume,
    out_volume,
    ref,
    opts,
    trace_after_output=False,
):
    enable_trace = opts.trace_size > 0
    if opts.verbosity >= 1:
        print("trace size = ", str(opts.trace_size))
        print("enable_trace = ", str(enable_trace))

    app = setup_aie(
        opts.xclbin,
        opts.instr,
        in1_volume,
        in1_dtype,
        in2_volume,
        in2_dtype,
        out_volume,
        out_dtype,
        enable_trace=enable_trace,
        trace_size=opts.trace_size,
        verbosity=opts.verbosity,
        trace_after_output=trace_after_output,
    )

    out_size = out_volume * out_data.itemsize
    if opts.verbosity >= 1:
        print("out_size: " + str(out_size))

    start = time.time_ns()
    if enable_trace:
        full_output, trace_buffer = execute(
            app, in1_data, in2_data, enable_trace, trace_after_output
        )
    else:
        full_output = execute(app, in1_data, in2_data, enable_trace, trace_after_output)
    stop = time.time_ns()
    npu_time = stop - start
    print("npu_time: ", npu_time)

    aie_output = full_output[:out_size].view(out_dtype)
    if enable_trace:
        if trace_after_output:
            trace_buffer = full_output[out_size:].view(np.uint32)
        else:
            trace_buffer = trace_buffer.view(np.uint32)

    if enable_trace:
        if opts.verbosity >= 1:
            print("trace_buffer shape: ", trace_buffer.shape)
            print("trace_buffer dtype: ", trace_buffer.dtype)
        write_out_trace(trace_buffer, str(opts.trace_file))

    # Copy output results and verify they are correct
    errors = 0
    if opts.verify:
        if opts.verbosity >= 1:
            print("Verifying results ...")
        e = np.equal(ref, aie_output)
        errors = np.size(e) - np.count_nonzero(e)

    if not errors:
        print("\nPASS!\n")
        return 0
    else:
        print("\nError count: ", errors)
        print("\nFailed.\n")
        return 1
