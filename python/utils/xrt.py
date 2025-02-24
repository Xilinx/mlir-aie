# xrt.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
import copy
import time
import numpy as np
import pyxrt as xrt


class AIE_Application:

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

    def register_buffer(self, group_id, *args, **kwargs):
        self.buffers[group_id] = AIE_Buffer(self, group_id, *args, **kwargs)

    def run(self):
        self.insts_buffer.sync_to_device()
        h = self.call()
        r = h.wait()
        if r != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise Exception(f"Kernel returned {r}")

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
        del self.kernel
        del self.device


class AIE_Buffer:

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

    def read(self):
        self.sync_from_device()
        return self.bo.read(self.len_bytes, 0).view(self.dtype).reshape(self.shape)

    def write(self, v, offset=0):
        self.bo.write(v.view(np.uint8), offset)
        self.sync_to_device()

    def sync_to_device(self):
        return self.bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def sync_from_device(self):
        return self.bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    def __del__(self):
        del self.bo
        self.bo = None


class AIE_Application_Error(Exception):
    pass


insts_cache = {}


def read_insts(insts_path):
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
):
    app = AIE_Application(xclbin_path, insts_path, kernel_name)

    if in_0_shape or in_0_dtype:
        app.register_buffer(3, shape=in_0_shape, dtype=in_0_dtype)
    if in_1_shape or in_1_dtype:
        app.register_buffer(4, shape=in_1_shape, dtype=in_1_dtype)

    if enable_trace:
        out_buf_len_bytes = np.prod(out_buf_shape) * np.dtype(out_buf_dtype).itemsize
        out_buf_shape = (out_buf_len_bytes + trace_size,)
        out_buf_dtype = np.uint8
    app.register_buffer(5, shape=out_buf_shape, dtype=out_buf_dtype)
    return app


def extract_trace(out_buf, out_buf_shape, out_buf_dtype, trace_size):
    trace_size_words = trace_size // 4
    out_buf_flat = out_buf.reshape((-1,)).view(np.uint32)
    output_prefix = (
        out_buf_flat[:-trace_size_words].view(out_buf_dtype).reshape(out_buf_shape)
    )
    trace_suffix = out_buf_flat[-trace_size_words:]
    return output_prefix, trace_suffix


def write_out_trace(trace, file_name):
    out_str = "\n".join(f"{i:0{8}x}" for i in trace if i != 0)
    with open(file_name, "w") as f:
        f.write(out_str)


def execute(app, input_one=None, input_two=None):
    if not (input_one is None):
        app.buffers[3].write(input_one)
    if not (input_two is None):
        app.buffers[4].write(input_two)
    app.run()
    return app.buffers[5].read()


def xrt_test_run(
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
):
    enable_trace = opts.trace_size > 0

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
    )

    out_size = out_volume * out_data.itemsize
    # print("out_size: " + str(out_size))

    start = time.time_ns()
    full_output = execute(app, in1_data, in2_data)
    stop = time.time_ns()
    npu_time = stop - start
    print("npu_time: ", npu_time)

    aie_output = full_output[:out_size].view(out_dtype)
    if enable_trace:
        trace_buffer = full_output[out_size:].view(np.uint32)

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
