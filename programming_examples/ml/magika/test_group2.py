#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

import sys
import math

import time
import os
import numpy as np
import aie.utils.test as test_utils
import aie.iron as iron
from aie.utils import TraceConfig, HostRuntime, NPUKernel, DefaultNPURuntime
from pathlib import Path


def get_evm(array_len, gold, dut):
    diff_pwr = 0.0
    tota_pwr = 0.0

    print(f"array_len: {array_len}, gold.size: {gold.size}, dut_size: {dut.size}")

    for i in range(array_len):
        diff = gold[i] - dut[i]
        diff_pwr += diff * diff
        tota_pwr += gold[i] * gold[i]
    evm = 10 * math.log10(diff_pwr / tota_pwr)
    # evm = 0
    return evm


def main(opts):
    design = "magika"
    xclbin_path = opts.xclbin
    insts_path = opts.instr

    log_folder = "log/"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    num_iter = 1
    npu_time_total = 0
    npu_time_min = 9999999
    npu_time_max = 0
    trace_size = opts.trace_size
    enable_trace = opts.trace_size > 0
    trace_after_output = False
    trace_file = "log/trace_" + design + ".txt"
    # ------------------------------------------------------
    # Configure this to match your design's buffer size
    # ------------------------------------------------------
    dtype_in = np.dtype("int16")
    dtype_out = np.dtype("int32")

    shape_in = (2048,)
    shape_out = (214,)

    dummy = (4,)
    dummy_dtype = np.dtype("int32")
    dummy_data = np.zeros([4], dtype=dummy_dtype)

    min = 0
    max = 255

    # ------------------------------------------------------
    # Get device, load the xclbin & kernel and register them
    # ------------------------------------------------------
    npu_kernel = NPUKernel(xclbin_path, insts_path, kernel_name=opts.kernel)
    kernel_handle = DefaultNPURuntime.load(npu_kernel)

    # ------------------------------------------------------
    # Pytorch baseline
    # ------------------------------------------------------
    lines = np.loadtxt("./data/g1.txt")
    int_inp = lines.reshape(
        1024,
    )[:512]

    # modify group2 input to insert 4 0's after every 12 beats of data
    count = 0
    int_inp_mod = []
    for i in int_inp:
        int_inp_mod.append(i)
        count += 1
        if count % 12 == 0:
            int_inp_mod.append(0)
            int_inp_mod.append(0)
            int_inp_mod.append(0)
            int_inp_mod.append(0)

    # ------------------------------------------------------
    # Reorder input data-layout
    # ------------------------------------------------------
    import ml_dtypes

    before_input_orig = int_inp.astype(dtype_in)
    before_input_orig.tofile(log_folder + "/before_g2_orig.txt", sep=",", format="%d")

    before_input = np.array(int_inp_mod).astype(ml_dtypes.bfloat16).view(dtype_in)
    before_input.tofile(log_folder + "/before_g2.txt", sep=",", format="%d")

    print(f"before_input_orig size: {before_input_orig.size}")
    print(f"before_input size: {before_input.size}")
    # print(before_input)

    in1 = iron.tensor(before_input, dtype=dtype_in)
    dummy = iron.zeros(4, dtype=dummy_dtype)
    out = iron.zeros(shape_out[0], dtype=dtype_out)
    buffers = [in1, dummy, out]
    # buffers = [in1, out]

    trace_config = None
    if enable_trace:
        trace_config = TraceConfig(
            trace_size=trace_size,
            trace_file=trace_file,
            trace_after_last_tensor=trace_after_output,
            enable_ctrl_pkts=False,
            last_tensor_shape=out.shape,
            last_tensor_dtype=out.dtype,
        )
        HostRuntime.prepare_args_for_trace(buffers, trace_config)

    # ------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------
    for i in range(num_iter):
        start = time.time_ns()
        ret = DefaultNPURuntime.run(kernel_handle, buffers)
        stop = time.time_ns()

        if enable_trace:
            trace_buffer, _ = HostRuntime.extract_trace_from_args(buffers, trace_config)
            trace_buffer = trace_buffer.view(np.uint32)
            trace_config.write_trace(trace_buffer)

        out_tensor = out.numpy()
        if not isinstance(out_tensor, np.ndarray):
            out_tensor = out_tensor.numpy()
        aie_output = out_tensor

        npu_time = stop - start
        npu_time_total = npu_time_total + npu_time

    print("aie_output")
    print("aie output size: " + str(aie_output.size))
    # print(aie_output)

    # ------------------------------------------------------
    # Reorder output data-layout
    # ------------------------------------------------------
    aie_output.tofile(log_folder + "/aie_output_int.txt", sep="\n", format="%d")

    aie_output_float = aie_output.view(np.float32)
    aie_output_float.tofile(log_folder + "/aie_output_float.txt", sep="\n", format="%E")

    # ------------------------------------------------------
    # Compare the AIE output and the golden reference
    # ------------------------------------------------------
    print("\nAvg NPU time: {}us.".format(int((npu_time_total / num_iter) / 1000)))

    ref = np.loadtxt("./data/g2.txt")

    this_evm = get_evm(
        shape_out[0],
        ref.astype(float),
        aie_output_float.astype(float),
    )
    print(f"this evm: {this_evm}")

    if this_evm < -30:
        print("\nPASS.\n")
    else:
        print("\nFAIL. Error vector magitude (EVM) > -30\n")

    print("\nDone.\n")
    exit(0)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
