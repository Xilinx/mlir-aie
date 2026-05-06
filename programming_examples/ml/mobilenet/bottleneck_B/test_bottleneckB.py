#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

import torch
import torch.nn as nn
import sys
import math
from aie.utils.ml import DataShaper
import time
import os
import numpy as np
import aie.iron as iron
from aie.utils import DefaultNPURuntime
from aie.utils import TraceConfig, HostRuntime, NPUKernel, DefaultNPURuntime
import aie.utils.test as test_utils
import json

sys.path.append("..")
import mb_utils

log_dir = "log/"
data_dir = "data/"

vectorSize = 8

bneck_10_InW1 = 14
bneck_10_InH1 = 14
bneck_10_InC1 = 80
bneck_10_OutC1 = 480

bneck_10_InW2 = 14
bneck_10_InH2 = 14
bneck_10_OutC2 = bneck_10_OutC1

bneck_10_InW3 = 14
bneck_10_InH3 = 14
bneck_10_OutC3 = 112

bneck_11_OutC1 = 336
bneck_11_OutC2 = 336
bneck_11_OutC3 = 112
kdim = 3
stride = 1
padding = 1

bneck_12_InH1 = 14
bneck_12_InW1 = 14
bneck_12_OutC1 = 336

bneck_12_InH2 = 14
bneck_12_InW2 = 14
bneck_12_OutC2 = 336

bneck_12_InW3 = 7
bneck_12_InH3 = 7
bneck_12_OutC3 = 80

OutC = bneck_12_OutC3
OutH = bneck_12_InH3
OutW = bneck_12_InW3

OutC_vec = math.floor(OutC / vectorSize)

InC_vec = math.floor(bneck_10_InC1 / vectorSize)
wts_size = (
    (bneck_10_InC1 * bneck_10_OutC1)
    + (3 * 3 * bneck_10_OutC2)
    + (bneck_10_OutC2 * bneck_10_OutC3)
    + (bneck_10_OutC3 * bneck_11_OutC1)
    + (3 * 3 * bneck_11_OutC2)
    + (bneck_11_OutC2 * bneck_11_OutC3)
    + (bneck_11_OutC3 * bneck_12_OutC1)
    + (3 * 3 * bneck_12_OutC2)
    + (bneck_12_OutC2 * bneck_12_OutC3)
)


def main(opts):
    design = "mobilenet_bottleneck_B"
    xclbin_path = opts.xclbin
    insts_path = opts.instr

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    num_iter = 1
    npu_time_total = 0
    npu_time_min = 9999999
    npu_time_max = 0
    trace_size = 16384
    enable_trace = False
    trace_file = "log/trace_" + design + ".txt"

    # ------------------------------------------------------
    # Configure this to match your design's buffer size
    # ------------------------------------------------------
    dtype_in = np.dtype("int8")
    dtype_wts = np.dtype("int8")
    dtype_out = np.dtype("int8")

    shape_total_wts = (wts_size, 1)
    shape_in_act = (bneck_10_InH1, InC_vec, bneck_10_InW1, vectorSize)  #'YCXC8' , 'CYX'
    shape_out = (OutH, OutC_vec, OutW, vectorSize)  # bneck_12_OutC3/8
    shape_out_final = (OutC_vec * vectorSize, OutH, OutW)  # bneck_12_OutC3/8
    # ------------------------------------------------------
    # Initialize activation, weights, scaling factor for int8 model
    # ------------------------------------------------------
    input = torch.randn(1, bneck_10_InC1, bneck_10_InH1, bneck_10_InW1)
    # ------------------------------------------------------
    # Get device, load the xclbin & kernel and register them
    # ------------------------------------------------------
    npu_kernel = NPUKernel(xclbin_path, insts_path)
    kernel_handle = DefaultNPURuntime.load(npu_kernel)

    golden_output = np.loadtxt(
        data_dir + "golden_output.txt", delimiter=",", dtype="int8"
    )
    golden_output = golden_output.reshape(
        1, bneck_12_OutC3, bneck_12_InH3, bneck_12_InW3
    )
    ds = DataShaper()

    before_input = np.loadtxt(
        data_dir + "before_ifm_mem_fmt_1x1.txt", delimiter=",", dtype="int8"
    )
    before_input = before_input.reshape(bneck_10_InC1, bneck_10_InH1, bneck_10_InW1)
    # print("JL: before_input shape:", before_input.shape)
    # print("JL: before_input type:", type(before_input))
    # print("JL: before_input dtype:", before_input.dtype)
    ifm_mem_fmt = ds.reorder_mat(before_input, "YCXC8", "CYX")

    bn10_wts1 = np.loadtxt(data_dir + "bn10_1_chain.txt", delimiter=",", dtype="int32")
    bn10_wts2 = np.loadtxt(data_dir + "bn10_2_chain.txt", delimiter=",", dtype="int32")
    bn10_wts3 = np.loadtxt(data_dir + "bn10_3_chain.txt", delimiter=",", dtype="int32")

    bn11_wts1 = np.loadtxt(data_dir + "bn11_1_chain.txt", delimiter=",", dtype="int32")
    bn11_wts2 = np.loadtxt(data_dir + "bn11_2_chain.txt", delimiter=",", dtype="int32")
    bn11_wts3 = np.loadtxt(data_dir + "bn11_3_chain.txt", delimiter=",", dtype="int32")

    bn12_wts1 = np.loadtxt(data_dir + "bn12_1_chain.txt", delimiter=",", dtype="int32")
    bn12_wts2 = np.loadtxt(data_dir + "bn12_2_chain.txt", delimiter=",", dtype="int32")
    bn12_wts3 = np.loadtxt(data_dir + "bn12_3_chain.txt", delimiter=",", dtype="int32")
    bn12_wts2_3 = np.loadtxt(
        data_dir + "bn12_2_3_chain.txt", delimiter=",", dtype="int32"
    )

    bn10_total_wts = np.concatenate((bn10_wts1, bn10_wts2, bn10_wts3), axis=None)
    bn11_total_wts = np.concatenate((bn11_wts1, bn11_wts2, bn11_wts3), axis=None)
    bn12_total_wts = np.concatenate((bn12_wts1, bn12_wts2, bn12_wts3), axis=None)
    total_wts = np.concatenate(
        (bn10_total_wts, bn11_total_wts, bn12_total_wts), axis=None
    )

    total_wts.tofile(log_dir + "after_weights_mem_fmt_final.txt", sep=",", format="%d")
    print(total_wts.shape)

    # ------------------------------------------------------
    # Setup buffers run loop
    # ------------------------------------------------------
    in1 = iron.tensor(ifm_mem_fmt, dtype=dtype_in)
    in2 = iron.tensor(total_wts, dtype=dtype_wts)
    out = iron.zeros(shape_out, dtype=dtype_out)
    buffers = [in1, in2, out]

    trace_config = None
    if enable_trace:
        trace_config = TraceConfig(
            trace_size=trace_size,
            trace_file=trace_file,
            enable_ctrl_pkts=False,
            last_tensor_shape=out.shape,
            last_tensor_dtype=out.dtype,
        )
        HostRuntime.prepare_args_for_trace(buffers, trace_config)

    # ------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------

    times = []
    for i in range(num_iter):
        start = time.time_ns()
        DefaultNPURuntime.run(kernel_handle, buffers)
        stop = time.time_ns()
        npu_time = stop - start
        times.append(npu_time)

        # ------------------------------------------------------
        # Reorder output data-layout
        # ------------------------------------------------------
        aie_output = out.numpy()  # .astype(dtype_out)
        temp_out = aie_output.reshape(shape_out)
        temp_out = ds.reorder_mat(temp_out, "CDYX", "YCXD")
        ofm_mem_fmt = temp_out.reshape(shape_out_final)
        ofm_mem_fmt.tofile(
            log_dir + "after_ofm_mem_fmt_final.txt", sep=",", format="%d"
        )
        ofm_mem_fmt_out = torch.from_numpy(ofm_mem_fmt).unsqueeze(0)
        # print("\nIter:{}, NPU time: {}us.".format(i,int((npu_time) / 1000)))
        #
        print("AIE:", ofm_mem_fmt_out)
        print("Golden (int):", golden_output)

        # ------------------------------------------------------
        # Compare the AIE output and the golden reference
        # ------------------------------------------------------

        # print("Golden  (float): ",q_bottleneck_out.value.detach())

        golden = mb_utils.convert_to_numpy(golden_output)
        ofm_mem_fmt_out = mb_utils.convert_to_numpy(ofm_mem_fmt_out)
        max_diff_int = np.max((golden) - (ofm_mem_fmt_out))
        # print("atol: {} max difference (float/int): {} / {}".format(atol,max_diff,max_diff_int))
        print("max difference (int): {}".format(max_diff_int))

    average_time = sum(times) / num_iter
    best_time = min(times)
    print(
        "\nNPU time= Avg: {}us, Best: {}us.".format(
            int((average_time) / 1000), int((best_time) / 1000)
        )
    )

    if np.allclose(
        golden,
        ofm_mem_fmt_out,
        rtol=0,
        atol=5,
    ):
        print("\nPASS!\n")
        exit(0)
    else:
        print("\nFailed.\n")
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
