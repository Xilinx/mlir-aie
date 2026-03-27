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

sys.path.append("..")
import mb_utils

log_dir = "log/"
data_dir = "data/"

vectorSize = 8


def main(opts):
    bn = opts.bn
    design = "mobilenet_bottleneck_A_bn" + bn
    print(design)

    if bn == "0":
        tensorInW = 112
        tensorInH = 112
        tensorInC = 16
        tensorOutC = 16
        depthwiseStride = 1
        depthWiseChannels = 72
    elif bn == "1":
        tensorInW = 112
        tensorInH = 112
        tensorInC = 16
        tensorOutC = 24
        depthwiseStride = 2
        depthWiseChannels = 64
    elif bn == "2":
        tensorInW = 56
        tensorInH = 56
        tensorInC = 24
        tensorOutC = tensorInC
        depthwiseStride = 1
        depthWiseChannels = 72
    elif bn == "3":
        tensorInW = 56
        tensorInH = 56
        tensorInC = 24
        tensorOutC = 40
        depthwiseStride = 2
        depthWiseChannels = 72
    elif bn == "6":
        tensorInW = 28
        tensorInH = 28
        tensorInC = 40
        tensorOutC = 80
        depthwiseStride = 2
        depthWiseChannels = 240
    elif bn == "7":
        tensorInW = 14
        tensorInH = 14
        tensorInC = 80
        tensorOutC = 80
        depthwiseStride = 1
        depthWiseChannels = 200
    elif bn == "8":
        tensorInW = 14
        tensorInH = 14
        tensorInC = 80
        tensorOutC = 80
        depthwiseStride = 1
        depthWiseChannels = 184
    else:
        print("ERROR: invalid or unsupported BN layer selected")
        exit(-1)

    bneck_InW1 = tensorInW
    bneck_InH1 = tensorInH
    bneck_InC1 = tensorInC
    bneck_OutC1 = depthWiseChannels

    bneck_InW2 = bneck_InW1
    bneck_InH2 = bneck_InH1
    bneck_OutC2 = bneck_OutC1

    bneck_InW3 = bneck_InW2 // depthwiseStride
    bneck_InH3 = bneck_InH2 // depthwiseStride
    bneck_OutC3 = tensorOutC

    bneck_InC1_vec = math.floor(bneck_InC1 / vectorSize)
    bneck_OutC3_vec = math.floor(bneck_OutC3 / vectorSize)

    tensorOutW = bneck_InW3
    tensorOutH = bneck_InH3

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
    # Get device, load the xclbin & kernel and register them
    # ------------------------------------------------------
    npu_kernel = NPUKernel(xclbin_path, insts_path)
    kernel_handle = DefaultNPURuntime.load(npu_kernel)

    # ------------------------------------------------------
    # Configure this to match your design's buffer size
    # ------------------------------------------------------
    dtype_in = np.dtype("int8")
    dtype_wts = np.dtype("int8")
    dtype_out = np.dtype("int8")

    shape_total_wts = (
        bneck_InC1 * bneck_OutC1 + 3 * 3 * bneck_OutC2 + bneck_OutC2 * bneck_OutC3,
        1,
    )
    shape_in_act = (
        bneck_InH1,
        bneck_InC1_vec,
        bneck_InW1,
        vectorSize,
    )  #'YCXC8' , 'CYX'
    shape_out = (bneck_InH3, bneck_OutC3_vec, bneck_InW3, vectorSize)  # HCWC8
    shape_out_final = (
        bneck_OutC3_vec * vectorSize,
        bneck_InH3,
        bneck_InW3,
    )  # CHW

    golden_output = np.loadtxt(
        data_dir + "golden_output_bn" + bn + "_single.txt", delimiter=",", dtype="int8"
    )
    golden_output = golden_output.reshape(1, tensorOutC, tensorOutH, tensorOutW)  # TODO

    ds = DataShaper()

    # before_input = int_inp.squeeze().data.numpy().astype(dtype_in)
    # before_input.tofile(
    #     log_dir + "/before_ifm_mem_fmt_1x1.txt", sep=",", format="%d"
    # )

    before_input = np.loadtxt(
        data_dir + "input_bn" + bn + "_single.txt", delimiter=",", dtype="uint8"
    )
    before_input = before_input.reshape(tensorInC, tensorInH, tensorInW)

    ifm_mem_fmt = ds.reorder_mat(before_input, "YCXC8", "CYX")
    ifm_mem_fmt.tofile(log_dir + "/after_ifm_mem_fmt.txt", sep=",", format="%d")

    # **************************** bn1 ****************************
    # bn1_wts1 = ds.reorder_mat(
    #     block_1_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    # )
    # bn1_wts2 = ds.reorder_mat(
    #     block_1_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    # )
    # bn1_wts3 = ds.reorder_mat(
    #     block_1_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    # )

    # total_wts = np.concatenate((bn1_wts1, bn1_wts2, bn1_wts3), axis=None)

    # total_wts.tofile(
    #     log_dir + "/bn1_after_weights_mem_fmt_final.txt", sep=",", format="%d"
    # )
    # print(total_wts.shape)

    total_wts = np.loadtxt(
        data_dir + "bn" + bn + "_single.txt", delimiter=",", dtype="int8"
    )

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
            trace_after_last_tensor=False,
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
        DefaultNPURuntime.run(kernel_handle, buffers)
        stop = time.time_ns()
        npu_time = stop - start
        npu_time_total = npu_time_total + npu_time

    # ------------------------------------------------------
    # Reorder output data-layout
    # ------------------------------------------------------
    aie_output = out.numpy()  # .astype(dtype_out)
    temp_out = aie_output.reshape(shape_out)
    temp_out = ds.reorder_mat(temp_out, "CDYX", "YCXD")
    ofm_mem_fmt = temp_out.reshape(shape_out_final)
    ofm_mem_fmt.tofile(log_dir + "after_ofm_mem_fmt_final.txt", sep=",", format="%d")
    ofm_mem_fmt_out = torch.from_numpy(ofm_mem_fmt).unsqueeze(0)
    print(ofm_mem_fmt_out)
    # ------------------------------------------------------
    # Compare the AIE output and the golden reference
    # ------------------------------------------------------
    print("\nAvg NPU time: {}us.".format(int((npu_time_total / num_iter) / 1000)))
    from mb_utils import convert_to_numpy

    golden = convert_to_numpy(golden_output)
    ofm_mem_fmt_out = convert_to_numpy(ofm_mem_fmt_out)
    max_diff_int = np.max((golden) - (ofm_mem_fmt_out))
    print("max difference (int): {}".format(max_diff_int))

    if np.allclose(
        ofm_mem_fmt_out,
        golden_output,
        rtol=0,
        atol=1,
    ):
        print("\nPASS!\n")
        exit(0)
    else:
        print("\nFailed.\n")
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    p.add_argument(
        "-bn",
        dest="bn",
        default="1",
        type=str,
        help="selected BN layer",
    )
    opts = p.parse_args(sys.argv[1:])
    main(opts)
