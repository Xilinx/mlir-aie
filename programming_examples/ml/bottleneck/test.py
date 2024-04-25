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
from aie.utils.xrt import setup_aie, extract_trace, write_out_trace, execute
import aie.utils.test as test_utils

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)


def main(opts):
    design = "bottleneck_int8"
    xclbin_path = opts.xclbin
    insts_path = opts.instr

    log_folder = "log/"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

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
    dtype_out = np.dtype("uint8")

    shape_in_act = (32, 32, 32, 8)
    shape_in_wts1 = (8, 32, 1, 1, 8, 8)  # out,in,ky,kx,in8,out8
    shape_in_wts2 = (8, 8, 3, 3, 8, 8)  # out,in,ky,kx,in8,out8
    shape_in_wts3 = (32, 8, 1, 1, 8, 8)  # out,in,ky,kx,in8,out8
    shape_total_wts = (69632, 1)
    shape_out = (32, 32, 32, 8)

    # ------------------------------------------------------
    # Initialize activation, weights, scaling factor for int8 model
    # ------------------------------------------------------
    int_inp = torch.randint(1, 100, (1, 256, 32, 32)).type(torch.FloatTensor)
    int_weight1 = torch.randint(50, 100, (64, 256, 1, 1)).type(torch.FloatTensor)
    int_weight2 = torch.randint(50, 100, (64, 64, 3, 3)).type(torch.FloatTensor)
    int_weight3 = torch.randint(50, 100, (256, 64, 1, 1)).type(torch.FloatTensor)

    inp_scale1 = 0.5
    inp_scale2 = 0.5
    inp_scale3 = 0.5
    inp_scale4 = 0.5

    weight_scale1 = 0.5
    weight_scale2 = 0.5
    weight_scale3 = 0.5

    combined_scale1 = -math.log2(inp_scale1 * weight_scale1 / inp_scale2)
    combined_scale2 = -math.log2(inp_scale2 * weight_scale2 / inp_scale3)
    combined_scale3 = -math.log2(inp_scale3 * weight_scale3 / inp_scale1)
    combined_scale4 = -math.log2(inp_scale1 / inp_scale4)
    conv_scale = 0.0039  # scale to convert int8 output to floating point
    relu_scale = 0.0078  # scale to convert int8 output to floating point
    min = 0
    max = 255

    # ------------------------------------------------------
    # Get device, load the xclbin & kernel and register them
    # ------------------------------------------------------
    app = setup_aie(
        xclbin_path,
        insts_path,
        shape_in_act,
        dtype_in,
        shape_total_wts,
        dtype_wts,
        shape_out,
        dtype_out,
        enable_trace=enable_trace,
        trace_size=trace_size,
    )

    # ------------------------------------------------------
    # Define your golden reference
    # ------------------------------------------------------
    class bottleneck_int8(nn.Module):
        def __init__(self, in_planes=256, planes=64):
            super(bottleneck_int8, self).__init__()
            self.conv1 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
            self.conv2 = nn.Conv2d(
                64, 64, kernel_size=3, padding=1, padding_mode="zeros", bias=False
            )
            self.conv3 = nn.Conv2d(64, 256, kernel_size=1, bias=False)

            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()

        def forward(self, x):
            conv1_out = self.conv1(x) * inp_scale1 * weight_scale1
            relu1_out = torch.clamp(
                torch.round(self.relu1(conv1_out) / inp_scale2), min, max
            )  # convert to int and apply relu
            conv2_out = self.conv2(relu1_out) * inp_scale2 * weight_scale2
            relu2_out = torch.clamp(
                torch.round(self.relu2(conv2_out) / inp_scale3), min, max
            )
            conv3_out = self.conv3(relu2_out) * inp_scale3 * weight_scale3
            same_scale_init = torch.clamp(
                torch.round(conv3_out / inp_scale1), -128, 127
            )
            skip_add = inp_scale1 * (same_scale_init + int_inp)
            final_out = inp_scale4 * (
                torch.clamp(torch.round(skip_add / inp_scale4), min, max)
            )
            return final_out

    # ------------------------------------------------------
    # Pytorch baseline
    # ------------------------------------------------------
    model = bottleneck_int8()
    model.eval()
    model.conv1.weight.data.copy_(int_weight1)
    model.conv2.weight.data.copy_(int_weight2)
    model.conv3.weight.data.copy_(int_weight3)

    golden_output = model(int_inp)

    # ------------------------------------------------------
    # Reorder input data-layout
    # ------------------------------------------------------
    ds = DataShaper()
    before_input = int_inp.squeeze().data.numpy().astype(dtype_in)
    before_input.tofile(
        log_folder + "/before_ifm_mem_fmt_1x1.txt", sep=",", format="%d"
    )
    ifm_mem_fmt = ds.reorder_mat(before_input, "YCXC8", "CYX")
    ifm_mem_fmt.tofile(log_folder + "/after_ifm_mem_fmt_1x1.txt", sep=",", format="%d")

    wts1 = ds.reorder_mat(int_weight1.data.numpy().astype(dtype_in), "OIYXI8O8", "OIYX")
    wts2 = ds.reorder_mat(int_weight2.data.numpy().astype(dtype_in), "OIYXI8O8", "OIYX")
    wts3 = ds.reorder_mat(int_weight3.data.numpy().astype(dtype_in), "OIYXI8O8", "OIYX")

    total_wts = np.concatenate((wts1, wts2, wts3), axis=None)
    total_wts.tofile(log_folder + "/weights_mem_fmt_final.txt", sep=",", format="%d")

    # ------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------
    for i in range(num_iter):
        start = time.time_ns()
        aie_output = execute(app, ifm_mem_fmt, total_wts) * inp_scale4
        stop = time.time_ns()

        if enable_trace:
            aie_output, trace = extract_trace(
                aie_output, shape_out, dtype_out, trace_size
            )
            write_out_trace(trace, trace_file)

        npu_time = stop - start
        npu_time_total = npu_time_total + npu_time

    # ------------------------------------------------------
    # Reorder output data-layout
    # ------------------------------------------------------
    temp_out = aie_output.reshape(32, 32, 32, 8)
    temp_out = ds.reorder_mat(temp_out, "CDYX", "YCXD")
    ofm_mem_fmt = temp_out.reshape(256, 32, 32)
    ofm_mem_fmt.tofile(
        log_folder + "/after_ofm_mem_fmt_final.txt", sep=",", format="%d"
    )
    ofm_mem_fmt_out = torch.from_numpy(ofm_mem_fmt).unsqueeze(0)

    # ------------------------------------------------------
    # Compare the AIE output and the golden reference
    # ------------------------------------------------------
    print("\nAvg NPU time: {}us.".format(int((npu_time_total / num_iter) / 1000)))

    if np.allclose(
        ofm_mem_fmt_out.detach().numpy(),
        golden_output.detach().numpy(),
        rtol=0,
        atol=inp_scale4,
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
