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
    design = "conv2d_with_relu"
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

    shape_total_wts = (4096, 1)
    shape_in_act = (32, 8, 32, 8)  #'YCXC8' , 'CYX'
    shape_in_wts1 = (8, 8, 1, 1, 8, 8)  # out,in,ky,kx,in8,out8
    shape_out = (32, 8, 32, 8)

    # ------------------------------------------------------
    # Initialize activation, weights, scaling factor for int8 model
    # ------------------------------------------------------
    int_inp = torch.randint(1, 100, (1, 64, 32, 32)).type(torch.FloatTensor)
    int_weight = torch.randint(50, 100, (64, 64, 1, 1)).type(torch.FloatTensor)
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
    class conv2d_relu_int_model(nn.Module):
        def __init__(self, in_planes=64, planes=64):
            super(conv2d_relu_int_model, self).__init__()
            self.conv = nn.Conv2d(64, 64, kernel_size=1, bias=False)
            self.relu = nn.ReLU()

        def forward(self, x):
            out_int = self.conv(x)
            out_float = out_int * conv_scale
            out_int = self.relu(out_float)
            out_float = relu_scale * torch.clamp(
                torch.round(out_int / relu_scale), min, max
            )  # converting to int to do proper clipping
            return out_float

    # ------------------------------------------------------
    # Pytorch baseline
    # ------------------------------------------------------
    model = conv2d_relu_int_model()
    model.eval()
    model.conv.weight.data.copy_(int_weight)
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

    wts1 = ds.reorder_mat(int_weight.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX")
    total_wts = np.concatenate((wts1), axis=None)
    total_wts.tofile(log_folder + "/weights_mem_fmt_final.txt", sep=",", format="%d")

    # ------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------
    for i in range(num_iter):
        start = time.time_ns()
        aie_output = execute(app, ifm_mem_fmt, total_wts) * relu_scale
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
    temp_out = aie_output.reshape(32, 8, 32, 8)
    temp_out = ds.reorder_mat(temp_out, "CDYX", "YCXD")
    ofm_mem_fmt = temp_out.reshape(64, 32, 32)
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
        atol=2 * relu_scale,
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
