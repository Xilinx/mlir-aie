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
    design = "conv2d"
    xclbin_path = opts.xclbin
    insts_path = opts.instr

    log_folder = "log/"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    width = int(opts.width)
    height = int(opts.height)
    ci = int(opts.in_channels)
    co = int(opts.out_channels)

    ci8 = ci // 8
    co8 = co // 8

    num_iter = 1
    npu_time_total = 0
    npu_time_min = 9999999
    npu_time_max = 0
    trace_size = opts.trace_size
    enable_trace = False if not trace_size else True
    trace_file = "log/trace_" + design + ".txt"
    # ------------------------------------------------------
    # Configure this to match your design's buffer size
    # ------------------------------------------------------
    dtype_in = np.dtype("int8")
    dtype_wts = np.dtype("int8")
    dtype_out = np.dtype("int8")

    ci_co = ci * co
    shape_total_wts = (ci_co, 1)
    shape_in_act = (height, ci8, width, 8)  #'YCXC8' , 'CYX'
    shape_in_wts1 = (co8, ci8, 1, 1, 8, 8)  # out,in,ky,kx,in8,out8
    shape_out = (height, co8, width, 8)

    # ------------------------------------------------------
    # Initialize activation, weights, scaling factor for int8 model
    # ------------------------------------------------------
    int_inp = torch.randint(1, 20, (1, ci, height, width)).type(torch.FloatTensor)
    int_weight = torch.randint(50, 80, (co, ci, 1, 1)).type(torch.FloatTensor)
    conv_scale = 7.6294e-06  # scale to convert int8 output to floating point
    int8_scale = 0.0078  # scale to convert int8 output to floating point
    min = -128
    max = 127
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
        trace_after_output=True,
    )

    # ------------------------------------------------------
    # Define your golden reference
    # ------------------------------------------------------
    class conv2d_int_model(nn.Module):
        def __init__(self, in_planes=ci, out_planes=co):
            super(conv2d_int_model, self).__init__()
            # self.conv = nn.Conv2d(64, 64, kernel_size=1, bias=False)
            self.conv = nn.Conv2d(ci, co, kernel_size=1, bias=False)

        def forward(self, x):
            out_int = self.conv(x)
            out_quant = out_int * conv_scale  # int8 x int8 leads to int32 output
            out_float = int8_scale * torch.clamp(
                torch.round(out_quant / int8_scale), min, max
            )  # converting to int8 range
            return out_float

    # ------------------------------------------------------
    # Pytorch baseline
    # ------------------------------------------------------
    model = conv2d_int_model()
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
        entire_buffer = execute(app, ifm_mem_fmt, total_wts)
        stop = time.time_ns()

        if enable_trace:
            # Separate data and trace
            data_buffer, trace_buffer = extract_trace(
                entire_buffer, shape_out, dtype_out, trace_size
            )
            # Scale the data
            data_buffer = data_buffer * int8_scale
            # Write out the trace
            write_out_trace(trace_buffer, trace_file)
        else:
            data_buffer = entire_buffer * int8_scale
            trace_buffer = None

        npu_time = stop - start
        npu_time_total = npu_time_total + npu_time

    # ------------------------------------------------------
    # Reorder output data-layout
    # ------------------------------------------------------
    temp_out = data_buffer.reshape(height, co8, width, 8)
    temp_out = ds.reorder_mat(temp_out, "CDYX", "YCXD")
    ofm_mem_fmt = temp_out.reshape(co, height, width)
    if enable_trace:
        ofm_log_filename = "/after_ofm_mem_fmt_final_trace.txt"
    else:
        ofm_log_filename = "/after_ofm_mem_fmt_final.txt"
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
        atol=2 * int8_scale,
    ):
        print("\nPASS!\n")
        exit(0)
    else:
        print("\nFailed.\n")
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    p.add_argument(
        "-wd",
        "--width",
        dest="width",
        default=32,
        help="Width of convolution tile",
    )
    p.add_argument(
        "-ht",
        "--height",
        dest="height",
        default=32,
        help="Height of convolution tile",
    )
    p.add_argument(
        "-ic",
        "--in_channels",
        dest="in_channels",
        default=64,
        help="Number of input channels for convolution tile",
    )
    p.add_argument(
        "-oc",
        "--out_channels",
        dest="out_channels",
        default=64,
        help="Number of output channels for convolution tile",
    )
    opts = p.parse_args(sys.argv[1:])
    main(opts)
