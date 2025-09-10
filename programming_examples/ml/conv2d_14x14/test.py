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
    ksz = int(opts.kernel_size)

    ci8 = ci // 8
    co8 = co // 8

    width_out = width // ksz
    height_out = height // ksz

    num_iter = 1
    npu_time_total = 0
    npu_time_min = 9999999
    npu_time_max = 0
    trace_size = opts.trace_size
    enable_trace = False if not trace_size else True
    print("Trace status: " + str(enable_trace))
    trace_file = "log/trace_" + design + ".txt"

    # ------------------------------------------------------
    # Configure this to match your design's buffer size
    # ------------------------------------------------------
    dtype_in = np.dtype("uint8")
    dtype_wts = np.dtype("int8")
    dtype_out = np.dtype("int8")

    ci_co_ksz_ksz = ci * co * ksz * ksz
    co_group = co // 16  # 72

    shape_total_wts = (ci_co_ksz_ksz, 1)
    shape_in_act = (co_group, height, width, ci)  #'YX (rgba)
    shape_in_wts1 = (co, ci, ksz, ksz)  # out,in,ky,kx
    shape_out = (co, height_out, width_out)

    print("shape_in_act: ")
    print(shape_in_act)
    print("shape_total_wts: ")
    print(shape_total_wts)
    print("shape_out: ")
    print(shape_out)

    # ------------------------------------------------------
    # Initialize activation, weights, scaling factor for int8 model
    # ------------------------------------------------------
    # int_inp = torch.randint(1, 20, (1, ci, height, width)).type(torch.FloatTensor)
    # int_inp = torch.randint(0, 127, (1, ci, height, width)).type(torch.FloatTensor) # ch, height, width
    int_inp = torch.randint(0, 255, (1, ci, height, width)).type(
        torch.FloatTensor
    )  # ch, height, width

    # int_weight = torch.randint(50, 80, (co, ci, ksz, ksz)).type(torch.FloatTensor)
    int_weight = torch.randint(2, 20, (co, ci, ksz, ksz)).type(
        torch.FloatTensor
    )  # co, ci, kh, kw
    # int_weight = torch.randint(5, 10, (co, ci, ksz, ksz)).type(torch.FloatTensor)

    # conv_scale = 7.6294e-06  # scale to convert int8 output to floating point 1/2^17
    # int8_scale = 0.0078  # scale to convert int8 output to floating point, 1/2^7

    # conv_scale = 9.5367e-07  # scale to convert int8 output to floating point 1/2^20
    conv_scale = 1.9073e-06  # scale to convert int8 output to floating point 1/2^19
    # conv_scale = 3.8146e-06  # scale to convert int8 output to floating point 1/2^18

    # int8_scale = 0.0009765  # scale to convert int8 output to floating point, 1/2^10
    # int8_scale = 0.001953  # scale to convert int8 output to floating point, 1/2^9
    # int8_scale = 0.003906  # scale to convert int8 output to floating point, 1/2^8
    # int8_scale = 0.015625  # scale to convert int8 output to floating point, 1/2^6
    int8_scale = 0.03125  # scale to convert int8 output to floating point, 1/2^5

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
            self.conv = nn.Conv2d(
                in_channels=ci,
                out_channels=co,
                kernel_size=ksz,
                stride=ksz,
                padding=0,
                bias=False,
            )

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
    int_inp_np = int_inp.squeeze().data.numpy().astype(dtype_in)
    act_before_mem_fmt = int_inp_np.reshape(3584, 896)
    np.savetxt(
        log_folder + "/act_before_mem_fmt_CYX.txt",
        act_before_mem_fmt,
        fmt="%d",
        delimiter=",",
    )

    ifm_mem_fmt = ds.reorder_mat(int_inp_np, "YXC", "CYX")
    act_after_mem_fmt = ifm_mem_fmt.reshape(802816, 4)
    np.savetxt(
        log_folder + "/act_after_mem_fmt_YXC.txt",
        act_after_mem_fmt,
        fmt="%d",
        delimiter=",",
    )

    print("ifm_mem_fmt:")
    print(type(ifm_mem_fmt))
    print(ifm_mem_fmt.shape)

    ifm_mem_fmt_72 = np.tile(ifm_mem_fmt, co_group)
    print("ifm_mem_fmt_72:")
    print(ifm_mem_fmt_72.shape)

    int_weight_np = int_weight.data.numpy().astype(dtype_wts)
    weights_before_mem_fmt = int_weight_np.reshape(64512, 14)
    np.savetxt(
        log_folder + "/weights_before_mem_fmt_OIYX.txt",
        weights_before_mem_fmt,
        fmt="%d",
        delimiter=",",
    )

    wts1 = ds.reorder_mat(int_weight.data.numpy().astype(dtype_wts), "OYXIO8", "OIYX")
    total_wts = np.concatenate((wts1), axis=None)

    weights_after_mem_fmt = total_wts.reshape(112896, 8)
    np.savetxt(
        log_folder + "/weights_after_mem_fmt_OYXIO8.txt",
        weights_after_mem_fmt,
        fmt="%d",
        delimiter=",",
    )

    # ------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------
    for i in range(num_iter):
        start = time.time_ns()
        # entire_buffer = execute(app, ifm_mem_fmt, total_wts)
        entire_buffer = execute(app, ifm_mem_fmt_72, total_wts)
        stop = time.time_ns()

        if enable_trace:
            #  Separate data and trace
            data_buffer, trace_buffer = extract_trace(
                entire_buffer, shape_out, dtype_out, trace_size
            )
            # Scale the data
            scaled_data_buffer = data_buffer * int8_scale
            # Write out the trace
            write_out_trace(trace_buffer, trace_file)
        else:
            data_buffer = entire_buffer
            scaled_data_buffer = entire_buffer * int8_scale
            trace_buffer = None

        npu_time = stop - start
        npu_time_total = npu_time_total + npu_time

    # ------------------------------------------------------
    # Reorder output data-layout
    # ------------------------------------------------------
    temp_out = scaled_data_buffer.reshape(co_group, height_out, width_out, 16)
    temp_out = ds.reorder_mat(temp_out, "CDYX", "CYXD")
    ofm_mem_fmt = temp_out.reshape(co, height_out, width_out)

    temp_out_int = data_buffer.reshape(co_group, height_out, width_out, 16)
    temp_out_int = ds.reorder_mat(temp_out_int, "CDYX", "CYXD")
    ofm_int = temp_out_int.reshape(co * height_out * 8, 8).astype(np.int8)
    np.savetxt(log_folder + "/ofm_int_CYXX8.txt", ofm_int, fmt="%d", delimiter=",")

    if enable_trace:
        ofm_log_filename = "/ofm_after_mem_fmt_trace.txt"
    else:
        ofm_log_filename = "/ofm_after_mem_fmt.txt"

    ofm_float = ofm_mem_fmt.reshape(co * height_out * 8, 8)  # still in float
    np.savetxt(
        log_folder + "/ofm_float_CYXX8.txt", ofm_float, fmt="%.4f", delimiter=","
    )

    ofm_mem_fmt_out = torch.from_numpy(ofm_mem_fmt).unsqueeze(0)

    # ------------------------------------------------------
    # Compare the AIE output and the golden reference
    # ------------------------------------------------------

    print("\nAvg NPU time: {}us.".format(int((npu_time_total / num_iter) / 1000)))

    print("Weight")
    print(int_weight.size())
    print(int_weight)

    print("golden_output:")
    print(golden_output.shape)
    print(golden_output)

    print("ofm_mem_fmt_out:")
    print(ofm_mem_fmt_out.shape)
    print(ofm_mem_fmt_out)

    golden_output_2d = (
        golden_output.detach().numpy().reshape(co, height_out * width_out)
    )
    np.savetxt(
        log_folder + "/golden_output_2d.txt",
        golden_output_2d,
        fmt="%.4f",
        delimiter=",",
    )

    golden_output_int = (
        (golden_output.detach().numpy().reshape(co * 64 * 8, 8)) / int8_scale
    ).astype(int)
    np.savetxt(
        log_folder + "/golden_output_int.txt",
        golden_output_int,
        fmt="%d",
        delimiter=",",
    )

    output_numpy = ofm_mem_fmt_out.detach().numpy()
    golden_numpy = golden_output.detach().numpy()

    # Full testbench for vector kernel
    output_numpy_sub = output_numpy
    golden_numpy_sub = golden_numpy

    # Test passes only for submatrice in scalar example (co_group = 16)
    # output_numpy_sub = output_numpy[0:, 0:255, 0:, 0:]
    # golden_numpy_sub = golden_numpy[0:, 0:255, 0:, 0:]

    print("output_numpy_sub")
    print(output_numpy_sub.shape)

    print("golden_numpy_sub")
    print(golden_numpy_sub.shape)

    golden_numpy_sub_int = (golden_numpy_sub / int8_scale).astype(int)
    output_numpy_sub_int = (output_numpy_sub / int8_scale).astype(int)

    max_difference = np.max(np.abs(golden_numpy_sub_int - output_numpy_sub_int))
    print("max_abs_difference:", max_difference)

    avg_difference = np.average(np.abs(golden_numpy_sub_int - output_numpy_sub_int))
    print("avg_abs_difference:", avg_difference)

    print(
        "max golden int value: ",
        np.max(golden_numpy_sub_int),
        ", min golden int value: ",
        np.min(golden_numpy_sub_int),
    )

    # Find the indices where the mismatch happens
    mismatch_indices = np.where(golden_numpy_sub_int != output_numpy_sub_int)

    # Extract mismatch values
    mismatch_values_golden = golden_numpy_sub_int[mismatch_indices]
    mismatch_values_ofm = output_numpy_sub_int[mismatch_indices]

    # UNCOMMENT BELOW TO PRINT MISMATCHES
    # print("Mismatch indices and corresponding values:")
    # for idx, (golden_value, ofm_value) in zip(
    #     zip(*mismatch_indices), zip(mismatch_values_golden, mismatch_values_ofm)
    # ):
    #     print(f"Index: {idx}, Golden value: {golden_value}, OFM value: {ofm_value}, diff: {golden_value-ofm_value}")

    if np.allclose(
        output_numpy_sub,
        golden_numpy_sub,
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
    p.add_argument(
        "-ksz",
        "--kernel_size",
        dest="kernel_size",
        default=14,
        help="Size of kernel",
    )
    opts = p.parse_args(sys.argv[1:])
    main(opts)
