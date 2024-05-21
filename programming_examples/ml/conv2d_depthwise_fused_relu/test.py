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
from brevitas.nn import QuantConv2d, QuantIdentity, QuantReLU
from brevitas.quant.fixed_point import (
    Int8ActPerTensorFixedPoint,
    Int8WeightPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)


def main(opts):
    design = "conv2d_depthwise_with_relu"
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
    dtype_in = np.dtype("uint8")
    dtype_wts = np.dtype("int8")
    dtype_out = np.dtype("uint8")

    shape_total_wts = (144, 1)
    shape_in_act = (7, 2, 7, 8)  #'YCXC8' , 'CYX'
    shape_in_wts1 = (1, 2, 3, 3, 8, 1)  # out,in,ky,kx,in8,out8
    shape_out = (7, 2, 7, 8)
    quant_conv1 = QuantConv2d(
        16,
        16,
        kernel_size=3,
        padding=1,
        padding_mode="zeros",
        bit_width=8,
        weight_bit_width=8,
        groups=16,
        bias=False,
        weight_quant=Int8WeightPerTensorFixedPoint,
        return_quant_tensor=True,
    )
    quant_conv1.eval()
    quant_id_1 = QuantIdentity(
        act_quant=Uint8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
    )
    quant_relu1 = QuantReLU(
        act_quant=Uint8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
    )
    
    
    # ------------------------------------------------------
    # Initialize activation, weights, scaling factor for int8 model
    # ------------------------------------------------------
    input = torch.randn(1, 16, 7, 7)


    # int_inp = torch.randint(1, 100, (1, 64, 7, 7)).type(torch.FloatTensor)
    
    # int_weight = torch.randint(50, 100, (64, 64, 1, 1)).type(torch.FloatTensor)
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
    class QuantConv2d3x3(nn.Module):
        def __init__(self, in_planes=16, planes=16):
            super(QuantConv2d3x3, self).__init__()
            self.quant_id_1 = QuantIdentity(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.conv1 = QuantConv2d(
                in_planes,
                planes,
                kernel_size=3,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=in_planes,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

        def forward(self, x):
            out_q = self.quant_id_1(x)
            out = self.conv1(out_q)
            out = self.relu1(out)
            return out

    quant_bottleneck_model = QuantConv2d3x3()
    quant_bottleneck_model.eval()
    int_weight = quant_bottleneck_model.conv1.quant_weight().int(float_datatype=True)
    q_bottleneck_out = quant_bottleneck_model(input)
    golden_output = q_bottleneck_out.int(float_datatype=True).data.numpy().astype(dtype_out)

    q_inp = quant_bottleneck_model.quant_id_1(input)
    int_inp = q_inp.int(float_datatype=True)
   
    inp_scale1 = quant_bottleneck_model.quant_id_1.quant_act_scale()
    inp_scale2 = quant_bottleneck_model.relu1.quant_act_scale()
    weight_scale1 = quant_bottleneck_model.conv1.quant_weight_scale()
    combined_scale1 = -torch.log2(inp_scale1 * weight_scale1 / inp_scale2)
    
    # print("combined_scale after first conv3x3:", combined_scale1.item())
    # ------------------------------------------------------
    # Reorder input data-layout
    # ------------------------------------------------------
    golden_output.tofile(
        log_folder + "/golden_output.txt", sep=",", format="%d"
    )
    ds = DataShaper()
    before_input = int_inp.squeeze().data.numpy().astype(dtype_in)
    before_input.tofile(
        log_folder + "/before_ifm_mem_fmt_1x1.txt", sep=",", format="%d"
    )
    ifm_mem_fmt = ds.reorder_mat(before_input, "YCXC8", "CYX")
    ifm_mem_fmt.tofile(log_folder + "/after_ifm_mem_fmt.txt", sep=",", format="%d")

    int_weight.data.numpy().astype(dtype_wts).tofile(log_folder + "/before_weights_mem_fmt_final.txt", sep=",", format="%d")
    wts1 = ds.reorder_mat(int_weight.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX")
    total_wts = np.concatenate((wts1), axis=None)
  
    total_wts.tofile(log_folder + "/after_weights_mem_fmt_final.txt", sep=",", format="%d")

    # ------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------
    for i in range(num_iter):
        start = time.time_ns()
        aie_output = execute(app, ifm_mem_fmt, total_wts) 
        stop = time.time_ns()
        npu_time = stop - start
        npu_time_total = npu_time_total + npu_time

    # ------------------------------------------------------
    # Reorder output data-layout
    # ------------------------------------------------------
    temp_out = aie_output.reshape(7, 2, 7, 8)
    temp_out = ds.reorder_mat(temp_out, "CDYX", "YCXD")
    ofm_mem_fmt = temp_out.reshape(16, 7, 7)
    ofm_mem_fmt.tofile(
        log_folder + "/after_ofm_mem_fmt_final.txt", sep=",", format="%d"
    )
    ofm_mem_fmt_out = torch.from_numpy(ofm_mem_fmt).unsqueeze(0)
    # ------------------------------------------------------
    # Compare the AIE output and the golden reference
    # ------------------------------------------------------
    print("\nAvg NPU time: {}us.".format(int((npu_time_total / num_iter) / 1000)))

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
    opts = p.parse_args(sys.argv[1:])
    main(opts)
