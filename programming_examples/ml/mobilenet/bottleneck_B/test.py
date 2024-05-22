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
vectorSize=8

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

kdim=3
stride=1
padding=1

# tensorOutW = math.floor((tensorInW-kdim+2*padding)/stride)+1
# tensorOutH = math.floor((tensorInH-kdim+2*padding)/stride)+1

# tensorL2InC = tensorInC
# tensorL2OutC = tensorInC

def main(opts):
    design = "mobilenet_bottleneck_B"
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
    dtype_out = np.dtype("int8")

    shape_total_wts = (96480, 1)
    shape_in_act = (bneck_10_InH1, 10, bneck_10_InW1, vectorSize)  #'YCXC8' , 'CYX'
    shape_out = (bneck_10_InH3, 14, bneck_10_InW3, vectorSize)
  
    
    # ------------------------------------------------------
    # Initialize activation, weights, scaling factor for int8 model
    # ------------------------------------------------------
    input = torch.randn(1, bneck_10_InC1, bneck_10_InH1, bneck_10_InW1)
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
    class QuantBottleneck(nn.Module):
        def __init__(self, in_planes=16, expand=16,project=16):
            super(QuantBottleneck, self).__init__()
            self.quant_id_1 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.quant_conv1 = QuantConv2d(
                in_planes,
                expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.quant_conv2 = QuantConv2d(
                expand,
                expand,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.quant_conv3 = QuantConv2d(
                expand,
                project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

        def forward(self, x):
            out_q = self.quant_id_1(x)
            out = self.quant_conv1(out_q)
            out = self.quant_relu1(out)
            out = self.quant_conv2(out)
            out = self.quant_relu2(out)
            out = self.quant_conv3(out)
            return out

    quant_bottleneck_model = QuantBottleneck(in_planes=80, expand=480,project=112)
    quant_bottleneck_model.eval()
    
    q_bottleneck_out = quant_bottleneck_model(input)
    golden_output = q_bottleneck_out.int(float_datatype=True).data.numpy().astype(dtype_out)
    print("Golden::Brevitas::", golden_output)
    q_inp = quant_bottleneck_model.quant_id_1(input)
    int_inp = q_inp.int(float_datatype=True)
    # print(input.shape)
    # print(int_weight.shape)
    # print(q_bottleneck_out.shape)

    init_scale = quant_bottleneck_model.quant_id_1.quant_act_scale()
    block_0_relu_1 = quant_bottleneck_model.quant_relu1.quant_act_scale()
    block_0_relu_2 = quant_bottleneck_model.quant_relu2.quant_act_scale()

    block_0_weight_scale1 = quant_bottleneck_model.quant_conv1.quant_weight_scale()
    block_0_weight_scale2 = quant_bottleneck_model.quant_conv2.quant_weight_scale()
    block_0_weight_scale3 = quant_bottleneck_model.quant_conv3.quant_weight_scale()
    block_0_combined_scale1 = -torch.log2(
        init_scale * block_0_weight_scale1 / block_0_relu_1
    )
    block_0_combined_scale2 = -torch.log2(
        block_0_relu_1 * block_0_weight_scale2 / block_0_relu_2
    )  
    block_0_combined_scale3 = -torch.log2(
        block_0_relu_2 * block_0_weight_scale3
    )   
    print("combined_scale after conv1x1:", block_0_combined_scale1.item())
    print("combined_scale after conv3x3:", block_0_combined_scale2.item())
    print("combined_scale after conv1x1:", block_0_combined_scale3.item())
    # ------------------------------------------------------
    # Reorder input data-layout
    # ------------------------------------------------------
    block_0_int_weight_1 = quant_bottleneck_model.quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_0_int_weight_2 = quant_bottleneck_model.quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_0_int_weight_3 = quant_bottleneck_model.quant_conv3.quant_weight().int(
        float_datatype=True
    )
  
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
    # print(int_weight.shape)
    wts1 = ds.reorder_mat(
        block_0_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    wts2 = ds.reorder_mat(
        block_0_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    wts3 = ds.reorder_mat(
        block_0_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    total_wts = np.concatenate((wts1, wts2, wts3), axis=None)
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
    temp_out = aie_output.reshape(bneck_10_InH3, 14, bneck_10_InW3, vectorSize)
    temp_out = ds.reorder_mat(temp_out, "CDYX", "YCXD")
    ofm_mem_fmt = temp_out.reshape(bneck_10_OutC3, bneck_10_InH3, bneck_10_InW3)
    ofm_mem_fmt.tofile(
        log_folder + "/after_ofm_mem_fmt_final.txt", sep=",", format="%d"
    )
    ofm_mem_fmt_out = torch.from_numpy(ofm_mem_fmt).unsqueeze(0)
    print(ofm_mem_fmt_out)
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
