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
from dolphin import print_dolphin
from brevitas.nn import QuantConv2d, QuantIdentity, QuantReLU
from brevitas.quant.fixed_point import (
    Int8ActPerTensorFixedPoint,
    Int8WeightPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
vectorSize=8

tensorInW = 112
tensorInH = 112 
tensorInC = 16
tensorOutC = 24

depthwiseStride = 2
depthWiseChannels = 64

bneck_0_InW2 = tensorInW
bneck_0_InH2 = tensorInH
bneck_0_InC2 = tensorInC
bneck_0_OutC2 = bneck_0_InC2

bneck_0_InW3 = bneck_0_InW2
bneck_0_InH3 = bneck_0_InH2
bneck_0_InC3 = bneck_0_OutC2
bneck_0_OutC3 = bneck_0_InC3

bneck_0_InC2_vec =  math.floor(bneck_0_InC2/vectorSize)
bneck_0_OutC3_vec =  math.floor(bneck_0_OutC3/vectorSize)

bneck_1_InW1 = tensorInW
bneck_1_InH1 = tensorInH
bneck_1_InC1 = tensorInC
bneck_1_OutC1 = depthWiseChannels

bneck_1_InW2 = bneck_1_InW1 
bneck_1_InH2 = bneck_1_InH1 
bneck_1_OutC2 = bneck_1_OutC1

bneck_1_InW3 = bneck_1_InW2 // depthwiseStride
bneck_1_InH3 = bneck_1_InH2 // depthwiseStride
bneck_1_OutC3 = tensorOutC

#bneck_1_InC1_vec =  math.floor(bneck_1_InC1/vectorSize)
bneck_1_OutC3_vec =  math.floor(bneck_1_OutC3/vectorSize)

tmp_outC_vec = math.floor(depthWiseChannels/vectorSize)

def main(opts):
    design = "mobilenet_bottleneck_A_bn0_bn1"
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

    shape_total_wts = (3*3*bneck_0_OutC2 + bneck_0_OutC2*bneck_0_OutC3 + bneck_1_InC1*bneck_1_OutC1 + 3*3*bneck_1_OutC2 + bneck_1_OutC2*bneck_1_OutC3, 1)
    shape_in_act = (bneck_0_InH2, bneck_0_InC2_vec, bneck_0_InW2, vectorSize)  #'YCXC8' , 'CYX'
    shape_out = (bneck_1_InH3, bneck_1_OutC3_vec, bneck_1_InW3, vectorSize) # HCWC8
    shape_out_final = (bneck_1_OutC3_vec*vectorSize, bneck_1_InH3, bneck_1_InW3) # CHW


    # ------------------------------------------------------
    # Initialize activation, weights, scaling factor for int8 model
    # ------------------------------------------------------
    input = torch.randn(1, bneck_0_InC2_vec*vectorSize, bneck_0_InH2, bneck_0_InW2)
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
    class QuantBottleneck0(nn.Module):
        def __init__(self, in_planes=16, bn0_expand=16,bn0_project=16, bn1_expand=16,bn1_project=16):
            super(QuantBottleneck0, self).__init__()
            self.quant_id_1 = QuantIdentity(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

            self.bn0_quant_conv2 = QuantConv2d(
                bn0_expand,
                bn0_expand,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn0_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn0_quant_conv3 = QuantConv2d(
                bn0_expand,
                bn0_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            
            self.bn0_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

            self.bn0_quant_id_2 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

            self.bn0_add = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

            # force alignment between scales going into add
            self.bn0_quant_id_2.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl = self.quant_id_1.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl
            self.bn0_quant_id_2.act_quant.fused_activation_quant_proxy.tensor_quant.int_scaling_impl = self.quant_id_1.act_quant.fused_activation_quant_proxy.tensor_quant.int_scaling_impl

            self.bn1_quant_conv1 = QuantConv2d(
                in_planes,
                bn1_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn1_quant_conv2 = QuantConv2d(
                bn1_expand,
                bn1_expand,
                kernel_size=3,
                stride=depthwiseStride,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn1_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn1_quant_conv3 = QuantConv2d(
                bn1_expand,
                bn1_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn1_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn1_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn1_quant_id_2 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

        def forward(self, x):
            out_q = self.quant_id_1(x)
            out = self.bn0_quant_conv2(out_q)
            out = self.bn0_quant_relu2(out)
            out = self.bn0_quant_conv3(out)
            out = self.bn0_quant_id_2(out)
            out = out+out_q
            out = self.bn0_add(out)

            out = self.bn1_quant_conv1(out)
            out = self.bn1_quant_relu1(out)
            out = self.bn1_quant_conv2(out)
            out = self.bn1_quant_relu2(out)
            out = self.bn1_quant_conv3(out)
            out = self.bn1_quant_id_2(out)
            return out

    quant_bottleneck_model = QuantBottleneck0(in_planes=bneck_0_InC2, bn0_expand=bneck_0_InC2,bn0_project=bneck_0_OutC3, bn1_expand=bneck_1_OutC1,bn1_project=bneck_1_OutC3)
    quant_bottleneck_model.eval()
    
    q_bottleneck_out = quant_bottleneck_model(input)
    golden_output = q_bottleneck_out.int(float_datatype=True).data.numpy().astype(dtype_out)
    print("Golden::Brevitas::", golden_output)
    q_inp = quant_bottleneck_model.quant_id_1(input)
    int_inp = q_inp.int(float_datatype=True)
 
    block_0_inp_scale= quant_bottleneck_model.quant_id_1.quant_act_scale()

    block_0_relu_2 = quant_bottleneck_model.bn0_quant_relu2.quant_act_scale()
    block_0_skip_add = quant_bottleneck_model.bn0_add.quant_act_scale()

    block_0_weight_scale2 = quant_bottleneck_model.bn0_quant_conv2.quant_weight_scale()
    block_0_weight_scale3 = quant_bottleneck_model.bn0_quant_conv3.quant_weight_scale()

    block_0_combined_scale2 = -torch.log2(
        block_0_inp_scale * block_0_weight_scale2 / block_0_relu_2
    )  
    block_0_combined_scale3 = -torch.log2(
        block_0_relu_2 * block_0_weight_scale3/block_0_inp_scale
    )   
    block_0_combined_scale_skip = -torch.log2(
        block_0_inp_scale / block_0_skip_add
    )  # After addition | clip -128-->127

    block_1_relu_1 = quant_bottleneck_model.bn1_quant_relu1.quant_act_scale()
    block_1_relu_2 = quant_bottleneck_model.bn1_quant_relu2.quant_act_scale()
    block_1_final_scale = quant_bottleneck_model.bn1_quant_id_2.quant_act_scale()

    block_1_weight_scale1 = quant_bottleneck_model.bn1_quant_conv1.quant_weight_scale()
    block_1_weight_scale2 = quant_bottleneck_model.bn1_quant_conv2.quant_weight_scale()
    block_1_weight_scale3 = quant_bottleneck_model.bn1_quant_conv3.quant_weight_scale()
    block_1_combined_scale1 = -torch.log2(
        block_0_skip_add * block_1_weight_scale1 / block_1_relu_1
        
    )
    block_1_combined_scale2 = -torch.log2(
        block_1_relu_1 * block_1_weight_scale2 / block_1_relu_2
    )  
    block_1_combined_scale3 = -torch.log2(
        block_1_relu_2 * block_1_weight_scale3/block_1_final_scale
    )

    print("********************BN0*******************************")
    print("combined_scale after conv3x3:", block_0_combined_scale2.item())
    print("combined_scale after conv1x1:", block_0_combined_scale3.item())
    print("combined_scale after skip add:", block_0_combined_scale_skip.item())
    print("********************BN0*******************************")

    print("********************BN1*******************************")
    print("combined_scale after conv1x1:", block_1_combined_scale1.item())
    print("combined_scale after conv3x3:", block_1_combined_scale2.item())
    print("combined_scale after conv1x1:", block_1_combined_scale3.item())
    print("********************BN1*******************************")
    
    # ------------------------------------------------------
    # Reorder input data-layout
    # ------------------------------------------------------

    
    block_0_int_weight_2 = quant_bottleneck_model.bn0_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_0_int_weight_3 = quant_bottleneck_model.bn0_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    block_1_int_weight_1 = quant_bottleneck_model.bn1_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_1_int_weight_2 = quant_bottleneck_model.bn1_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_1_int_weight_3 = quant_bottleneck_model.bn1_quant_conv3.quant_weight().int(
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

    # **************************** bn0 ****************************
    bn0_wts2 = ds.reorder_mat(
        block_0_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn0_wts3 = ds.reorder_mat(
        block_0_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    # **************************** bn1 ****************************
    bn1_wts1 = ds.reorder_mat(
        block_1_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn1_wts2 = ds.reorder_mat(
        block_1_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn1_wts3 = ds.reorder_mat(
        block_1_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )

    total_wts = np.concatenate((bn0_wts2, bn0_wts3, bn1_wts1, bn1_wts2, bn1_wts3), axis=None)

    total_wts.tofile(log_folder + "/after_weights_mem_fmt_final.txt", sep=",", format="%d")
    print(total_wts.shape)
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
    temp_out = aie_output.reshape(shape_out)
    temp_out = ds.reorder_mat(temp_out, "CDYX", "YCXD")
    ofm_mem_fmt = temp_out.reshape(shape_out_final)
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
        atol=3,
    ):
        print("\nPASS!\n")
        print_dolphin()
        exit(0)
    else:
        print("\nFailed.\n")
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)