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

bneck_13_InW1 = 7
bneck_13_InH1 = 6
bneck_13_InC1 = 160
bneck_13_OutC1 = 960
WeightChunks=2 #2 splits for input channel and then output 

bneck_13_InW2 = bneck_13_InW1
bneck_13_InH2 = bneck_13_InH1
bneck_13_OutC2 = bneck_13_OutC1

bneck_13_InW3 = bneck_13_InW1
bneck_13_InH3 = bneck_13_InH1
bneck_13_OutC3 = 64

bneck_13_InC1_vec =  math.floor(bneck_13_InC1/vectorSize)
bneck_13_OutC2_vec =  math.floor(bneck_13_OutC2/vectorSize)


def chunk_weights_depth_cascade(int_weight, InC, WeightChunks):
    chunk_size = InC // WeightChunks
    chunks = []
    input_channels = int_weight.shape[1]
    output_channels = int_weight.shape[0]

    for i in range(WeightChunks):
        start_index = i * chunk_size
        end_index = input_channels if i == WeightChunks - 1 else (i + 1) * chunk_size
        for out_c_start in range(0, output_channels, 8):
            out_c_end = min(out_c_start + 8, output_channels)
            chunk = int_weight[out_c_start:out_c_end, start_index:end_index, :, :]
            # print("oc={}:{},ic={}:{}".format(out_c_start,out_c_end,start_index,end_index))
            chunks.append(chunk)
    return chunks

def reorder_and_concatenate_chunks(int_weight, InC, WeightChunks, ds, dtype_wts):
    # Chunk the weights
    chunks = chunk_weights_depth_cascade(int_weight, InC, WeightChunks)
    
    # Reorder each chunk
    reordered_chunks = []
    for idx, chunk in enumerate(chunks):
        reordered_chunk = ds.reorder_mat(chunk.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX")
        reordered_chunks.append(reordered_chunk)
    
    # Concatenate the reordered chunks
    total_wts = np.concatenate(reordered_chunks, axis=None)
    print(int_weight.shape)
    print(total_wts.shape)
    
    return total_wts



wts_size=(bneck_13_OutC1*bneck_13_InC1)
# wts_size=(bneck_13_OutC1*bneck_13_InC1)+(3*3*bneck_13_OutC2)

def main(opts):
    design = "mobilenet_bottleneck_C"
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
    print(wts_size)
    shape_total_wts = (wts_size, 1)
    shape_in_act = (bneck_13_InH1, bneck_13_InC1_vec, bneck_13_InW1, vectorSize)  #'YCXC8' , 'CYX'
    shape_out = (bneck_13_InH1, bneck_13_OutC2_vec, bneck_13_InW1, vectorSize) #bneck_12_OutC3/8
    shape_out_final = (bneck_13_OutC2_vec*vectorSize, bneck_13_InH1, bneck_13_InW1) #bneck_12_OutC3/8
    
    # ------------------------------------------------------
    # Initialize activation, weights, scaling factor for int8 model
    # ------------------------------------------------------
    input = torch.randn(1, bneck_13_InC1_vec*vectorSize, bneck_13_InH1, bneck_13_InW1)
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
        def __init__(self, in_planes=16, bn13_expand=16,bn13_project=16,bn14_expand=16,bn11_project=16,bn12_expand=16,bn12_project=16):
            super(QuantBottleneck, self).__init__()
            self.quant_id_1 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

            self.bn13_quant_conv1 = QuantConv2d(
                in_planes,
                bn13_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn13_quant_conv2 = QuantConv2d(
                bn13_expand,
                bn13_expand,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn13_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn13_quant_conv3 = QuantConv2d(
                bn13_expand,
                bn13_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn13_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn13_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn13_add = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

        def forward(self, x):
            out_q = self.quant_id_1(x)
            out = self.bn13_quant_conv1(out_q)
            out = self.bn13_quant_relu1(out)
            # out = self.bn13_quant_conv2(out)
            # out = self.bn13_quant_relu2(out)
            # out = self.bn13_quant_conv3(out)
            # out = self.quant_id_1(out)
            # out=out+out_q
            # out = self.bn13_add(out)
            return out

    quant_bottleneck_model = QuantBottleneck(in_planes=bneck_13_InC1, bn13_expand=bneck_13_OutC2,bn13_project=bneck_13_OutC3)
    quant_bottleneck_model.eval()
    
    q_bottleneck_out = quant_bottleneck_model(input)
    golden_output = q_bottleneck_out.int(float_datatype=True).data.numpy().astype(dtype_out)
    print("Golden::Brevitas::", golden_output)
    q_inp = quant_bottleneck_model.quant_id_1(input)
    int_inp = q_inp.int(float_datatype=True)


 
    block_13_inp_scale1= quant_bottleneck_model.quant_id_1.quant_act_scale()

    block_13_relu_1 = quant_bottleneck_model.bn13_quant_relu1.quant_act_scale()
    block_13_relu_2 = quant_bottleneck_model.bn13_quant_relu2.quant_act_scale()
    block_13_skip_add = quant_bottleneck_model.bn13_add.quant_act_scale()

    block_13_weight_scale1 = quant_bottleneck_model.bn13_quant_conv1.quant_weight_scale()
    block_13_weight_scale2 = quant_bottleneck_model.bn13_quant_conv2.quant_weight_scale()
    block_13_weight_scale3 = quant_bottleneck_model.bn13_quant_conv3.quant_weight_scale()
    block_13_combined_scale1 = -torch.log2(
        block_13_inp_scale1 * block_13_weight_scale1 / block_13_relu_1
    )
    block_13_combined_scale2 = -torch.log2(
        block_13_relu_1 * block_13_weight_scale2 / block_13_relu_2
    )  
    block_13_combined_scale3 = -torch.log2(
        block_13_relu_2 * block_13_weight_scale3/block_13_inp_scale1
    )   
    block_13_combined_scale_skip = -torch.log2(
        block_13_inp_scale1 / block_13_skip_add
    )  # After addition | clip -128-->127



    print("********************BN13*******************************")
    print("combined_scale after conv1x1:", block_13_combined_scale1.item())
    print("combined_scale after conv3x3:", block_13_combined_scale2.item())
    print("combined_scale after conv1x1:", block_13_combined_scale3.item())
    print("combined_scale after skip add:", block_13_combined_scale_skip.item())
    print("********************BN12*******************************")

    # print("combined_scale after conv1x1:", ( block_0_relu_2 * block_0_weight_scale3).item())
    # ------------------------------------------------------
    # Reorder input data-layout
    # ------------------------------------------------------

    block_13_int_weight_1 = quant_bottleneck_model.bn13_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_13_int_weight_2 = quant_bottleneck_model.bn13_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_13_int_weight_3 = quant_bottleneck_model.bn13_quant_conv3.quant_weight().int(
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
    if(bneck_13_InW1>1 and bneck_13_InH1==1):
        ifm_mem_fmt = ds.reorder_mat(before_input, "CXC8", "CX")
    elif(bneck_13_InW1>1 and bneck_13_InH1>1):
            ifm_mem_fmt = ds.reorder_mat(before_input, "YCXC8", "CYX")

    else:
        ifm_mem_fmt = ds.reorder_mat(before_input, "CC8", "C")
    ifm_mem_fmt.tofile(log_folder + "/after_ifm_mem_fmt.txt", sep=",", format="%d")

    # **************************** bn13 ****************************
    # bn13_wts1 = ds.reorder_mat(
    #     block_13_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    # )
    bn13_wts1 = reorder_and_concatenate_chunks(block_13_int_weight_1, bneck_13_InC1, WeightChunks, ds, dtype_wts)
    bn13_wts2 = ds.reorder_mat(
        block_13_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn13_wts3 = ds.reorder_mat(
        block_13_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )


    total_wts = np.concatenate((bn13_wts1,), axis=None)


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
        atol=1,
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