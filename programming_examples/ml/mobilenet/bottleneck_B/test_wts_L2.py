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
from aie.utils.xrt import setup_aie, extract_trace, write_out_trace, execute,execute_inference,write_wts
import aie.utils.test as test_utils
from dolphin import print_dolphin
from brevitas.nn import QuantConv2d, QuantIdentity, QuantReLU
from brevitas.quant.fixed_point import (
    Int8ActPerTensorFixedPoint,
    Int8WeightPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)
def convert_to_numpy(array):
    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    else:
        raise TypeError("Unsupported array type")
    
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)


import json

# Function to read scale factors from JSON file
def read_scale_factors(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
# Function to write scale factors to JSON file
def write_scale_factors(file_path, scale_factors):
    with open(file_path, 'w') as file:
        json.dump(scale_factors, file, indent=4)

# Read the existing scale factors
file_path = 'scale_factors.json'
scale_factors = read_scale_factors(file_path)

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

bneck_11_OutC1 = 336
bneck_11_OutC2 = 336
bneck_11_OutC3 = 112
kdim=3
stride=1
padding=1

bneck_12_InH1 = 14
bneck_12_InW1 = 14
bneck_12_OutC1 = 336

bneck_12_InH2 = 14
bneck_12_InW2 = 14
bneck_12_OutC2 = 336

bneck_12_InW3 = 7
bneck_12_InH3 = 7
bneck_12_OutC3 = 80

OutC=bneck_12_OutC3
OutH=bneck_12_InH3
OutW=bneck_12_InW3

OutC_vec =  math.floor(OutC/vectorSize)

InC_vec =  math.floor(bneck_10_InC1/vectorSize)
wts_size=((bneck_10_InC1*bneck_10_OutC1)+(3*3*bneck_10_OutC2)+(bneck_10_OutC2*bneck_10_OutC3)+
            (bneck_10_OutC3*bneck_11_OutC1)+(3*3*bneck_11_OutC2)+(bneck_11_OutC2*bneck_11_OutC3)+
            (bneck_11_OutC3*bneck_12_OutC1)+(3*3*bneck_12_OutC2)+(bneck_12_OutC2*bneck_12_OutC3))

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

    shape_total_wts = (wts_size, 1)
    shape_in_act = (bneck_10_InH1, InC_vec, bneck_10_InW1, vectorSize)  #'YCXC8' , 'CYX'
    shape_out = (OutH, OutC_vec, OutW, vectorSize) #bneck_12_OutC3/8
    shape_out_final = (OutC_vec*vectorSize, OutH, OutW) #bneck_12_OutC3/8
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
        def __init__(self, in_planes=16, bn10_expand=16,bn10_project=16,bn11_expand=16,bn11_project=16,bn12_expand=16,bn12_project=16):
            super(QuantBottleneck, self).__init__()
            self.quant_id_1 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn10_quant_conv1 = QuantConv2d(
                in_planes,
                bn10_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn10_quant_conv2 = QuantConv2d(
                bn10_expand,
                bn10_expand,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn10_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn10_quant_conv3 = QuantConv2d(
                bn10_expand,
                bn10_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn10_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn10_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn10_quant_id_2 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
# bn11

            self.bn11_quant_conv1 = QuantConv2d(
                bn10_project,
                bn11_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn11_quant_conv2 = QuantConv2d(
                bn11_expand,
                bn11_expand,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn11_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn11_quant_conv3 = QuantConv2d(
                bn11_expand,
                bn11_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn11_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn11_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            # self.bn11_quant_id_2 = QuantIdentity(
            #     act_quant=Int8ActPerTensorFixedPoint,
            #     bit_width=8,
            #     return_quant_tensor=True,
            # )
            self.bn11_add = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

            self.bn11_quant_id_2 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
# bn12
# # force alignment between scales going into add
#             self.bn10_quant_id_2.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl = self.bn11_quant_id_2.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl
#             self.bn10_quant_id_2.act_quant.fused_activation_quant_proxy.tensor_quant.int_scaling_impl = self.bn11_quant_id_2.act_quant.fused_activation_quant_proxy.tensor_quant.int_scaling_impl


            self.bn12_quant_conv1 = QuantConv2d(
                bn11_project,
                bn12_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn12_quant_conv2 = QuantConv2d(
                bn12_expand,
                bn12_expand,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn12_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn12_quant_conv3 = QuantConv2d(
                bn12_expand,
                bn12_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn12_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn12_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn12_quant_id_2 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
        def forward(self, x):
            out = self.quant_id_1(x)
            out = self.bn10_quant_conv1(out)
            out = self.bn10_quant_relu1(out)
            out = self.bn10_quant_conv2(out)
            out = self.bn10_quant_relu2(out)
            out = self.bn10_quant_conv3(out)
            out_lhs = self.bn10_quant_id_2(out)
            # bn11
            out = self.bn11_quant_conv1(out_lhs)
            out = self.bn11_quant_relu1(out)
            out = self.bn11_quant_conv2(out)
            out = self.bn11_quant_relu2(out)
            out = self.bn11_quant_conv3(out)
            out = self.bn10_quant_id_2(out)
            out = out+out_lhs
            out = self.bn11_add(out)
            # bn12
            out = self.bn12_quant_conv1(out)
            out = self.bn12_quant_relu1(out)
            out = self.bn12_quant_conv2(out)
            out = self.bn12_quant_relu2(out)
            out = self.bn12_quant_conv3(out)
            out = self.bn12_quant_id_2(out)
          
            return out

    quant_bottleneck_model = QuantBottleneck(in_planes=bneck_10_InC1, bn10_expand=bneck_10_OutC2,bn10_project=bneck_10_OutC3, 
                                             bn11_expand=bneck_11_OutC2,bn11_project=bneck_11_OutC3, bn12_expand=bneck_12_OutC2,bn12_project=bneck_12_OutC3)
    quant_bottleneck_model.eval()
    
    from utils import ExpandChannels
    from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
    import torchvision
    import torch.utils.data as data_utils
    from torchvision import transforms
    # Define the image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ExpandChannels(target_channels=bneck_10_InC1)  # Expand to 80 channels
    ])
    data_dir = "data"
    
    # test_dataset = torchvision.datasets.ImageNet(
    #     root=data_dir, train=False, transform=transform, download=True)
    
    # # Create a subset and DataLoader for the single image
    # indices = torch.arange(32)
    # val_sub = data_utils.Subset(test_dataset, indices)
    # calib_loader = torch.utils.data.DataLoader(dataset=val_sub, batch_size=32, shuffle=False)
    
    src_data="/group/xrlabs2/imagenet/calibration"
    datset=torchvision.datasets.ImageFolder(
        src_data,
        transform)
    indices = torch.arange(4)
    val_sub = data_utils.Subset(datset, indices)
    calib_loader = torch.utils.data.DataLoader(dataset=val_sub, batch_size=32, shuffle=False)
    calibrate(calib_loader, quant_bottleneck_model)
    quant_bottleneck_model.eval()
    for name, param in quant_bottleneck_model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
    from brevitas.fx import brevitas_symbolic_trace
    # model = brevitas_symbolic_trace(quant_bottleneck_model)
    # print(model.graph)
    # print(model)

    q_bottleneck_out = quant_bottleneck_model(input)
    golden_output = q_bottleneck_out.int(float_datatype=True).data.numpy().astype(dtype_out)
    # print("Golden::Brevitas::", golden_output)
    q_inp = quant_bottleneck_model.quant_id_1(input)
    int_inp = q_inp.int(float_datatype=True)
    # print(input.shape)
    # print(int_weight.shape)
    # print(q_bottleneck_out)


    init_scale = quant_bottleneck_model.quant_id_1.quant_act_scale()
    block_10_relu_1 = quant_bottleneck_model.bn10_quant_relu1.quant_act_scale()
    block_10_relu_2 = quant_bottleneck_model.bn10_quant_relu2.quant_act_scale()
    block_10_final_scale = quant_bottleneck_model.bn10_quant_id_2.quant_act_scale()

    block_10_weight_scale1 = quant_bottleneck_model.bn10_quant_conv1.quant_weight_scale()
    block_10_weight_scale2 = quant_bottleneck_model.bn10_quant_conv2.quant_weight_scale()
    block_10_weight_scale3 = quant_bottleneck_model.bn10_quant_conv3.quant_weight_scale()

    block_10_combined_scale1 = -torch.log2(
        init_scale * block_10_weight_scale1 / block_10_relu_1
    )
    block_10_combined_scale2 = -torch.log2(
        block_10_relu_1 * block_10_weight_scale2 / block_10_relu_2
    )  
    block_10_combined_scale3 = -torch.log2(
        block_10_relu_2 * block_10_weight_scale3/block_10_final_scale
    )   
    block_11_relu_1 =       quant_bottleneck_model.bn11_quant_relu1.quant_act_scale()
    block_11_relu_2 =       quant_bottleneck_model.bn11_quant_relu2.quant_act_scale()
    block_11_skip_add =     quant_bottleneck_model.bn11_add.quant_act_scale()
    block_11_final_scale =  quant_bottleneck_model.bn11_quant_id_2.quant_act_scale()

    block_11_weight_scale1 = quant_bottleneck_model.bn11_quant_conv1.quant_weight_scale()
    block_11_weight_scale2 = quant_bottleneck_model.bn11_quant_conv2.quant_weight_scale()
    block_11_weight_scale3 = quant_bottleneck_model.bn11_quant_conv3.quant_weight_scale()
    
    block_11_combined_scale1 = -torch.log2(
        block_10_final_scale * block_11_weight_scale1 / block_11_relu_1
    )
    block_11_combined_scale2 = -torch.log2(
        block_11_relu_1 * block_11_weight_scale2 / block_11_relu_2
    )  
    block_11_combined_scale3 = -torch.log2(
        block_11_relu_2 * block_11_weight_scale3/block_10_final_scale
    )   
    block_11_combined_scale_skip = -torch.log2(
        block_10_final_scale / block_11_skip_add
    )  # After addition | clip -128-->127

    block_12_relu_1 =        quant_bottleneck_model.bn12_quant_relu1.quant_act_scale()
    block_12_relu_2 =        quant_bottleneck_model.bn12_quant_relu2.quant_act_scale()
    block_12_final_scale =   quant_bottleneck_model.bn12_quant_id_2.quant_act_scale()

    block_12_weight_scale1 = quant_bottleneck_model.bn12_quant_conv1.quant_weight_scale()
    block_12_weight_scale2 = quant_bottleneck_model.bn12_quant_conv2.quant_weight_scale()
    block_12_weight_scale3 = quant_bottleneck_model.bn12_quant_conv3.quant_weight_scale()

    block_12_combined_scale1 = -torch.log2(
        block_11_skip_add * block_12_weight_scale1 / block_12_relu_1
    )
    block_12_combined_scale2 = -torch.log2(
        block_12_relu_1 * block_12_weight_scale2 / block_12_relu_2
    )  
    block_12_combined_scale3 = -torch.log2(
        block_12_relu_2 * block_12_weight_scale3/block_12_final_scale
    )   
    
    atol=block_12_final_scale.item()


    print("********************BN10*******************************")
    print("combined_scale after conv1x1:", block_10_combined_scale1.item())
    print("combined_scale after conv3x3:", block_10_combined_scale2.item())
    print("combined_scale after conv1x1:", block_10_combined_scale3.item())

    scale_factors["BN10"]["conv1x1_1"] = int(block_10_combined_scale1.item())
    scale_factors["BN10"]["conv3x3"] = int(block_10_combined_scale2.item() )
    scale_factors["BN10"]["conv1x1_2"] = int(block_10_combined_scale3.item())
    
    print("********************BN11*******************************")
    print("combined_scale after conv1x1:", block_11_combined_scale1.item())
    print("combined_scale after conv3x3:", block_11_combined_scale2.item())
    print("combined_scale after conv1x1:", block_11_combined_scale3.item())
    print("combined_scale after skip add:", block_11_combined_scale_skip.item())

    scale_factors["BN11"]["conv1x1_1"] = int(block_11_combined_scale1.item())
    scale_factors["BN11"]["conv3x3"] = int(block_11_combined_scale2.item())
    scale_factors["BN11"]["conv1x1_2"] = int(block_11_combined_scale3.item())
    scale_factors["BN11"]["skip_add"] = int (block_11_combined_scale_skip.item())

    print("********************BN12*******************************")
    print("combined_scale after conv1x1:", block_12_combined_scale1.item())
    print("combined_scale after conv3x3:", block_12_combined_scale2.item())
    print("combined_scale after conv1x1:", block_12_combined_scale3.item())
    scale_factors["BN12"]["conv1x1_1"] = int(block_12_combined_scale1.item())
    scale_factors["BN12"]["conv3x3"] = int(block_12_combined_scale2.item() )
    scale_factors["BN12"]["conv1x1_2"] = int(block_12_combined_scale3.item())


    write_scale_factors(file_path, scale_factors)
    # ------------------------------------------------------
    # Reorder input data-layout
    # ------------------------------------------------------
    block_10_int_weight_1 = quant_bottleneck_model.bn10_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_10_int_weight_2 = quant_bottleneck_model.bn10_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_10_int_weight_3 = quant_bottleneck_model.bn10_quant_conv3.quant_weight().int(
        float_datatype=True
    )
  
    block_11_int_weight_1 = quant_bottleneck_model.bn11_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_11_int_weight_2 = quant_bottleneck_model.bn11_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_11_int_weight_3 = quant_bottleneck_model.bn11_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    block_12_int_weight_1 = quant_bottleneck_model.bn12_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_12_int_weight_2 = quant_bottleneck_model.bn12_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_12_int_weight_3 = quant_bottleneck_model.bn12_quant_conv3.quant_weight().int(
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
    # **************************** bn10 ****************************
    wts1 = ds.reorder_mat(
        block_10_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    wts2 = ds.reorder_mat(
        block_10_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    wts3 = ds.reorder_mat(
        block_10_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
   
    # **************************** bn11 ****************************
    bn11_wts1 = ds.reorder_mat(
        block_11_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn11_wts2 = ds.reorder_mat(
        block_11_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn11_wts3 = ds.reorder_mat(
        block_11_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    # **************************** bn12 ****************************
    bn12_wts1 = ds.reorder_mat(
        block_12_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn12_wts2 = ds.reorder_mat(
        block_12_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn12_wts3 = ds.reorder_mat(
        block_12_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn10_total_wts = np.concatenate((wts1, wts2, wts3), axis=None)
    bn11_total_wts = np.concatenate((bn11_wts1, bn11_wts2, bn11_wts3), axis=None)
    bn12_total_wts = np.concatenate((bn12_wts1, bn12_wts2, bn12_wts3), axis=None)
    total_wts = np.concatenate((bn10_total_wts,bn11_total_wts,bn12_total_wts), axis=None)
    total_wts.tofile(log_folder + "/after_weights_mem_fmt_final.txt", sep=",", format="%d")
    print(total_wts.shape)
    # ------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------
    
    
    times = []
    write_wts(app,total_wts)
    for i in range(num_iter):
        start = time.time_ns()
        aie_output = execute_inference(app, ifm_mem_fmt) 
        stop = time.time_ns()
        npu_time = stop - start
        times.append(npu_time)

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
        # print("\nIter:{}, NPU time: {}us.".format(i,int((npu_time) / 1000)))
        # 
        print("AIE:",ofm_mem_fmt_out)
        print("Golden (int):",golden_output)

        # ------------------------------------------------------
        # Compare the AIE output and the golden reference
        # ------------------------------------------------------
    

        # print("Golden  (float): ",q_bottleneck_out.value.detach())
        
        golden=convert_to_numpy(golden_output)
        ofm_mem_fmt_out=convert_to_numpy(ofm_mem_fmt_out)
        max_diff_int = np.max((golden)-(ofm_mem_fmt_out))
        # print("atol: {} max difference (float/int): {} / {}".format(atol,max_diff,max_diff_int))
        print("max difference (int): {}".format(max_diff_int))
    average_time = sum(times) / num_iter
    best_time = min(times)
    print("\nNPU time= Avg: {}us, Best: {}us.".format(int((average_time) / 1000),int((best_time) / 1000)))

    if np.allclose(
        golden,
        ofm_mem_fmt_out,
        rtol=0,
        atol=5,
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
