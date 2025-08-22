#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
import sys
import onnx
import torchvision
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import sys
import math
from aie.utils.ml import DataShaper
import time
import os
import numpy as np
from aie.utils.xrt import setup_aie, extract_trace, write_out_trace, execute
import aie.utils.test as test_utils
import torch.utils.data as data_utils
from dolphin import print_dolphin,print_three_dolphins
from brevitas.nn import QuantConv2d, QuantIdentity, QuantReLU
from brevitas.quant.fixed_point import (
    Int8ActPerTensorFixedPoint,
    Int8WeightPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)
import torch.nn.functional as F
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate

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
    

def convert_to_numpy(array):
    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    else:
        raise TypeError("Unsupported array type")
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)


import json
def pad_tensor(tensor, target_shape):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    # Calculate the padding dimensions
    pad_size = target_shape[1] - tensor.shape[1]
    padding_dims = (0, 0, 0, 0, 0, pad_size)  # Padding along the second dimension (index 1)
    padded_tensor = F.pad(tensor, padding_dims)
    return padded_tensor
# Function to read scale factors from JSON file
def read_scale_factors(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
# Function to write scale factors to JSON file
def write_scale_factors(file_path, scale_factors):
    with open(file_path, 'w') as file:
        json.dump(scale_factors, file, indent=4)

def pad_weights(weights, target_shape):
    if isinstance(weights, np.ndarray):
        weights = torch.from_numpy(weights)
    
    # Calculate the padding dimensions for the second dimension (index 1)
    pad_size = target_shape[1] - weights.shape[1]
    padding_dims = (0, 0, 0, 0, 0, pad_size)  # Padding along the second dimension (index 1)
    padded_weights = F.pad(weights, padding_dims)
    return padded_weights


# Read the existing scale factors
file_path = 'scale_factors_final.json'
scale_factors = read_scale_factors(file_path)

vectorSize=8


tensorInW = 224
tensorInH = 224
tensorInC = 8

tensor_init_OutC=16
tensor_init_OutH=tensorInW//2
tensor_init_OutW=tensorInH//2


bneck_0_InW2 = tensor_init_OutW
bneck_0_InH2 = tensor_init_OutH
bneck_0_InC2 = tensor_init_OutC
bneck_0_OutC2 = bneck_0_InC2

bneck_0_InW3 = bneck_0_InW2
bneck_0_InH3 = bneck_0_InH2
bneck_0_InC3 = bneck_0_OutC2
bneck_0_OutC3 = bneck_0_InC3

# config for bn2
bn1_depthwiseStride = 2
bn1_depthWiseChannels = 64
bneck_1_OutC=24

# each layer
bneck_1_InW1 = bneck_0_InW3
bneck_1_InH1 = bneck_0_InH3
bneck_1_InC1 = bneck_0_InC3
bneck_1_OutC1 = bn1_depthWiseChannels

bneck_1_InW2 = bneck_1_InW1
bneck_1_InH2 = bneck_1_InH1
bneck_1_OutC2 = bneck_1_OutC1

bneck_1_InW3 = bneck_1_InW2 // bn1_depthwiseStride
bneck_1_InH3 = bneck_1_InH2 // bn1_depthwiseStride
bneck_1_OutC3 = bneck_1_OutC

# config for bn2
bn2_depthwiseStride = 1
bn2_depthWiseChannels = 72
bneck_2_OutC=24

# each layer
bneck_2_InW1 = bneck_1_InW3
bneck_2_InH1 = bneck_1_InH3
bneck_2_InC1 = bneck_1_OutC3
bneck_2_OutC1 = bn2_depthWiseChannels

bneck_2_InW2 = bneck_2_InW1
bneck_2_InH2 = bneck_2_InH1
bneck_2_OutC2 = bneck_2_OutC1

bneck_2_InW3 = bneck_2_InW2 // bn2_depthwiseStride
bneck_2_InH3 = bneck_2_InH2 // bn2_depthwiseStride
bneck_2_OutC3 = bneck_2_OutC

# config for bn3
bn3_depthwiseStride = 2
bn3_depthWiseChannels = 72
bneck_3_OutC=40

# each layer
bneck_3_InW1 = bneck_2_InW3
bneck_3_InH1 = bneck_2_InH3
bneck_3_InC1 = bneck_2_OutC3
bneck_3_OutC1 = bn3_depthWiseChannels

bneck_3_InW2 = bneck_3_InW1
bneck_3_InH2 = bneck_3_InH1
bneck_3_OutC2 = bneck_3_OutC1

bneck_3_InW3 = bneck_3_InW2 // bn3_depthwiseStride
bneck_3_InH3 = bneck_3_InH2 // bn3_depthwiseStride
bneck_3_OutC3 = bneck_3_OutC


# config for bn5
bn4_depthwiseStride = 1
bn4_depthWiseChannels = 120
bneck_4_OutC=40

# each layer
bneck_4_InW1 = bneck_3_InW3
bneck_4_InH1 = bneck_3_InH3
bneck_4_InC1 = bneck_3_OutC3
bneck_4_OutC1 = bn4_depthWiseChannels

bneck_4_InW2 = bneck_4_InW1
bneck_4_InH2 = bneck_4_InH1
bneck_4_OutC2 = bneck_4_OutC1

bneck_4_InW3 = bneck_4_InW2 // bn4_depthwiseStride
bneck_4_InH3 = bneck_4_InH2 // bn4_depthwiseStride
bneck_4_OutC3 = bneck_4_OutC

# config for bn5
bn5_depthwiseStride = 1
bn5_depthWiseChannels = 120
bneck_5_InW1 = 28
bneck_5_InH1 = 28
bneck_5_InC1 = 40
bneck_5_OutC=40

bneck_5_OutC1 = bn5_depthWiseChannels

bneck_5_InW2 = bneck_5_InW1
bneck_5_InH2 = bneck_5_InH1
bneck_5_OutC2 = bneck_5_OutC1

bneck_5_InW3 = bneck_5_InW2 // bn5_depthwiseStride
bneck_5_InH3 = bneck_5_InH2 // bn5_depthwiseStride
bneck_5_OutC3 = bneck_5_OutC

# config for bn6
bneck_6_tensorInW = bneck_5_InW3
bneck_6_tensorInH = bneck_5_InH3
bneck_6_tensorInC = bneck_5_OutC3
bneck_6_tensorOutC = 80
bn6_depthwiseStride = 2
bn6_depthWiseChannels = 240

bneck_6_InW1 = bneck_6_tensorInW
bneck_6_InH1 = bneck_6_tensorInH
bneck_6_InC1 = bneck_6_tensorInC
bneck_6_OutC1 = bn6_depthWiseChannels

bneck_6_InW2 = bneck_6_InW1 
bneck_6_InH2 = bneck_6_InH1 
bneck_6_OutC2 = bneck_6_OutC1

bneck_6_InW3 = bneck_6_InW2 // bn6_depthwiseStride
bneck_6_InH3 = bneck_6_InH2 // bn6_depthwiseStride
bneck_6_OutC3 = bneck_6_tensorOutC

# config for bn7
bneck_7_tensorInW = bneck_6_InW3
bneck_7_tensorInH = bneck_6_InH3 
bneck_7_tensorInC = bneck_6_OutC3
bneck_7_tensorOutC = 80

bn7_depthwiseStride = 1
bn7_depthWiseChannels = 200

bneck_7_InW1 = bneck_7_tensorInW
bneck_7_InH1 = bneck_7_tensorInH
bneck_7_InC1 = bneck_7_tensorInC
bneck_7_OutC1 = bn7_depthWiseChannels

bneck_7_InW2 = bneck_7_InW1
bneck_7_InH2 = bneck_7_InH1
bneck_7_OutC2 = bneck_7_OutC1

bneck_7_InW3 = bneck_7_InW2
bneck_7_InH3 = bneck_7_InH2
bneck_7_OutC3 = bneck_7_tensorOutC

# config for bn8
bneck_8_tensorInW = bneck_7_InW3
bneck_8_tensorInH = bneck_7_InH3 
bneck_8_tensorInC = bneck_7_OutC3
bneck_8_tensorOutC = 80
bneck_8_depthwiseStride = 1
bneck_8_depthWiseChannels = 184

bneck_8_InW1 = bneck_8_tensorInW
bneck_8_InH1 = bneck_8_tensorInH
bneck_8_InC1 = bneck_8_tensorInC
bneck_8_OutC1 = bneck_8_depthWiseChannels

bneck_8_InW2 = bneck_8_InW1
bneck_8_InH2 = bneck_8_InH1
bneck_8_OutC2 = bneck_8_OutC1

bneck_8_InW3 = bneck_8_InW2
bneck_8_InH3 = bneck_8_InH2
bneck_8_OutC3 = bneck_8_tensorOutC


# config for bn8
bneck_9_tensorInW = bneck_8_InW3
bneck_9_tensorInH = bneck_8_InH3 
bneck_9_tensorInC = bneck_8_OutC3
bneck_9_tensorOutC = 80
bneck_9_depthwiseStride = 1
bneck_9_depthWiseChannels = 184

bneck_9_InW1 = bneck_9_tensorInW
bneck_9_InH1 = bneck_9_tensorInH
bneck_9_InC1 = bneck_9_tensorInC
bneck_9_OutC1 = bneck_9_depthWiseChannels

bneck_9_InW2 = bneck_9_InW1
bneck_9_InH2 = bneck_9_InH1
bneck_9_OutC2 = bneck_9_OutC1

bneck_9_InW3 = bneck_9_InW2
bneck_9_InH3 = bneck_9_InH2
bneck_9_OutC3 = bneck_9_tensorOutC


bneck_10_InW1 = 14
bneck_10_InH1 = 14
bneck_10_InC1 = bneck_9_OutC3
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

bneck_11_InW3 = 14
bneck_11_InH3 = 14

bneck_12_InH1 = 14
bneck_12_InW1 = 14
bneck_12_OutC1 = 336

bneck_12_InH2 = 14
bneck_12_InW2 = 14
bneck_12_OutC2 = 336

bneck_12_InW3 = 7
bneck_12_InH3 = 7
bneck_12_OutC3 = 80

# tensorOutW = bneck_3_InW3 
# tensorOutH = bneck_3_InH3
# tensorOutC = bneck_3_OutC3


bneck_13_InW1 = 7
bneck_13_InH1 = 7
bneck_13_InC1 = 80
bneck_13_OutC1 = 960
WeightChunks=2 #2 splits for input channel and then output 

bneck_13_InW2 = bneck_13_InW1
bneck_13_InH2 = bneck_13_InH1
bneck_13_OutC2 = bneck_13_OutC1

bneck_13_InW3 = bneck_13_InW1
bneck_13_InH3 = bneck_13_InH1
bneck_13_OutC3 = 80

post_L1_InW = 7
post_L1_InH = 7
post_L1_InC = 80


post_L1_OutC= 960 
post_L1_OutW=1
post_L1_OutH=1

post_L1_OutC_padd = 1280 # added for padding

post_L2_OutC= post_L1_OutC_padd
post_L2_OutW=1
post_L2_OutH=1

post_kdim=7
post_stride=1

tensorOutW = post_L2_OutW 
tensorOutH = post_L2_OutH
tensorOutC = post_L2_OutC

# tensorOutW = bneck_13_InW3 
# tensorOutH = bneck_13_InH3
# tensorOutC = bneck_13_OutC3
# Target shape for the padded weights
target_weights_shape = (post_L1_OutC_padd, post_L1_OutC_padd, 1, 1)
# Pad the input to the target shape
target_shape = (1, post_L1_OutC_padd, 1, 1)
InC_vec =  math.floor(3*tensorInC/vectorSize)
OutC_vec =  math.floor(tensorOutC/vectorSize)


def main(opts):
    design = "mobilenet_complete"
    xclbin_path = opts.xclbin
    insts_path = opts.instr
    ds = DataShaper()
    log_folder = "log/"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    weight_folder = "weights/"
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)

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
    dtype_out_aie = np.dtype("uint16")
    # dtype_out = np.dtype("int8")

    shape_total_wts =(  
                        # (3 * 3 * tensorInC * tensor_init_OutC) +
                        # (3*3*bneck_0_OutC2 + bneck_0_OutC2*bneck_0_OutC3 + bneck_1_InC1*bneck_1_OutC1 + 3*3*bneck_1_OutC2 + bneck_1_OutC2*bneck_1_OutC3)+
                        # (bneck_2_InC1*bneck_2_OutC1 + 3*3*bneck_2_OutC2 + bneck_2_OutC2*bneck_2_OutC3)+
                        # (bneck_3_InC1*bneck_3_OutC1 + 3*3*bneck_3_OutC2 + bneck_3_OutC2*bneck_3_OutC3)+
                        # (bneck_4_InC1*bneck_4_OutC1 + 3*3*bneck_4_OutC2 + bneck_4_OutC2*bneck_4_OutC3)+
                        # (bneck_5_InC1*bneck_5_OutC1 + 3*3*bneck_5_OutC2 + bneck_5_OutC2*bneck_5_OutC3)+
                        # (bneck_6_InC1*bneck_6_OutC1 + 3*3*bneck_6_OutC2 + bneck_6_OutC2*bneck_6_OutC3)+
                        # (bneck_7_InC1*bneck_7_OutC1 + 3*3*bneck_7_OutC2 + bneck_7_OutC2*bneck_7_OutC3)+
                        # (bneck_8_InC1*bneck_8_OutC1 + 3*3*bneck_8_OutC2 + bneck_8_OutC2*bneck_8_OutC3)+
                        # (bneck_9_InC1*bneck_9_OutC1 + 3*3*bneck_9_OutC2 + bneck_9_OutC2*bneck_9_OutC3)+
                        # (bneck_10_InC1*bneck_10_OutC1)+(3*3*bneck_10_OutC2)+(bneck_10_OutC2*bneck_10_OutC3)+
                        # (bneck_10_OutC3*bneck_11_OutC1)+(3*3*bneck_11_OutC2)+(bneck_11_OutC2*bneck_11_OutC3)+
                        # (bneck_11_OutC3*bneck_12_OutC1)+(3*3*bneck_12_OutC2)+(bneck_12_OutC2*bneck_12_OutC3)+
                        2*((bneck_13_OutC1*bneck_13_InC1)+(bneck_13_OutC2*bneck_13_OutC3)),1)
    
    print("total weights:::",shape_total_wts)
    shape_in_act = (tensorInH, InC_vec, tensorInW, vectorSize)  #'YCXC8' , 'CYX'
    shape_out = (tensorOutH, OutC_vec, tensorOutW, vectorSize) # HCWC8
    shape_out_final = (OutC_vec*vectorSize, tensorOutH, tensorOutW) # CHW
   
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
        dtype_out_aie,
        enable_trace=enable_trace,
        trace_size=trace_size,
    )
    class QuantMobilenet(nn.Module):
        def __init__(self, in_planes=8,
                     bn0_expand=16,bn0_project=16,
                     bn1_expand=16,bn1_project=16,
                     bn2_expand=16,bn2_project=16,
                     bn3_expand=16,bn3_project=16,
                     bn4_expand=16,bn4_project=16, 
                     bn5_expand=16,bn5_project=16, 
                     bn6_expand=16,bn6_project=16, 
                     bn7_expand=16,bn7_project=16, 
                     bn8_expand=16,bn8_project=16,
                     bn9_expand=16,bn9_project=16,
                     bn10_expand=16,bn10_project=16,
                     bn11_expand=16,bn11_project=16,
                     bn12_expand=16,bn12_project=16, 
                     bn13_expand=16,bn13_project=16,
                     bn14_expand=16,bn14_project=16,
                     post_conv1=20,post_layer2_conv1=20,post_conv2=20):
            super(QuantMobilenet, self).__init__()
            self.quant_id_1 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.init_conv = QuantConv2d(
                in_planes,
                bn0_expand,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.init_relu = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
        # *************************** A ******************************
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
            self.bn0_quant_id_2.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl = self.init_relu.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl
            self.bn0_quant_id_2.act_quant.fused_activation_quant_proxy.tensor_quant.int_scaling_impl = self.init_relu.act_quant.fused_activation_quant_proxy.tensor_quant.int_scaling_impl

            # 
            self.bn1_quant_conv1 = QuantConv2d(
                bn0_project,
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
                stride=bn1_depthwiseStride,
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
            # bn2
            self.bn2_quant_conv1 = QuantConv2d(
                bn1_project,
                bn2_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn2_quant_conv2 = QuantConv2d(
                bn2_expand,
                bn2_expand,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn2_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn2_quant_conv3 = QuantConv2d(
                bn2_expand,
                bn2_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn2_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn2_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn2_add = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            # self.bn2_quant_id = QuantIdentity(
            #     act_quant=Int8ActPerTensorFixedPoint,
            #     bit_width=8,
            #     return_quant_tensor=True,
            # )
            # # force alignment between scales going into add
            # self.bn2_quant_id.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl = self.bn1_quant_id_2.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl
            # self.bn2_quant_id.act_quant.fused_activation_quant_proxy.tensor_quant.int_scaling_impl = self.bn1_quant_id_2.act_quant.fused_activation_quant_proxy.tensor_quant.int_scaling_impl

# bn3
            self.bn3_quant_conv1 = QuantConv2d(
                bn2_project,
                bn3_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn3_quant_conv2 = QuantConv2d(
                bn3_expand,
                bn3_expand,
                kernel_size=3,
                stride=bn3_depthwiseStride,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn3_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn3_quant_conv3 = QuantConv2d(
                bn3_expand,
                bn3_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn3_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn3_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn3_quant_id_2 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
# 
            self.bn4_quant_conv1 = QuantConv2d(
                bn3_project,
                bn4_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn4_quant_conv2 = QuantConv2d(
                bn4_expand,
                bn4_expand,
                kernel_size=3,
                stride=bn4_depthwiseStride,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn4_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn4_quant_conv3 = QuantConv2d(
                bn4_expand,
                bn4_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn4_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn4_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn4_add = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
# 
            self.bn5_quant_conv1 = QuantConv2d(
                bn4_project,
                bn5_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn5_quant_conv2 = QuantConv2d(
                bn5_expand,
                bn5_expand,
                kernel_size=3,
                stride=bn5_depthwiseStride,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn5_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn5_quant_conv3 = QuantConv2d(
                bn5_expand,
                bn5_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn5_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn5_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn5_add = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

            self.bn6_quant_conv1 = QuantConv2d(
                bn5_project,
                bn6_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn6_quant_conv2 = QuantConv2d(
                bn6_expand,
                bn6_expand,
                kernel_size=3,
                stride=bn6_depthwiseStride,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn6_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn6_quant_conv3 = QuantConv2d(
                bn6_expand,
                bn6_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn6_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn6_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn6_quant_id_2 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

            # bn7
            self.bn7_quant_conv1 = QuantConv2d(
                bn6_project,
                bn7_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn7_quant_conv2 = QuantConv2d(
                bn7_expand,
                bn7_expand,
                kernel_size=3,
                stride=bn7_depthwiseStride,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn7_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn7_quant_conv3 = QuantConv2d(
                bn7_expand,
                bn7_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn7_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn7_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn7_add = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            # bn8
            self.bn8_quant_conv1 = QuantConv2d(
                bn7_project,
                bn8_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn8_quant_conv2 = QuantConv2d(
                bn8_expand,
                bn8_expand,
                kernel_size=3,
                stride=bneck_8_depthwiseStride,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn8_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn8_quant_conv3 = QuantConv2d(
                bn8_expand,
                bn8_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn8_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn8_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn8_add = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn8_quant_id_2 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

            # bn9
            self.bn9_quant_conv1 = QuantConv2d(
                bn8_project,
                bn9_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn9_quant_conv2 = QuantConv2d(
                bn9_expand,
                bn9_expand,
                kernel_size=3,
                stride=bneck_8_depthwiseStride,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn9_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn9_quant_conv3 = QuantConv2d(
                bn9_expand,
                bn9_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn9_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn9_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn9_add = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
# bn10
            self.bn10_quant_conv1 = QuantConv2d(
                bn9_project,
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
            # C mappings

            self.bn13_quant_conv1 = QuantConv2d(
                bn12_project,
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

            # bn14

            self.bn14_quant_conv1 = QuantConv2d(
                bn13_project,
                bn14_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn14_quant_conv2 = QuantConv2d(
                bn14_expand,
                bn14_expand,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn14_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn14_quant_conv3 = QuantConv2d(
                bn14_expand,
                bn14_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn14_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn14_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn14_add = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
# Post
            self.post_quant_conv1 = QuantConv2d(
                bn14_project,
                post_conv1,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.post_relu = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

            self.post_quant_id_1 = QuantIdentity(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.post_avg_pool=nn.AvgPool2d(post_kdim,stride=post_stride)
            self.post_layer2_quant_conv1 = QuantConv2d(
                post_conv1,
                post_layer2_conv1,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.post_layer2_relu = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

            self.post_layer3_quant_conv1 = QuantConv2d(
                post_layer2_conv1,
                post_conv2,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.post_layer3_relu = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

        def forward(self, x):
            out = self.quant_id_1(x)
            out = self.init_conv(out)
            out_q = self.init_relu(out)

            out = self.bn0_quant_conv2(out_q)
            out = self.bn0_quant_relu2(out)
            out = self.bn0_quant_conv3(out)
            out = self.bn0_quant_id_2(out)
            out = out+out_q
            out = self.bn0_add(out)

            # # bn1
            out = self.bn1_quant_conv1(out)
            out = self.bn1_quant_relu1(out)
            out = self.bn1_quant_conv2(out)
            out = self.bn1_quant_relu2(out)
            out = self.bn1_quant_conv3(out)
            out_q = self.bn1_quant_id_2(out)

            # # # # # # bn2
            out = self.bn2_quant_conv1(out_q)
            out = self.bn2_quant_relu1(out)
            out = self.bn2_quant_conv2(out)
            out = self.bn2_quant_relu2(out)
            out = self.bn2_quant_conv3(out)
            out = self.bn1_quant_id_2(out)
            out = out+out_q
            out = self.bn2_add(out)

            # # # # # # bn3
            out = self.bn3_quant_conv1(out)
            out = self.bn3_quant_relu1(out)
            out = self.bn3_quant_conv2(out)
            out = self.bn3_quant_relu2(out)
            out = self.bn3_quant_conv3(out)
            out_q = self.bn3_quant_id_2(out)

            # # # # # bn4
            out = self.bn4_quant_conv1(out_q)
            out = self.bn4_quant_relu1(out)
            out = self.bn4_quant_conv2(out)
            out = self.bn4_quant_relu2(out)
            out = self.bn4_quant_conv3(out)
            out = self.bn3_quant_id_2(out)
            out = out+out_q
            out_q = self.bn4_add(out)

            # # # # bn5
            out = self.bn5_quant_conv1(out_q)
            out = self.bn5_quant_relu1(out)
            out = self.bn5_quant_conv2(out)
            out = self.bn5_quant_relu2(out)
            out = self.bn5_quant_conv3(out)
            out = self.bn4_add(out)
            out = out+out_q
            out = self.bn5_add(out)
            
            # bn6
            out = self.bn6_quant_conv1(out)
            out = self.bn6_quant_relu1(out)
            out = self.bn6_quant_conv2(out)
            out = self.bn6_quant_relu2(out)
            out = self.bn6_quant_conv3(out)
            out_q = self.bn6_quant_id_2(out)

            
            # # bn7
            out = self.bn7_quant_conv1(out_q)
            out = self.bn7_quant_relu1(out)
            out = self.bn7_quant_conv2(out)
            out = self.bn7_quant_relu2(out)
            out = self.bn7_quant_conv3(out)
            out = self.bn6_quant_id_2(out)
            out = out+out_q
            out_q = self.bn7_add(out)

            # bn8

            out = self.bn8_quant_conv1(out_q)
            out = self.bn8_quant_relu1(out)
            out = self.bn8_quant_conv2(out)
            out = self.bn8_quant_relu2(out)
            out = self.bn8_quant_conv3(out)
            out = self.bn7_add(out)
            out = out+out_q
            out_q = self.bn8_add(out)

            # bn9

            out = self.bn9_quant_conv1(out_q)
            out = self.bn9_quant_relu1(out)
            out = self.bn9_quant_conv2(out)
            out = self.bn9_quant_relu2(out)
            out = self.bn9_quant_conv3(out)
            out = self.bn8_add(out)
            out = out+out_q
            out_q = self.bn9_add(out)

            # bn10
            out = self.bn10_quant_conv1(out_q)
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
            out=out+out_lhs
            out = self.bn11_add(out)
            # bn12
            out = self.bn12_quant_conv1(out)
            out = self.bn12_quant_relu1(out)
            out = self.bn12_quant_conv2(out)
            out = self.bn12_quant_relu2(out)
            out = self.bn12_quant_conv3(out)
            bn13_out_q = self.bn12_quant_id_2(out)


            bn13_out = self.bn13_quant_conv1(bn13_out_q)
            bn13_out = self.bn13_quant_relu1(bn13_out)
            bn13_out = self.bn13_quant_conv2(bn13_out)
            bn13_out = self.bn13_quant_relu2(bn13_out)
            bn13_out = self.bn13_quant_conv3(bn13_out)
            # out = self.bn13_quant_relu3(out)
            bn13_out = self.bn12_quant_id_2(bn13_out)
            bn13_out=bn13_out+bn13_out_q
            bn13_out = self.bn13_add(bn13_out)

            bn14_out = self.bn14_quant_conv1(bn13_out)
            bn14_out = self.bn14_quant_relu1(bn14_out)
            bn14_out = self.bn14_quant_conv2(bn14_out)
            bn14_out = self.bn14_quant_relu2(bn14_out)
            bn14_out = self.bn14_quant_conv3(bn14_out)

            bn14_out = self.bn13_add(bn14_out)
            bn14_out=bn13_out+bn14_out

            bn14_out = self.bn14_add(bn14_out)

            out=self.post_quant_conv1 (bn14_out)
            out=self.post_relu(out)
            out=self.post_avg_pool(out)
            # out=  F.avg_pool2d(out, post_kdim, post_stride)
            out = self.post_quant_id_1(out)

            #FC1+FC2
            out=self.post_layer2_quant_conv1 (out)
            out=self.post_layer2_relu(out)
            out=self.post_layer3_quant_conv1 (out)
            out=self.post_layer3_relu(out)


            return out

    quant_model = QuantMobilenet(in_planes=tensorInC, 
                                            bn0_expand=bneck_0_InC2,bn0_project=bneck_0_OutC3,
                                            bn1_expand=bneck_1_OutC1,bn1_project=bneck_1_OutC3,
                                            bn2_expand=bneck_2_OutC1,bn2_project=bneck_2_OutC3,
                                            bn3_expand=bneck_3_OutC1,bn3_project=bneck_3_OutC3, 
                                            bn4_expand=bneck_4_OutC1,bn4_project=bneck_4_OutC3, 
                                            bn5_expand=bneck_5_OutC1,bn5_project=bneck_5_OutC3, 
                                            bn6_expand=bneck_6_OutC1,bn6_project=bneck_6_OutC3,
                                            bn7_expand=bneck_7_OutC1,bn7_project=bneck_7_OutC3, 
                                            bn8_expand=bneck_8_OutC1,bn8_project=bneck_8_OutC3,
                                            bn9_expand=bneck_9_OutC1,bn9_project=bneck_9_OutC3,
                                            bn10_expand=bneck_10_OutC2,bn10_project=bneck_10_OutC3, 
                                            bn11_expand=bneck_11_OutC2,bn11_project=bneck_11_OutC3,
                                            bn12_expand=bneck_12_OutC2,bn12_project=bneck_12_OutC3, 
                                            bn13_expand=bneck_13_OutC2,bn13_project=bneck_13_OutC3,
                                            bn14_expand=bneck_13_OutC2,bn14_project=bneck_13_OutC3,
                                            post_conv1=post_L1_OutC,post_layer2_conv1=post_L2_OutC, post_conv2=post_L2_OutC)
    from utils import ExpandChannels
    from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
    import torchvision
    import torch.utils.data as data_utils
    from torchvision import transforms
    # # Define the image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(tensorInW),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ExpandChannels(target_channels=tensorInC)  # Expand to 80 channels
    ])

    # data_dir = "data"
    
    # # test_dataset = torchvision.datasets.ImageNet(
    # #     root=data_dir, train=False, transform=transform, download=True)
    
    # # # Create a subset and DataLoader for the single image
    # # indices = torch.arange(32)
    # # val_sub = data_utils.Subset(test_dataset, indices)
    # # calib_loader = torch.utils.data.DataLoader(dataset=val_sub, batch_size=32, shuffle=False)
    
    src_data="/group/xrlabs2/imagenet/calibration"
    datset=torchvision.datasets.ImageFolder(
        src_data,
        transform)
    indices = torch.arange(4)
    val_sub = data_utils.Subset(datset, indices)
    calib_loader = torch.utils.data.DataLoader(dataset=val_sub, batch_size=32, shuffle=False)

    quant_model.eval()
    calibrate(calib_loader, quant_model)
    quant_model.eval()

        # Extract a single image
    for batch in calib_loader:
        images, labels = batch
        input = images[0].unsqueeze(0)   # Get the first image from the batch
        input_label = labels[0]  # Get the corresponding label
        break

    # input = torch.randn(1, tensorInC, tensorInH, tensorInW)

    q_bottleneck_out = quant_model(input)
    golden_output = q_bottleneck_out.int(float_datatype=True).data.numpy().astype(dtype_out)
    padded_golden_output = pad_tensor(golden_output, target_shape)
    q_inp = quant_model.quant_id_1(input)
    int_inp = q_inp.int(float_datatype=True)
    print(int_inp)
    print(input.shape)

    
    inp_scale= quant_model.quant_id_1.act_quant.scale()

    init_relu = quant_model.init_relu.act_quant.scale()
    init_weight_scale = quant_model.init_conv.weight_quant.scale()
    init_combined_scale = -torch.log2(inp_scale * init_weight_scale / init_relu)
    print("********************INIT*******************************")
    print("combined_scale after INIT conv3x3:", init_combined_scale.item())
    scale_factors["INIT"]["conv3x3"] = int(init_combined_scale.item() )

    block_0_relu_2 = quant_model.bn0_quant_relu2.act_quant.scale()
    block_0_skip_add = quant_model.bn0_add.act_quant.scale()

    block_0_weight_scale2 = quant_model.bn0_quant_conv2.weight_quant.scale()
    block_0_weight_scale3 = quant_model.bn0_quant_conv3.weight_quant.scale()

    block_0_combined_scale2 = -torch.log2(
        init_relu * block_0_weight_scale2 / block_0_relu_2
    )  
    block_0_combined_scale3 = -torch.log2(
        block_0_relu_2 * block_0_weight_scale3/init_relu
    )   
    block_0_combined_scale_skip = -torch.log2(
        init_relu / block_0_skip_add
    )  # After addition | clip -128-->127

    block_1_relu_1 = quant_model.bn1_quant_relu1.act_quant.scale()
    block_1_relu_2 = quant_model.bn1_quant_relu2.act_quant.scale()
    block_1_final_scale = quant_model.bn1_quant_id_2.act_quant.scale()

    block_1_weight_scale1 = quant_model.bn1_quant_conv1.weight_quant.scale()
    block_1_weight_scale2 = quant_model.bn1_quant_conv2.weight_quant.scale()
    block_1_weight_scale3 = quant_model.bn1_quant_conv3.weight_quant.scale()
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
    scale_factors["BN0"]["conv3x3"] = int(block_0_combined_scale2.item() )
    scale_factors["BN0"]["conv1x1_2"] = int(block_0_combined_scale3.item())
    scale_factors["BN0"]["skip_add"] = int(block_0_combined_scale_skip.item())
    

    print("********************BN1*******************************")
    print("combined_scale after conv1x1:", block_1_combined_scale1.item())
    print("combined_scale after conv3x3:", block_1_combined_scale2.item())
    print("combined_scale after conv1x1:", block_1_combined_scale3.item())
    print("********************BN1*******************************")
    scale_factors["BN1"]["conv1x1_1"] = int(block_1_combined_scale1.item())
    scale_factors["BN1"]["conv3x3"] = int(block_1_combined_scale2.item())
    scale_factors["BN1"]["conv1x1_2"] = int(block_1_combined_scale3.item())
    scale_factors["BN1"]["skip_add"] = 0


    

    block_2_inp_scale1= block_1_final_scale

    block_2_relu_1 = quant_model.bn2_quant_relu1.act_quant.scale()
    block_2_relu_2 = quant_model.bn2_quant_relu2.act_quant.scale()

    block_2_skip_add = quant_model.bn2_add.act_quant.scale()

    block_2_weight_scale1 = quant_model.bn2_quant_conv1.weight_quant.scale()
    block_2_weight_scale2 = quant_model.bn2_quant_conv2.weight_quant.scale()
    block_2_weight_scale3 = quant_model.bn2_quant_conv3.weight_quant.scale()
    block_2_combined_scale1 = -torch.log2(
        block_2_inp_scale1 * block_2_weight_scale1 / block_2_relu_1
    )
    block_2_combined_scale2 = -torch.log2(
        block_2_relu_1 * block_2_weight_scale2 / block_2_relu_2
    )  
    block_2_combined_scale3 = -torch.log2(
        block_2_relu_2 * block_2_weight_scale3/block_2_inp_scale1
    )   
    block_2_combined_scale_skip = -torch.log2(
        block_2_inp_scale1 / block_2_skip_add
    )  # After addition | clip -128-->127



    print("********************BN2*******************************")
    print("combined_scale after conv1x1:", block_2_combined_scale1.item())
    print("combined_scale after conv3x3:", block_2_combined_scale2.item())
    print("combined_scale after conv1x1:", block_2_combined_scale3.item())
    print("combined_scale after skip add:", block_2_combined_scale_skip.item())
    print("********************BN2*******************************")
    scale_factors["BN2"]["conv1x1_1"] =int( block_2_combined_scale1.item())
    scale_factors["BN2"]["conv3x3"] = int(block_2_combined_scale2.item())
    scale_factors["BN2"]["conv1x1_2"] = int(block_2_combined_scale3.item() )
    scale_factors["BN2"]["skip_add"] =int( block_2_combined_scale_skip.item() )
    
    init_scale = block_2_skip_add
    block_3_relu_1 = quant_model.bn3_quant_relu1.act_quant.scale()
    block_3_relu_2 = quant_model.bn3_quant_relu2.act_quant.scale()
    block_3_final_scale = quant_model.bn3_quant_id_2.act_quant.scale()

    block_3_weight_scale1 = quant_model.bn3_quant_conv1.weight_quant.scale()
    block_3_weight_scale2 = quant_model.bn3_quant_conv2.weight_quant.scale()
    block_3_weight_scale3 = quant_model.bn3_quant_conv3.weight_quant.scale()
    block_3_combined_scale1 = -torch.log2(
        init_scale * block_3_weight_scale1 / block_3_relu_1
    )
    block_3_combined_scale2 = -torch.log2(
        block_3_relu_1 * block_3_weight_scale2 / block_3_relu_2
    )  
    block_3_combined_scale3 = -torch.log2(
        block_3_relu_2 * block_3_weight_scale3/block_3_final_scale
    )   

    print("********************bn3*******************************")
    print("combined_scale after conv1x1:", block_3_combined_scale1.item())
    print("combined_scale after conv3x3:", block_3_combined_scale2.item())
    print("combined_scale after conv1x1:", block_3_combined_scale3.item())
    print("********************bn3*******************************")
    scale_factors["BN3"]["conv1x1_1"] = int(block_3_combined_scale1.item())
    scale_factors["BN3"]["conv3x3"] = int(block_3_combined_scale2.item())
    scale_factors["BN3"]["conv1x1_2"] = int(block_3_combined_scale3.item())
    scale_factors["BN3"]["skip_add"] = 0



    block_4_inp_scale1= block_3_final_scale
    block_4_relu_1 = quant_model.bn4_quant_relu1.act_quant.scale()
    block_4_relu_2 = quant_model.bn4_quant_relu2.act_quant.scale()
    block_4_skip_add = quant_model.bn4_add.act_quant.scale()

    block_4_weight_scale1 = quant_model.bn4_quant_conv1.weight_quant.scale()
    block_4_weight_scale2 = quant_model.bn4_quant_conv2.weight_quant.scale()
    block_4_weight_scale3 = quant_model.bn4_quant_conv3.weight_quant.scale()
    block_4_combined_scale1 = -torch.log2(
        block_4_inp_scale1 * block_4_weight_scale1 / block_4_relu_1
    )
    block_4_combined_scale2 = -torch.log2(
        block_4_relu_1 * block_4_weight_scale2 / block_4_relu_2
    )  
    block_4_combined_scale3 = -torch.log2(
        block_4_relu_2 * block_4_weight_scale3/block_4_inp_scale1
    )   
    block_4_combined_scale_skip = -torch.log2(
        block_4_inp_scale1 / block_4_skip_add
    )  # After addition | clip -128-->127



    print("********************bn4*******************************")
    print("combined_scale after conv1x1:", block_4_combined_scale1.item())
    print("combined_scale after conv3x3:", block_4_combined_scale2.item())
    print("combined_scale after conv1x1:", block_4_combined_scale3.item())
    print("combined_scale after skip add:", block_4_combined_scale_skip.item())
    print("********************bn4*******************************")
    scale_factors["BN4"]["conv1x1_1"] = int(block_4_combined_scale1.item())
    scale_factors["BN4"]["conv3x3"] = int(block_4_combined_scale2.item())
    scale_factors["BN4"]["conv1x1_2"] = int(block_4_combined_scale3.item())
    scale_factors["BN4"]["skip_add"] = int(block_4_combined_scale_skip.item() )


    block_5_inp_scale1= block_4_skip_add
    block_5_relu_1 = quant_model.bn5_quant_relu1.act_quant.scale()
    block_5_relu_2 = quant_model.bn5_quant_relu2.act_quant.scale()
    block_5_skip_add = quant_model.bn5_add.act_quant.scale()

    block_5_weight_scale1 = quant_model.bn5_quant_conv1.weight_quant.scale()
    block_5_weight_scale2 = quant_model.bn5_quant_conv2.weight_quant.scale()
    block_5_weight_scale3 = quant_model.bn5_quant_conv3.weight_quant.scale()
    block_5_combined_scale1 = -torch.log2(
        block_5_inp_scale1 * block_5_weight_scale1 / block_5_relu_1
    )
    block_5_combined_scale2 = -torch.log2(
        block_5_relu_1 * block_5_weight_scale2 / block_5_relu_2
    )  
    block_5_combined_scale3 = -torch.log2(
        block_5_relu_2 * block_5_weight_scale3/block_5_inp_scale1
    )   
    block_5_combined_scale_skip = -torch.log2(
        block_5_inp_scale1 / block_5_skip_add
    )  # After addition | clip -128-->127



    print("********************bn5*******************************")
    print("combined_scale after conv1x1:", block_5_combined_scale1.item())
    print("combined_scale after conv3x3:", block_5_combined_scale2.item())
    print("combined_scale after conv1x1:", block_5_combined_scale3.item())
    print("combined_scale after skip add:", block_5_combined_scale_skip.item())
    print("********************bn5*******************************")
    scale_factors["BN5"]["conv1x1_1"] = int(block_5_combined_scale1.item())
    scale_factors["BN5"]["conv3x3"] = int(block_5_combined_scale2.item())
    scale_factors["BN5"]["conv1x1_2"] = int(block_5_combined_scale3.item())
    scale_factors["BN5"]["skip_add"] = int(block_5_combined_scale_skip.item() )



    block_6_relu_1 = quant_model.bn6_quant_relu1.act_quant.scale()
    block_6_relu_2 = quant_model.bn6_quant_relu2.act_quant.scale()
    block_6_final_scale = quant_model.bn6_quant_id_2.act_quant.scale()

    block_6_weight_scale1 = quant_model.bn6_quant_conv1.weight_quant.scale()
    block_6_weight_scale2 = quant_model.bn6_quant_conv2.weight_quant.scale()
    block_6_weight_scale3 = quant_model.bn6_quant_conv3.weight_quant.scale()
    block_6_combined_scale1 = -torch.log2(
        block_5_skip_add * block_6_weight_scale1 / block_6_relu_1
    )
    block_6_combined_scale2 = -torch.log2(
        block_6_relu_1 * block_6_weight_scale2 / block_6_relu_2
    )  
    block_6_combined_scale3 = -torch.log2(
        block_6_relu_2 * block_6_weight_scale3/block_6_final_scale
    )   

    print("********************BN6*******************************")
    print("combined_scale after conv1x1:", block_6_combined_scale1.item())
    print("combined_scale after conv3x3:", block_6_combined_scale2.item())
    print("combined_scale after conv1x1:", block_6_combined_scale3.item())
    print("********************BN6*******************************")
    scale_factors["BN6"]["conv1x1_1"] = int(block_6_combined_scale1.item())
    scale_factors["BN6"]["conv3x3"] = int(block_6_combined_scale2.item())
    scale_factors["BN6"]["conv1x1_2"] = int(block_6_combined_scale3.item())
    scale_factors["BN6"]["skip_add"] = 0

    block_7_inp_scale1= block_6_final_scale

    block_7_relu_1 = quant_model.bn7_quant_relu1.act_quant.scale()
    block_7_relu_2 = quant_model.bn7_quant_relu2.act_quant.scale()
    block_7_skip_add = quant_model.bn7_add.act_quant.scale()

    block_7_weight_scale1 = quant_model.bn7_quant_conv1.weight_quant.scale()
    block_7_weight_scale2 = quant_model.bn7_quant_conv2.weight_quant.scale()
    block_7_weight_scale3 = quant_model.bn7_quant_conv3.weight_quant.scale()
    block_7_combined_scale1 = -torch.log2(
        block_7_inp_scale1 * block_7_weight_scale1 / block_7_relu_1
    )
    block_7_combined_scale2 = -torch.log2(
        block_7_relu_1 * block_7_weight_scale2 / block_7_relu_2
    )  
    block_7_combined_scale3 = -torch.log2(
        block_7_relu_2 * block_7_weight_scale3/block_7_inp_scale1
    )   
    block_7_combined_scale_skip = -torch.log2(
        block_7_inp_scale1 / block_7_skip_add
    )  # After addition | clip -128-->127

    print("********************BN7*******************************")
    print("combined_scale after conv1x1:", block_7_combined_scale1.item())
    print("combined_scale after conv3x3:", block_7_combined_scale2.item())
    print("combined_scale after conv1x1:", block_7_combined_scale3.item())
    print("combined_scale after skip add:", block_7_combined_scale_skip.item())
    print("********************BN7*******************************")
    scale_factors["BN7"]["conv1x1_1"] = int(block_7_combined_scale1.item())
    scale_factors["BN7"]["conv3x3"] = int(block_7_combined_scale2.item())
    scale_factors["BN7"]["conv1x1_2"] = int(block_7_combined_scale3.item())
    scale_factors["BN7"]["skip_add"] = int(block_7_combined_scale_skip.item() )

    block_8_inp_scale1= block_7_skip_add
    block_8_relu_1 = quant_model.bn8_quant_relu1.act_quant.scale()
    block_8_relu_2 = quant_model.bn8_quant_relu2.act_quant.scale()
    block_8_skip_add = quant_model.bn8_add.act_quant.scale()
    block_8_weight_scale1 = quant_model.bn8_quant_conv1.weight_quant.scale()
    block_8_weight_scale2 = quant_model.bn8_quant_conv2.weight_quant.scale()
    block_8_weight_scale3 = quant_model.bn8_quant_conv3.weight_quant.scale()

    block_8_combined_scale1 = -torch.log2(
        block_8_inp_scale1 * block_8_weight_scale1 / block_8_relu_1
    )
    block_8_combined_scale2 = -torch.log2(
        block_8_relu_1 * block_8_weight_scale2 / block_8_relu_2
    )  
    block_8_combined_scale3 = -torch.log2(
        block_8_relu_2 * block_8_weight_scale3/block_8_inp_scale1
    )   
    block_8_combined_scale_skip = -torch.log2(
        block_8_inp_scale1 / block_8_skip_add
    )  # After addition | clip -128-->127

    print("********************BN8*******************************")
    print("combined_scale after conv1x1:", block_8_combined_scale1.item())
    print("combined_scale after conv3x3:", block_8_combined_scale2.item())
    print("combined_scale after conv1x1:", block_8_combined_scale3.item())
    print("combined_scale after skip add:", block_8_combined_scale_skip.item())
    print("********************BN8*******************************")
    scale_factors["BN8"]["conv1x1_1"] = int(block_8_combined_scale1.item())
    scale_factors["BN8"]["conv3x3"] = int(block_8_combined_scale2.item())
    scale_factors["BN8"]["conv1x1_2"] = int(block_8_combined_scale3.item())
    scale_factors["BN8"]["skip_add"] = int(block_8_combined_scale_skip.item() )


    block_9_inp_scale1= block_8_skip_add
    block_9_relu_1 = quant_model.bn9_quant_relu1.act_quant.scale()
    block_9_relu_2 = quant_model.bn9_quant_relu2.act_quant.scale()
    block_9_skip_add = quant_model.bn9_add.act_quant.scale()
    block_9_weight_scale1 = quant_model.bn9_quant_conv1.weight_quant.scale()
    block_9_weight_scale2 = quant_model.bn9_quant_conv2.weight_quant.scale()
    block_9_weight_scale3 = quant_model.bn9_quant_conv3.weight_quant.scale()
    block_9_combined_scale1 = -torch.log2(
        block_9_inp_scale1 * block_9_weight_scale1 / block_9_relu_1
    )
    block_9_combined_scale2 = -torch.log2(
        block_9_relu_1 * block_9_weight_scale2 / block_9_relu_2
    )  
    block_9_combined_scale3 = -torch.log2(
        block_9_relu_2 * block_9_weight_scale3/block_9_inp_scale1
    )   
    block_9_combined_scale_skip = -torch.log2(
        block_9_inp_scale1 / block_9_skip_add
    )  # After addition | clip -128-->127

    print("********************BN9*******************************")
    print("combined_scale after conv1x1:", block_9_combined_scale1.item())
    print("combined_scale after conv3x3:", block_9_combined_scale2.item())
    print("combined_scale after conv1x1:", block_9_combined_scale3.item())
    print("combined_scale after skip add:", block_9_combined_scale_skip.item())
    print("********************BN9*******************************")
    scale_factors["BN9"]["conv1x1_1"] = int(block_9_combined_scale1.item())
    scale_factors["BN9"]["conv3x3"] = int(block_9_combined_scale2.item())
    scale_factors["BN9"]["conv1x1_2"] = int(block_9_combined_scale3.item())
    scale_factors["BN9"]["skip_add"] = int(block_9_combined_scale_skip.item() )

    block_10_relu_1 = quant_model.bn10_quant_relu1.act_quant.scale()
    block_10_relu_2 = quant_model.bn10_quant_relu2.act_quant.scale()
    block_10_final_scale = quant_model.bn10_quant_id_2.act_quant.scale()

    block_10_weight_scale1 = quant_model.bn10_quant_conv1.weight_quant.scale()
    block_10_weight_scale2 = quant_model.bn10_quant_conv2.weight_quant.scale()
    block_10_weight_scale3 = quant_model.bn10_quant_conv3.weight_quant.scale()

    block_10_combined_scale1 = -torch.log2(
        block_9_skip_add * block_10_weight_scale1 / block_10_relu_1
    )
    block_10_combined_scale2 = -torch.log2(
        block_10_relu_1 * block_10_weight_scale2 / block_10_relu_2
    )  
    block_10_combined_scale3 = -torch.log2(
        block_10_relu_2 * block_10_weight_scale3/block_10_final_scale
    )   
    block_11_relu_1 =       quant_model.bn11_quant_relu1.act_quant.scale()
    block_11_relu_2 =       quant_model.bn11_quant_relu2.act_quant.scale()
    block_11_skip_add =     quant_model.bn11_add.act_quant.scale()

    block_11_weight_scale1 = quant_model.bn11_quant_conv1.weight_quant.scale()
    block_11_weight_scale2 = quant_model.bn11_quant_conv2.weight_quant.scale()
    block_11_weight_scale3 = quant_model.bn11_quant_conv3.weight_quant.scale()
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

    block_12_relu_1 =        quant_model.bn12_quant_relu1.act_quant.scale()
    block_12_relu_2 =        quant_model.bn12_quant_relu2.act_quant.scale()
    block_12_final_scale =   quant_model.bn12_quant_id_2.act_quant.scale()

    block_12_weight_scale1 = quant_model.bn12_quant_conv1.weight_quant.scale()
    block_12_weight_scale2 = quant_model.bn12_quant_conv2.weight_quant.scale()
    block_12_weight_scale3 = quant_model.bn12_quant_conv3.weight_quant.scale()

    block_12_combined_scale1 = -torch.log2(
        block_11_skip_add * block_12_weight_scale1 / block_12_relu_1
    )
    block_12_combined_scale2 = -torch.log2(
        block_12_relu_1 * block_12_weight_scale2 / block_12_relu_2
    )  
    block_12_combined_scale3 = -torch.log2(
        block_12_relu_2 * block_12_weight_scale3/block_12_final_scale
    )   
    
    
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
    scale_factors["BN11"]["conv3x3"] = int(block_11_combined_scale2.item() )
    scale_factors["BN11"]["conv1x1_2"] = int(block_11_combined_scale3.item())
    scale_factors["BN11"]["skip_add"] = int (block_11_combined_scale_skip.item())


    print("********************BN12*******************************")
    print("combined_scale after conv1x1:", block_12_combined_scale1.item())
    print("combined_scale after conv3x3:", block_12_combined_scale2.item())
    print("combined_scale after conv1x1:", block_12_combined_scale3.item())
    scale_factors["BN12"]["conv1x1_1"] = int(block_12_combined_scale1.item())
    scale_factors["BN12"]["conv3x3"] = int(block_12_combined_scale2.item() )
    scale_factors["BN12"]["conv1x1_2"] = int(block_12_combined_scale3.item())

    block_13_relu_1 = quant_model.bn13_quant_relu1.act_quant.scale()
    block_13_relu_2 = quant_model.bn13_quant_relu2.act_quant.scale()
    block_13_skip_add = quant_model.bn13_add.act_quant.scale()

    block_13_weight_scale1 = quant_model.bn13_quant_conv1.weight_quant.scale()
    block_13_weight_scale2 = quant_model.bn13_quant_conv2.weight_quant.scale()
    block_13_weight_scale3 = quant_model.bn13_quant_conv3.weight_quant.scale()
    block_13_combined_scale1 = -torch.log2(
        block_12_final_scale * block_13_weight_scale1 / block_13_relu_1
    )
    block_13_combined_scale2 = -torch.log2(
        block_13_relu_1 * block_13_weight_scale2 / block_13_relu_2
    )  
    block_13_combined_scale3 = -torch.log2(
        block_13_relu_2 * block_13_weight_scale3/block_12_final_scale
    )   
    block_13_combined_scale_skip = -torch.log2(
        block_12_final_scale / block_13_skip_add
    )  # After addition | clip -128-->127

    block_14_relu_1 = quant_model.bn14_quant_relu1.act_quant.scale()
    block_14_relu_2 = quant_model.bn14_quant_relu2.act_quant.scale()
    block_14_skip_add = quant_model.bn14_add.act_quant.scale()

    block_14_weight_scale1 = quant_model.bn14_quant_conv1.weight_quant.scale()
    block_14_weight_scale2 = quant_model.bn14_quant_conv2.weight_quant.scale()
    block_14_weight_scale3 = quant_model.bn14_quant_conv3.weight_quant.scale()
    block_14_combined_scale1 = -torch.log2(
        block_13_skip_add * block_14_weight_scale1 / block_14_relu_1
    )
    block_14_combined_scale2 = -torch.log2(
        block_14_relu_1 * block_14_weight_scale2 / block_14_relu_2
    )  
    block_14_combined_scale3 = -torch.log2(
        block_14_relu_2 * block_14_weight_scale3/block_13_skip_add
    )   
    block_14_combined_scale_skip = -torch.log2(
        block_13_skip_add / block_14_skip_add
    )  # After addition | clip -128-->127

    atol=block_14_skip_add.item()
    print("********************BN13*******************************")
    print("combined_scale after conv1x1:", block_13_combined_scale1.item())
    print("combined_scale after conv3x3:", block_13_combined_scale2.item())
    print("combined_scale after conv1x1:", block_13_combined_scale3.item())
    print("combined_scale after skip add:", block_13_combined_scale_skip.item())
    scale_factors["BN13"]["conv1x1_1"] = int(block_13_combined_scale1.item())
    scale_factors["BN13"]["conv3x3"] = int(block_13_combined_scale2.item())
    scale_factors["BN13"]["conv1x1_2"] = int(block_13_combined_scale3.item())
    scale_factors["BN13"]["skip_add"] = int(block_13_combined_scale_skip.item())

    print("********************BN14*******************************")
    print("combined_scale after conv1x1:", block_14_combined_scale1.item())
    print("combined_scale after conv3x3:", block_14_combined_scale2.item())
    print("combined_scale after conv1x1:", block_14_combined_scale3.item())
    print("combined_scale after skip add:", block_14_combined_scale_skip.item())
    scale_factors["BN14"]["conv1x1_1"] = int(block_14_combined_scale1.item())
    scale_factors["BN14"]["conv3x3"] = int(block_14_combined_scale2.item())
    scale_factors["BN14"]["conv1x1_2"] = int(block_14_combined_scale3.item())
    scale_factors["BN14"]["skip_add"] = int(block_14_combined_scale_skip.item())

    post_weight_scale1 = quant_model.post_quant_conv1.weight_quant.scale()
    post_relu = quant_model.post_relu.act_quant.scale()
    post_combined_conv1_scale = -torch.log2(block_14_skip_add * post_weight_scale1 / post_relu)

    post_layer2_weight_scale1 = quant_model.post_layer2_quant_conv1.weight_quant.scale()
    post_layer2_relu = quant_model.post_layer2_relu.act_quant.scale()
    post_layer2_combined_conv1_scale = -torch.log2(post_relu * post_layer2_weight_scale1 / post_layer2_relu)
    print("********************POST*******************************")
    print("combined_scale after POST conv1x1:", post_combined_conv1_scale.item())
    print("combined_scale after POST 2 conv1x1:", post_layer2_combined_conv1_scale.item())
    scale_factors["POST"]["conv1x1_1"] = int(post_combined_conv1_scale.item() )
    scale_factors["POST"]["FC1"] = int(post_layer2_combined_conv1_scale.item() )

    post_layer3_weight_scale1 = quant_model.post_layer3_quant_conv1.weight_quant.scale()
    post_layer3_relu = quant_model.post_layer3_relu.act_quant.scale()
    post_layer3_combined_conv1_scale = -torch.log2(post_layer2_relu * post_layer3_weight_scale1 / post_layer3_relu)
    print("********************FC2*******************************")
    print("combined_scale after FC2 conv1x1:", post_layer3_combined_conv1_scale.item())
    scale_factors["POST"]["FC2"] = int(post_layer3_combined_conv1_scale.item() )


    # print("combined_scale after conv1x1:", ( block_0_relu_2 * block_0_weight_scale3).item())
    # Write the updated scale factors back to the file
    write_scale_factors(file_path, scale_factors)
    # ------------------------------------------------------
    # Reorder input data-layout
    # ------------------------------------------------------
    init_weight = quant_model.init_conv.quant_weight().int(float_datatype=True)

    block_0_int_weight_2 = quant_model.bn0_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_0_int_weight_3 = quant_model.bn0_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    block_1_int_weight_1 = quant_model.bn1_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_1_int_weight_2 = quant_model.bn1_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_1_int_weight_3 = quant_model.bn1_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    block_2_int_weight_1 = quant_model.bn2_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_2_int_weight_2 = quant_model.bn2_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_2_int_weight_3 = quant_model.bn2_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    block_3_int_weight_1 = quant_model.bn3_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_3_int_weight_2 = quant_model.bn3_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_3_int_weight_3 = quant_model.bn3_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    block_4_int_weight_1 = quant_model.bn4_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_4_int_weight_2 = quant_model.bn4_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_4_int_weight_3 = quant_model.bn4_quant_conv3.quant_weight().int(
        float_datatype=True
    )


    block_5_int_weight_1 = quant_model.bn5_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_5_int_weight_2 = quant_model.bn5_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_5_int_weight_3 = quant_model.bn5_quant_conv3.quant_weight().int(
        float_datatype=True
    )


    block_6_int_weight_1 = quant_model.bn6_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_6_int_weight_2 = quant_model.bn6_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_6_int_weight_3 = quant_model.bn6_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    block_7_int_weight_1 = quant_model.bn7_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_7_int_weight_2 = quant_model.bn7_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_7_int_weight_3 = quant_model.bn7_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    block_8_int_weight_1 = quant_model.bn8_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_8_int_weight_2 = quant_model.bn8_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_8_int_weight_3 = quant_model.bn8_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    block_9_int_weight_1 = quant_model.bn9_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_9_int_weight_2 = quant_model.bn9_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_9_int_weight_3 = quant_model.bn9_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    block_10_int_weight_1 = quant_model.bn10_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_10_int_weight_2 = quant_model.bn10_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_10_int_weight_3 = quant_model.bn10_quant_conv3.quant_weight().int(
        float_datatype=True
    )
  
    block_11_int_weight_1 = quant_model.bn11_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_11_int_weight_2 = quant_model.bn11_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_11_int_weight_3 = quant_model.bn11_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    block_12_int_weight_1 = quant_model.bn12_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_12_int_weight_2 = quant_model.bn12_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_12_int_weight_3 = quant_model.bn12_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    block_13_int_weight_1 = quant_model.bn13_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_13_int_weight_2 = quant_model.bn13_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_13_int_weight_3 = quant_model.bn13_quant_conv3.quant_weight().int(
        float_datatype=True
    )


    block_14_int_weight_1 = quant_model.bn14_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_14_int_weight_2 = quant_model.bn14_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_14_int_weight_3 = quant_model.bn14_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    post_conv1_int_weight = quant_model.post_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    post_layer2_conv1_int_weight = quant_model.post_layer2_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    post_layer3_conv1_int_weight = quant_model.post_layer3_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    print("FC1 weights shape::", post_layer2_conv1_int_weight.shape)
    print("FC2 weights shape::", post_layer3_conv1_int_weight.shape)
    padded_post_layer2_conv1_int_weight = pad_weights(post_layer2_conv1_int_weight, target_weights_shape)
    padded_post_layer3_conv1_int_weight = pad_weights(post_layer3_conv1_int_weight, target_weights_shape)

    print("Padded FC1 weights shape::", padded_post_layer2_conv1_int_weight.shape)
    print("Padded FC2 weights shape::", padded_post_layer3_conv1_int_weight.shape)

  
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
    # **************************** init ****************************
    init_wts_fmt = ds.reorder_mat(
        init_weight.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    # **************************** bn0 ****************************
    bn0_wts2 = ds.reorder_mat(
        block_0_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn0_wts3 = ds.reorder_mat(
        block_0_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn0_total_wts = np.concatenate((bn0_wts2, bn0_wts3), axis=None)

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

    
    bn1_total_wts = np.concatenate((bn1_wts1, bn1_wts2, bn1_wts3), axis=None)
    


    bn01_total_wts = np.concatenate(( bn0_total_wts,bn1_total_wts), axis=None)
    # **************************** bn2 ****************************
    bn2_wts1 = ds.reorder_mat(
        block_2_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn2_wts2 = ds.reorder_mat(
        block_2_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn2_wts3 = ds.reorder_mat(
        block_2_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )

    bn2_total_wts = np.concatenate((bn2_wts1, bn2_wts2, bn2_wts3), axis=None)


    # **************************** bn3 ****************************
    bn3_wts1 = ds.reorder_mat(
        block_3_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn3_wts2 = ds.reorder_mat(
        block_3_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn3_wts3 = ds.reorder_mat(
        block_3_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )

    bn3_total_wts = np.concatenate((bn3_wts1, bn3_wts2, bn3_wts3), axis=None)
    
    # **************************** bn4 ****************************
    bn4_wts1 = ds.reorder_mat(
        block_4_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn4_wts2 = ds.reorder_mat(
        block_4_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn4_wts3 = ds.reorder_mat(
        block_4_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn4_total_wts = np.concatenate((bn4_wts1, bn4_wts2, bn4_wts3), axis=None)

    
    # **************************** bn5 ****************************
    bn5_wts1 = ds.reorder_mat(
        block_5_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn5_wts2 = ds.reorder_mat(
        block_5_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn5_wts3 = ds.reorder_mat(
        block_5_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn5_total_wts = np.concatenate((bn5_wts1, bn5_wts2, bn5_wts3), axis=None)
    bn4_5_total_wts = np.concatenate((bn4_total_wts,bn5_total_wts), axis=None)
    

    # **************************** bn6 ****************************
    bn6_wts1 = ds.reorder_mat(
        block_6_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn6_wts2 = ds.reorder_mat(
        block_6_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn6_wts3 = ds.reorder_mat(
        block_6_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn6_total_wts = np.concatenate((bn6_wts1, bn6_wts2, bn6_wts3), axis=None)
    
    # **************************** bn7 ****************************
    bn7_wts1 = ds.reorder_mat(
        block_7_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn7_wts2 = ds.reorder_mat(
        block_7_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn7_wts3 = ds.reorder_mat(
        block_7_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn7_total_wts = np.concatenate((bn7_wts1, bn7_wts2, bn7_wts3), axis=None)
    
     # **************************** bn8 ****************************
    bn8_wts1 = ds.reorder_mat(
        block_8_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn8_wts2 = ds.reorder_mat(
        block_8_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn8_wts3 = ds.reorder_mat(
        block_8_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )

    bn8_total_wts = np.concatenate((bn8_wts1, bn8_wts2, bn8_wts3), axis=None)
    

     # **************************** bn9 ****************************
    bn9_wts1 = ds.reorder_mat(
        block_9_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn9_wts2 = ds.reorder_mat(
        block_9_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn9_wts3 = ds.reorder_mat(
        block_9_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )

    bn9_total_wts = np.concatenate((bn9_wts1, bn9_wts2, bn9_wts3), axis=None)
    
    bn8_9_total_wts = np.concatenate((bn8_total_wts,bn9_total_wts), axis=None)
    
    # **************************** bn10 ****************************
    bn10_wts1 = ds.reorder_mat(
        block_10_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn10_wts2 = ds.reorder_mat(
        block_10_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn10_wts3 = ds.reorder_mat(
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
    bn10_total_wts = np.concatenate((bn10_wts1, bn10_wts2, bn10_wts3), axis=None)
    bn11_total_wts = np.concatenate((bn11_wts1, bn11_wts2, bn11_wts3), axis=None)
    bn12_total_wts = np.concatenate((bn12_wts1, bn12_wts2, bn12_wts3), axis=None)
    bn12_wts2_3 = np.concatenate((bn12_wts2,bn12_wts3), axis=None)
    b_block_total_wts = np.concatenate((bn10_total_wts,bn11_total_wts,bn12_total_wts), axis=None)

    
    
    # **************************** bn13 ****************************
    # bn13_wts1 = ds.reorder_mat(
    #     block_13_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    # )
    bn13_wts1 = reorder_and_concatenate_chunks(block_13_int_weight_1, bneck_13_InC1, WeightChunks, ds, dtype_wts)
    bn13_wts2 = ds.reorder_mat(block_13_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX")
    # bn13_wts3 = ds.reorder_mat(
    #     block_13_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    # )
    bn13_wts3_put = ds.reorder_mat(block_13_int_weight_3[:,0:bneck_13_OutC2//2,:,:].data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX")
    bn13_wts3_get = ds.reorder_mat(block_13_int_weight_3[:,bneck_13_OutC2//2:bneck_13_OutC2,:,:].data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    # **************************** bn14 ****************************
    bn14_wts1 = reorder_and_concatenate_chunks(block_14_int_weight_1, bneck_13_InC1, WeightChunks, ds, dtype_wts)
    bn14_wts2 = ds.reorder_mat(block_14_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX")
    # bn13_wts3 = ds.reorder_mat(
    #     block_13_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    # )
    bn14_wts3_put = ds.reorder_mat(block_14_int_weight_3[:,0:bneck_13_OutC2//2,:,:].data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX")
    bn14_wts3_get = ds.reorder_mat(block_14_int_weight_3[:,bneck_13_OutC2//2:bneck_13_OutC2,:,:].data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    c_total_wts = np.concatenate((bn13_wts1,bn13_wts3_put,bn13_wts3_get,bn14_wts1,bn14_wts3_put,bn14_wts3_get), axis=None)

    init_wts_fmt.tofile(weight_folder + "/init_chain.txt", sep=",", format="%d")
    bn0_total_wts.tofile(weight_folder + "/bn0_chain.txt", sep=",", format="%d")
    bn1_total_wts.tofile(weight_folder + "/bn1_chain.txt", sep=",", format="%d")
    bn01_total_wts.tofile(weight_folder + "/bn0_1_chain.txt", sep=",", format="%d")
    bn2_total_wts.tofile(weight_folder + "/bn2_chain.txt", sep=",", format="%d")
    bn3_total_wts.tofile(weight_folder + "/bn3_chain.txt", sep=",", format="%d")
    bn4_total_wts.tofile(weight_folder + "/bn4_chain.txt", sep=",", format="%d")
    bn5_total_wts.tofile(weight_folder + "/bn5_chain.txt", sep=",", format="%d")
    bn4_5_total_wts.tofile(weight_folder + "/bn4_5_chain.txt", sep=",", format="%d")
    bn6_total_wts.tofile(weight_folder + "/bn6_chain.txt", sep=",", format="%d")
    bn7_total_wts.tofile(weight_folder + "/bn7_chain.txt", sep=",", format="%d")
    bn8_total_wts.tofile(weight_folder + "/bn8_chain.txt", sep=",", format="%d")
    bn9_total_wts.tofile(weight_folder + "/bn9_chain.txt", sep=",", format="%d")
    bn8_9_total_wts.tofile(weight_folder + "/bn8_9_chain.txt", sep=",", format="%d")
    # 
    bn10_wts1.tofile(weight_folder + "/bn10_1_chain.txt", sep=",", format="%d")
    bn10_wts2.tofile(weight_folder + "/bn10_2_chain.txt", sep=",", format="%d")
    bn10_wts3.tofile(weight_folder + "/bn10_3_chain.txt", sep=",", format="%d")
    bn11_wts1.tofile(weight_folder + "/bn11_1_chain.txt", sep=",", format="%d")
    bn11_wts2.tofile(weight_folder + "/bn11_2_chain.txt", sep=",", format="%d")
    bn11_wts3.tofile(weight_folder + "/bn11_3_chain.txt", sep=",", format="%d")
    bn12_wts1.tofile(weight_folder + "/bn12_1_chain.txt", sep=",", format="%d")
    bn12_wts2.tofile(weight_folder + "/bn12_2_chain.txt", sep=",", format="%d")
    bn12_wts3.tofile(weight_folder + "/bn12_3_chain.txt", sep=",", format="%d")
    bn12_wts2_3.tofile(weight_folder + "/bn12_2_3_chain.txt", sep=",", format="%d")

    # 
    bn13_wts1.tofile(weight_folder + "/bn13_1_chain.txt", sep=",", format="%d")
    bn13_wts2.tofile(weight_folder + "/bn13_2_chain.txt", sep=",", format="%d")
    bn13_wts3_put.tofile(weight_folder + "/bn13_3_put_chain.txt", sep=",", format="%d")
    bn13_wts3_get.tofile(weight_folder + "/bn13_3_get_chain.txt", sep=",", format="%d")
    bn14_wts1.tofile(weight_folder + "/bn14_1_chain.txt", sep=",", format="%d")
    bn14_wts2.tofile(weight_folder + "/bn14_2_chain.txt", sep=",", format="%d")
    bn14_wts3_put.tofile(weight_folder + "/bn14_3_put_chain.txt", sep=",", format="%d")
    bn14_wts3_get.tofile(weight_folder + "/bn14_3_get_chain.txt", sep=",", format="%d")


    post_conv1_int_weight_fmt = ds.reorder_mat(post_conv1_int_weight.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX")
    post_conv1_int_weight_fmt.tofile(weight_folder + "/post_conv_chain.txt", sep=",", format="%d")
    
    # Split weights for 4 cores
    O_total = padded_post_layer2_conv1_int_weight.shape[0]
    I_total = padded_post_layer2_conv1_int_weight.shape[1]
   

    post_layer2_conv1_weights_splits = [padded_post_layer2_conv1_int_weight[k * (O_total // 4):(k + 1) * (O_total // 4), ...] for k in range(4)]
    post_layer2_conv1_int_weight_fmt = [ds.reorder_mat(weight_split.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX") for weight_split in post_layer2_conv1_weights_splits]
    
    post_layer3_conv1_weights_splits = [padded_post_layer3_conv1_int_weight[k * (O_total // 4):(k + 1) * (O_total // 4), ...] for k in range(4)]
    post_layer3_conv1_int_weight_fmt = [ds.reorder_mat(weight_split.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX") for weight_split in post_layer3_conv1_weights_splits]

    # post_layer2_conv1_int_weight_fmt = ds.reorder_mat(padded_post_layer2_conv1_int_weight.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX")
    # post_layer3_conv1_int_weight_fmt = ds.reorder_mat(padded_post_layer3_conv1_int_weight.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX")
    total_wts = np.concatenate((c_total_wts), axis=None)

    np.concatenate((post_layer2_conv1_int_weight_fmt), axis=None).tofile(weight_folder + "/FC1_chain.txt", sep=",", format="%d")
    post_layer2_conv1_int_weight_fmt[0].tofile(weight_folder + "/FC1_0_chain.txt", sep=",", format="%d")
    post_layer2_conv1_int_weight_fmt[1].tofile(weight_folder + "/FC1_1_chain.txt", sep=",", format="%d")
    post_layer2_conv1_int_weight_fmt[2].tofile(weight_folder + "/FC1_2_chain.txt", sep=",", format="%d")
    post_layer2_conv1_int_weight_fmt[3].tofile(weight_folder + "/FC1_3_chain.txt", sep=",", format="%d")
    np.concatenate((post_layer3_conv1_int_weight_fmt), axis=None).tofile(weight_folder + "/FC2_chain.txt", sep=",", format="%d")
    post_layer3_conv1_int_weight_fmt[0].tofile(weight_folder + "/FC2_0_chain.txt", sep=",", format="%d")
    post_layer3_conv1_int_weight_fmt[1].tofile(weight_folder + "/FC2_1_chain.txt", sep=",", format="%d")
    post_layer3_conv1_int_weight_fmt[2].tofile(weight_folder + "/FC2_2_chain.txt", sep=",", format="%d")
    post_layer3_conv1_int_weight_fmt[3].tofile(weight_folder + "/FC2_3_chain.txt", sep=",", format="%d")

    
    total_wts.tofile(log_folder + "/after_weights_mem_fmt_final.txt", sep=",", format="%d")
    # print("{}+{}+{}".format(bn6_wts1.shape, bn6_wts2.shape, bn6_wts3.shape))
    print(shape_total_wts)
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
    # print("Golden::Brevitas::", golden_output)
    # print("AIE::", ofm_mem_fmt_out)
    # ------------------------------------------------------
    # Compare the AIE output and the golden reference
    # ------------------------------------------------------
    print("\nAvg NPU time: {}us.".format(int((npu_time_total / num_iter) / 1000)))
    golden=convert_to_numpy(padded_golden_output)
    ofm_mem_fmt_out=convert_to_numpy(ofm_mem_fmt_out)
    max_difference = np.max((golden)-(ofm_mem_fmt_out))
    print("max_difference:",max_difference)
        # Find the indices where the mismatch happens
    # Find the indices where the mismatch happens
    mismatch_indices = np.where(golden != ofm_mem_fmt_out)

    # Extract mismatch values
    mismatch_values_golden = golden[mismatch_indices]
    mismatch_values_ofm = ofm_mem_fmt_out[mismatch_indices]

    # Print mismatch indices and corresponding values
    print("golden shape: ",golden.shape)
    print("Output shape: ",ofm_mem_fmt_out.shape)
    # print("Mismatch indices and corresponding values:")
    # for idx, (golden_value, ofm_value) in zip(zip(*mismatch_indices), zip(mismatch_values_golden, mismatch_values_ofm)):
    #     print(f"Index: {idx}, Golden value: {golden_value}, OFM value: {ofm_value}")
    if np.allclose(
        ofm_mem_fmt_out,
        golden_output,
        rtol=0,
        atol=3,
    ):
        print("\nPASS!\n")
        print_three_dolphins()
        exit(0)
    else:
        print("\nFailed.\n")
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
