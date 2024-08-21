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
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
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


tensorInW = 112
tensorInH = 112
tensorInC = 16

bneck_0_InW2 = tensorInW
bneck_0_InH2 = tensorInH
bneck_0_InC2 = tensorInC
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


# tensorOutW = bneck_3_InW3 
# tensorOutH = bneck_3_InH3
# tensorOutC = bneck_3_OutC3

tensorOutW = bneck_11_InW3 
tensorOutH = bneck_11_InH3
tensorOutC = bneck_11_OutC3

InC_vec =  math.floor(tensorInC/vectorSize)
OutC_vec =  math.floor(tensorOutC/vectorSize)


def main(opts):
    design = "mobilenet_bottleneck_A_chain"
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
    dtype_out = np.dtype("int8")

    shape_total_wts =(
                        (3*3*bneck_0_OutC2 + bneck_0_OutC2*bneck_0_OutC3 + bneck_1_InC1*bneck_1_OutC1 + 3*3*bneck_1_OutC2 + bneck_1_OutC2*bneck_1_OutC3)+
                        (bneck_2_InC1*bneck_2_OutC1 + 3*3*bneck_2_OutC2 + bneck_2_OutC2*bneck_2_OutC3)+
                        (bneck_3_InC1*bneck_3_OutC1 + 3*3*bneck_3_OutC2 + bneck_3_OutC2*bneck_3_OutC3)+
                        (bneck_4_InC1*bneck_4_OutC1 + 3*3*bneck_4_OutC2 + bneck_4_OutC2*bneck_4_OutC3)+
                        (bneck_5_InC1*bneck_5_OutC1 + 3*3*bneck_5_OutC2 + bneck_5_OutC2*bneck_5_OutC3)+
                        (bneck_6_InC1*bneck_6_OutC1 + 3*3*bneck_6_OutC2 + bneck_6_OutC2*bneck_6_OutC3)+
                        (bneck_7_InC1*bneck_7_OutC1 + 3*3*bneck_7_OutC2 + bneck_7_OutC2*bneck_7_OutC3)+
                        (bneck_8_InC1*bneck_8_OutC1 + 3*3*bneck_8_OutC2 + bneck_8_OutC2*bneck_8_OutC3)+
                        (bneck_9_InC1*bneck_9_OutC1 + 3*3*bneck_9_OutC2 + bneck_9_OutC2*bneck_9_OutC3)+
                        (bneck_10_InC1*bneck_10_OutC1)+(3*3*bneck_10_OutC2)+(bneck_10_OutC2*bneck_10_OutC3)+
                        (bneck_10_OutC3*bneck_11_OutC1)+(3*3*bneck_11_OutC2)+(bneck_11_OutC2*bneck_11_OutC3),1)
    
    print("total weights:::",shape_total_wts)
    shape_in_act = (tensorInH, InC_vec, tensorInW, vectorSize)  #'YCXC8' , 'CYX'
    shape_out = (tensorOutH, OutC_vec, tensorOutW, vectorSize) # HCWC8
    shape_out_final = (OutC_vec*vectorSize, tensorOutH, tensorOutW) # CHW
   
    # ------------------------------------------------------
    # Get device, load the xclbin & kernel and register them
    # ------------------------------------------------------
    print(xclbin_path)
    print(insts_path)
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
    class QuantBottleneckA(nn.Module):
        def __init__(self, in_planes=16,
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
                     bn12_expand=16,bn12_project=16):
            super(QuantBottleneckA, self).__init__()
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

            # 
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

        def forward(self, x):
            out_q = self.quant_id_1(x)
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

            # # # # # bn2
            out = self.bn2_quant_conv1(out_q)
            out = self.bn2_quant_relu1(out)
            out = self.bn2_quant_conv2(out)
            out = self.bn2_quant_relu2(out)
            out = self.bn2_quant_conv3(out)
            out = self.bn1_quant_id_2(out)
            out = out+out_q
            out = self.bn2_add(out)

            # # # # # bn3
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
            # out = self.bn4_add(out)
            # out = out+out_q
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
            # out = self.bn7_add(out)
            # out = out+out_q
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
            # # bn12
            # out = self.bn12_quant_conv1(out)
            # out = self.bn12_quant_relu1(out)
            # out = self.bn12_quant_conv2(out)
            # out = self.bn12_quant_relu2(out)
            # out = self.bn12_quant_conv3(out)
            # out = self.bn12_quant_id_2(out)


            return out

    quant_model = QuantBottleneckA(in_planes=tensorInC, 
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
                                            bn11_expand=bneck_11_OutC2,bn11_project=bneck_11_OutC3)
    from utils import ExpandChannels
    from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
    import torchvision
    import torch.utils.data as data_utils
    from torchvision import transforms
    # # Define the image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(tensorInW),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ExpandChannels(target_channels=16)  # Expand to 80 channels
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
    # quant_bottleneck_model = GGQuantBottleneck(in_planes=bneck_2_InC1, bn2_expand=bneck_2_OutC1,bn2_project=bneck_2_OutC3)

    calibrate(calib_loader, quant_model)
    quant_model.eval()

    input = torch.ones(1, InC_vec*vectorSize, tensorInH, tensorInW)

    q_bottleneck_out = quant_model(input)
    golden_output = q_bottleneck_out.int(float_datatype=True).data.numpy().astype(dtype_out)
    
    q_inp = quant_model.quant_id_1(input)
    int_inp = q_inp.int(float_datatype=True)

    
    block_0_inp_scale= quant_model.quant_id_1.quant_act_scale()

    block_0_relu_2 = quant_model.bn0_quant_relu2.quant_act_scale()
    block_0_skip_add = quant_model.bn0_add.quant_act_scale()

    block_0_weight_scale2 = quant_model.bn0_quant_conv2.quant_weight_scale()
    block_0_weight_scale3 = quant_model.bn0_quant_conv3.quant_weight_scale()

    block_0_combined_scale2 = -torch.log2(
        block_0_inp_scale * block_0_weight_scale2 / block_0_relu_2
    )  
    block_0_combined_scale3 = -torch.log2(
        block_0_relu_2 * block_0_weight_scale3/block_0_inp_scale
    )   
    block_0_combined_scale_skip = -torch.log2(
        block_0_inp_scale / block_0_skip_add
    )  # After addition | clip -128-->127

    block_1_relu_1 = quant_model.bn1_quant_relu1.quant_act_scale()
    block_1_relu_2 = quant_model.bn1_quant_relu2.quant_act_scale()
    block_1_final_scale = quant_model.bn1_quant_id_2.quant_act_scale()

    block_1_weight_scale1 = quant_model.bn1_quant_conv1.quant_weight_scale()
    block_1_weight_scale2 = quant_model.bn1_quant_conv2.quant_weight_scale()
    block_1_weight_scale3 = quant_model.bn1_quant_conv3.quant_weight_scale()
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

    block_2_relu_1 = quant_model.bn2_quant_relu1.quant_act_scale()
    block_2_relu_2 = quant_model.bn2_quant_relu2.quant_act_scale()

    block_2_skip_add = quant_model.bn2_add.quant_act_scale()

    block_2_weight_scale1 = quant_model.bn2_quant_conv1.quant_weight_scale()
    block_2_weight_scale2 = quant_model.bn2_quant_conv2.quant_weight_scale()
    block_2_weight_scale3 = quant_model.bn2_quant_conv3.quant_weight_scale()
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
    block_3_relu_1 = quant_model.bn3_quant_relu1.quant_act_scale()
    block_3_relu_2 = quant_model.bn3_quant_relu2.quant_act_scale()
    block_3_final_scale = quant_model.bn3_quant_id_2.quant_act_scale()

    block_3_weight_scale1 = quant_model.bn3_quant_conv1.quant_weight_scale()
    block_3_weight_scale2 = quant_model.bn3_quant_conv2.quant_weight_scale()
    block_3_weight_scale3 = quant_model.bn3_quant_conv3.quant_weight_scale()
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
    block_4_relu_1 = quant_model.bn4_quant_relu1.quant_act_scale()
    block_4_relu_2 = quant_model.bn4_quant_relu2.quant_act_scale()
    block_4_skip_add = quant_model.bn4_add.quant_act_scale()

    block_4_weight_scale1 = quant_model.bn4_quant_conv1.quant_weight_scale()
    block_4_weight_scale2 = quant_model.bn4_quant_conv2.quant_weight_scale()
    block_4_weight_scale3 = quant_model.bn4_quant_conv3.quant_weight_scale()
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
    block_5_relu_1 = quant_model.bn5_quant_relu1.quant_act_scale()
    block_5_relu_2 = quant_model.bn5_quant_relu2.quant_act_scale()
    block_5_skip_add = quant_model.bn5_add.quant_act_scale()

    block_5_weight_scale1 = quant_model.bn5_quant_conv1.quant_weight_scale()
    block_5_weight_scale2 = quant_model.bn5_quant_conv2.quant_weight_scale()
    block_5_weight_scale3 = quant_model.bn5_quant_conv3.quant_weight_scale()
    block_5_combined_scale1 = -torch.log2(
        block_5_inp_scale1 * block_5_weight_scale1 / block_5_relu_1
    )
    block_5_combined_scale2 = -torch.log2(
        block_5_relu_1 * block_5_weight_scale2 / block_5_relu_2
    )  
    block_5_combined_scale3 = -torch.log2(
        block_5_relu_2 * block_5_weight_scale3/block_5_skip_add
    )   
    block_5_combined_scale_skip = -torch.log2(
        block_5_skip_add / block_5_skip_add
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



    block_6_relu_1 = quant_model.bn6_quant_relu1.quant_act_scale()
    block_6_relu_2 = quant_model.bn6_quant_relu2.quant_act_scale()
    block_6_final_scale = quant_model.bn6_quant_id_2.quant_act_scale()

    block_6_weight_scale1 = quant_model.bn6_quant_conv1.quant_weight_scale()
    block_6_weight_scale2 = quant_model.bn6_quant_conv2.quant_weight_scale()
    block_6_weight_scale3 = quant_model.bn6_quant_conv3.quant_weight_scale()
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

    block_7_relu_1 = quant_model.bn7_quant_relu1.quant_act_scale()
    block_7_relu_2 = quant_model.bn7_quant_relu2.quant_act_scale()
    block_7_skip_add = quant_model.bn7_add.quant_act_scale()

    block_7_weight_scale1 = quant_model.bn7_quant_conv1.quant_weight_scale()
    block_7_weight_scale2 = quant_model.bn7_quant_conv2.quant_weight_scale()
    block_7_weight_scale3 = quant_model.bn7_quant_conv3.quant_weight_scale()
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
    block_8_relu_1 = quant_model.bn8_quant_relu1.quant_act_scale()
    block_8_relu_2 = quant_model.bn8_quant_relu2.quant_act_scale()
    block_8_skip_add = quant_model.bn8_add.quant_act_scale()
    block_8_weight_scale1 = quant_model.bn8_quant_conv1.quant_weight_scale()
    block_8_weight_scale2 = quant_model.bn8_quant_conv2.quant_weight_scale()
    block_8_weight_scale3 = quant_model.bn8_quant_conv3.quant_weight_scale()
    block_8_combined_scale1 = -torch.log2(
        block_8_inp_scale1 * block_8_weight_scale1 / block_8_relu_1
    )
    block_8_combined_scale2 = -torch.log2(
        block_8_relu_1 * block_8_weight_scale2 / block_8_relu_2
    )  
    block_8_combined_scale3 = -torch.log2(
        block_8_relu_2 * block_8_weight_scale3/block_8_skip_add
    )   
    block_8_combined_scale_skip = -torch.log2(
        block_8_skip_add / block_8_skip_add
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
    block_9_relu_1 = quant_model.bn9_quant_relu1.quant_act_scale()
    block_9_relu_2 = quant_model.bn9_quant_relu2.quant_act_scale()
    block_9_skip_add = quant_model.bn9_add.quant_act_scale()
    block_9_weight_scale1 = quant_model.bn9_quant_conv1.quant_weight_scale()
    block_9_weight_scale2 = quant_model.bn9_quant_conv2.quant_weight_scale()
    block_9_weight_scale3 = quant_model.bn9_quant_conv3.quant_weight_scale()
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

    block_10_relu_1 = quant_model.bn10_quant_relu1.quant_act_scale()
    block_10_relu_2 = quant_model.bn10_quant_relu2.quant_act_scale()
    block_10_final_scale = quant_model.bn10_quant_id_2.quant_act_scale()

    block_10_weight_scale1 = quant_model.bn10_quant_conv1.quant_weight_scale()
    block_10_weight_scale2 = quant_model.bn10_quant_conv2.quant_weight_scale()
    block_10_weight_scale3 = quant_model.bn10_quant_conv3.quant_weight_scale()

    block_10_combined_scale1 = -torch.log2(
        block_9_skip_add * block_10_weight_scale1 / block_10_relu_1
    )
    block_10_combined_scale2 = -torch.log2(
        block_10_relu_1 * block_10_weight_scale2 / block_10_relu_2
    )  
    block_10_combined_scale3 = -torch.log2(
        block_10_relu_2 * block_10_weight_scale3/block_10_final_scale
    )   
    block_11_relu_1 =       quant_model.bn11_quant_relu1.quant_act_scale()
    block_11_relu_2 =       quant_model.bn11_quant_relu2.quant_act_scale()
    block_11_skip_add =     quant_model.bn11_add.quant_act_scale()

    block_11_weight_scale1 = quant_model.bn11_quant_conv1.quant_weight_scale()
    block_11_weight_scale2 = quant_model.bn11_quant_conv2.quant_weight_scale()
    block_11_weight_scale3 = quant_model.bn11_quant_conv3.quant_weight_scale()
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
    
    

    write_scale_factors(file_path, scale_factors)
    # ------------------------------------------------------
    # Reorder input data-layout
    # ------------------------------------------------------
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

    # block_12_int_weight_1 = quant_model.bn12_quant_conv1.quant_weight().int(
    #     float_datatype=True
    # )
    # block_12_int_weight_2 = quant_model.bn12_quant_conv2.quant_weight().int(
    #     float_datatype=True
    # )
    # block_12_int_weight_3 = quant_model.bn12_quant_conv3.quant_weight().int(
    #     float_datatype=True
    # )
  
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

    bn01_total_wts = np.concatenate((bn0_wts2, bn0_wts3, bn1_wts1, bn1_wts2, bn1_wts3), axis=None)
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
    # bn12_wts1 = ds.reorder_mat(
    #     block_12_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    # )
    # bn12_wts2 = ds.reorder_mat(
    #     block_12_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    # )
    # bn12_wts3 = ds.reorder_mat(
    #     block_12_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    # )
    bn10_total_wts = np.concatenate((bn10_wts1, bn10_wts2, bn10_wts3), axis=None)
    bn11_total_wts = np.concatenate((bn11_wts1, bn11_wts2, bn11_wts3), axis=None)
    # bn12_total_wts = np.concatenate((bn12_wts1, bn12_wts2, bn12_wts3), axis=None)
    b_block_total_wts = np.concatenate((bn10_total_wts,bn11_total_wts), axis=None)
    

    total_wts = np.concatenate((bn01_total_wts,bn2_total_wts,bn3_total_wts,
                                bn4_total_wts,bn5_total_wts,bn6_total_wts,
                                bn7_total_wts,bn8_total_wts,bn9_total_wts,
                                b_block_total_wts), axis=None)

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
    print("Golden::Brevitas::", golden_output)
    print("AIE::", ofm_mem_fmt_out)
    # ------------------------------------------------------
    # Compare the AIE output and the golden reference
    # ------------------------------------------------------
    print("\nAvg NPU time: {}us.".format(int((npu_time_total / num_iter) / 1000)))

    zeros_tensor = torch.zeros_like(ofm_mem_fmt_out)
    is_all_zero = torch.allclose(ofm_mem_fmt_out, zeros_tensor)
    print("is_all_zero:",is_all_zero)
    golden=convert_to_numpy(golden_output)
    ofm_mem_fmt_out=convert_to_numpy(ofm_mem_fmt_out)
    max_difference = np.max((golden)-(ofm_mem_fmt_out))
    print("Error between AIE and Golden Brevitas:",max_difference)
            # Find indices where the arrays differ
    print(golden.shape)
    if golden.shape != ofm_mem_fmt_out.shape:
        raise ValueError("The input arrays must have the same shape")

    tolerance = 6
    different_indices = np.argwhere(np.abs(golden - ofm_mem_fmt_out) > tolerance)

    if np.allclose(
        ofm_mem_fmt_out,
        golden_output,
        rtol=0,
        atol=9,
    ):
        print("\nPASS!\n")
        print_three_dolphins()
        exit(0)
    else:
        print("\nFailed.\n")
        for index in different_indices:
            idx_tuple = tuple(index)
            # print(f"Index {idx_tuple}: GOLDEN has {golden_output[idx_tuple]}, AIE has {ofm_mem_fmt_out[idx_tuple]}, diff {np.abs(golden[idx_tuple] - ofm_mem_fmt_out[idx_tuple])}")
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)