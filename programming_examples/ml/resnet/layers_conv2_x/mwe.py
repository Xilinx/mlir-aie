#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

import torch
import torch.nn as nn
import sys
import time
import os
import numpy as np

from brevitas.nn import QuantConv2d, QuantIdentity, QuantReLU
from brevitas.quant.fixed_point import (
    Int8ActPerTensorFixedPoint,
    Int8WeightPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
input = torch.randn(1, 64, 32, 32)


# ------------------------------------------------------
# Define your golden reference
# ------------------------------------------------------
class QuantBottleneck_projected(nn.Module):
    expansion = 4

    def __init__(self, in_planes=64, planes=64):
        super(QuantBottleneck_projected, self).__init__()
        # block 0
        self.quant_id_1 = QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.quant_block_0_conv1 = QuantConv2d(
            in_planes,
            planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        self.quant_block_0_conv2 = QuantConv2d(
            planes,
            planes,
            kernel_size=3,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            padding=1,
            padding_mode="zeros",
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        self.quant_block_0_conv3 = QuantConv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        self.quant_block_0_relu1 = QuantReLU(
            act_quant=Uint8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.quant_block_0_relu2 = QuantReLU(
            act_quant=Uint8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.quant_block_0_relu3 = QuantReLU(
            act_quant=Uint8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )

        self.shortcut = QuantConv2d(
            in_planes,
            self.expansion * planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )

        # block 1
        self.quant_block_1_conv1 = QuantConv2d(
            self.expansion * in_planes,
            planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        self.quant_block_1_conv2 = QuantConv2d(
            planes,
            planes,
            kernel_size=3,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            padding=1,
            padding_mode="zeros",
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        self.quant_block_1_conv3 = QuantConv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        self.quant_block_1_relu1 = QuantReLU(
            act_quant=Uint8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.quant_block_1_relu2 = QuantReLU(
            act_quant=Uint8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.quant_block_1_relu3 = QuantReLU(
            act_quant=Uint8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )

        self.quant_add_1 = QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        # Quant_add_1 shares the scale factors with block_0_relu3, however one is signed and the other one is unsigned
        self.quant_add_1.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl = (
            self.quant_block_0_relu3.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl
        )
        self.quant_add_1.act_quant.fused_activation_quant_proxy.tensor_quant.int_scaling_impl = (
            self.quant_block_0_relu3.act_quant.fused_activation_quant_proxy.tensor_quant.int_scaling_impl
        )

        # block 2
        self.quant_block_2_conv1 = QuantConv2d(
            self.expansion * in_planes,
            planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        self.quant_block_2_conv2 = QuantConv2d(
            planes,
            planes,
            kernel_size=3,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            padding=1,
            padding_mode="zeros",
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        self.quant_block_2_conv3 = QuantConv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        self.quant_block_2_relu1 = QuantReLU(
            act_quant=Uint8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.quant_block_2_relu2 = QuantReLU(
            act_quant=Uint8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.quant_block_2_relu3 = QuantReLU(
            act_quant=Uint8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )

        self.quant_add_2 = QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        # Quant_add_1 shares the scale factors with block_0_relu3, however one is signed and the other one is unsigned
        self.quant_add_2.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl = (
            self.quant_block_1_relu3.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl
        )
        self.quant_add_2.act_quant.fused_activation_quant_proxy.tensor_quant.int_scaling_impl = (
            self.quant_block_1_relu3.act_quant.fused_activation_quant_proxy.tensor_quant.int_scaling_impl
        )

    def forward(self, x):
        out_q = self.quant_id_1(x)
        out_rhs = self.quant_block_0_conv1(out_q)
        out_rhs = self.quant_block_0_relu1(out_rhs)
        out_rhs = self.quant_block_0_conv2(out_rhs)
        out_rhs = self.quant_block_0_relu2(out_rhs)
        out_rhs = self.quant_block_0_conv3(out_rhs)
        out_rhs = self.quant_id_1(out_rhs)
        out_lhs = self.shortcut(out_q)
        out_lhs = self.quant_id_1(out_lhs)
        out_block_0 = out_rhs + out_lhs
        out_block_0 = self.quant_block_0_relu3(out_block_0)
        # block 1
        out_rhs1 = self.quant_block_1_conv1(out_block_0)
        out_rhs1 = self.quant_block_1_relu1(out_rhs1)
        out_rhs1 = self.quant_block_1_conv2(out_rhs1)
        out_rhs1 = self.quant_block_1_relu2(out_rhs1)
        out_rhs1 = self.quant_block_1_conv3(out_rhs1)
        out_rhs1 = self.quant_add_1(out_rhs1)
        out_block_1 = out_block_0 + out_rhs1
        # out_block_1=out_block_0
        out_block_1 = self.quant_block_1_relu3(out_block_1)

        # block 1
        out_rhs2 = self.quant_block_2_conv1(out_block_1)
        out_rhs2 = self.quant_block_2_relu1(out_rhs2)
        out_rhs2 = self.quant_block_2_conv2(out_rhs2)
        out_rhs2 = self.quant_block_2_relu2(out_rhs2)
        out_rhs2 = self.quant_block_2_conv3(out_rhs2)
        out_rhs2 = self.quant_add_2(out_rhs2)
        out_block_2 = out_block_1 + out_rhs2
        # out_block_1=out_block_0
        out_block_2 = self.quant_block_2_relu3(out_block_2)

        return out_block_2


quant_model = QuantBottleneck_projected()
quant_model.eval()

# ------------------------------------------------------
# Converted input to int
# ------------------------------------------------------
q_inp = quant_model.quant_id_1(input)
int_inp = q_inp.int(float_datatype=True)

# ------------------------------------------------------
# Extract scales
# ------------------------------------------------------
init_scale = quant_model.quant_id_1.quant_act_scale()
block_0_relu_1 = quant_model.quant_block_0_relu1.quant_act_scale()
block_0_relu_2 = quant_model.quant_block_0_relu2.quant_act_scale()
block_0_relu_3 = quant_model.quant_block_0_relu3.quant_act_scale()

block_0_weight_scale1 = quant_model.quant_block_0_conv1.quant_weight_scale()
block_0_weight_scale2 = quant_model.quant_block_0_conv2.quant_weight_scale()
block_0_weight_scale3 = quant_model.quant_block_0_conv3.quant_weight_scale()
block_0_weight_scale_skip = quant_model.shortcut.quant_weight_scale()

# Block 2

block_1_relu_1 = quant_model.quant_block_1_relu1.quant_act_scale()
block_1_relu_2 = quant_model.quant_block_1_relu2.quant_act_scale()
block_1_relu_3 = quant_model.quant_block_1_relu3.quant_act_scale()

block_1_weight_scale1 = quant_model.quant_block_1_conv1.quant_weight_scale()
block_1_weight_scale2 = quant_model.quant_block_1_conv2.quant_weight_scale()
block_1_weight_scale3 = quant_model.quant_block_1_conv3.quant_weight_scale()
block_1_quant_add_1 = quant_model.quant_add_1.quant_act_scale()

# Block 3
block_2_relu_1 = quant_model.quant_block_2_relu1.quant_act_scale()
block_2_relu_2 = quant_model.quant_block_2_relu2.quant_act_scale()
block_2_relu_3 = quant_model.quant_block_2_relu3.quant_act_scale()

block_2_weight_scale1 = quant_model.quant_block_2_conv1.quant_weight_scale()
block_2_weight_scale2 = quant_model.quant_block_2_conv2.quant_weight_scale()
block_2_weight_scale3 = quant_model.quant_block_2_conv3.quant_weight_scale()
block_2_quant_add_1 = quant_model.quant_add_2.quant_act_scale()

block_0_combined_scale1 = -torch.log2(
    init_scale * block_0_weight_scale1 / block_0_relu_1
)  # RHS after first conv1x1 | clip 0-->255
block_0_combined_scale2 = -torch.log2(
    block_0_relu_1 * block_0_weight_scale2 / block_0_relu_2
)  # RHS after second conv3x3 | clip 0-->255
block_0_combined_scale3 = -torch.log2(
    block_0_relu_2 * block_0_weight_scale3 / init_scale
)  # RHS after third conv1x1 | clip -128-->+127
block_0_combined_scale_skip = -torch.log2(
    init_scale * block_0_weight_scale_skip / init_scale
)  # LHS after conv1x1 | clip -128-->+127
block_0_combined_scale4 = -torch.log2(
    init_scale / block_0_relu_3
)  # After addition | clip 0-->255

block_1_combined_scale1 = -torch.log2(
    block_0_relu_3 * block_1_weight_scale1 / block_1_relu_1
)  # RHS after first conv1x1 | clip 0-->255
block_1_combined_scale2 = -torch.log2(
    block_1_relu_1 * block_1_weight_scale2 / block_1_relu_2
)  # RHS after second conv3x3 | clip 0-->255
block_1_combined_scale3 = -torch.log2(
    block_1_relu_2 * block_1_weight_scale3 / block_1_quant_add_1
)  # RHS after third conv1x1 | clip -128-->+127
block_1_combined_scale4 = -torch.log2(
    block_1_quant_add_1 / block_1_relu_3
)  # After addition | clip 0-->255

block_2_combined_scale1 = -torch.log2(
    block_1_relu_3 * block_2_weight_scale1 / block_2_relu_1
)  # RHS after first conv1x1 | clip 0-->255
block_2_combined_scale2 = -torch.log2(
    block_2_relu_1 * block_2_weight_scale2 / block_2_relu_2
)  # RHS after second conv3x3 | clip 0-->255
block_2_combined_scale3 = -torch.log2(
    block_2_relu_2 * block_2_weight_scale3 / block_2_quant_add_1
)  # RHS after third conv1x1 | clip -128-->+127
block_2_combined_scale4 = -torch.log2(
    block_2_quant_add_1 / block_2_relu_3
)  # After addition | clip 0-->255

print("combined_scale block_0 after first conv1x1:", block_0_combined_scale1.item())
print("combined_scale block_0 after second conv3x3:", block_0_combined_scale2.item())
print("combined_scale block_0 after third conv1x1:", block_0_combined_scale3.item())
print(
    "combined_scale block_0 after adding skip connection:",
    (block_0_combined_scale4).item(),
)
print("combined_scale block_0 after skip conv1x1:", block_0_combined_scale_skip.item())
print("--------------------------------------------------------------")
print("combined_scale block_1 after first conv1x1:", block_1_combined_scale1.item())
print("combined_scale block_1 after second conv3x3:", block_1_combined_scale2.item())
print("combined_scale block_1 after third conv1x1:", block_1_combined_scale3.item())
print(
    "combined_scale block_1 after adding skip connection:",
    (block_1_combined_scale4).item(),
)
print("--------------------------------------------------------------")
print("combined_scale block_2 after first conv1x1:", block_2_combined_scale1.item())
print("combined_scale block_2 after second conv3x3:", block_2_combined_scale2.item())
print("combined_scale block_2 after third conv1x1:", block_2_combined_scale3.item())
print(
    "combined_scale block_2 after adding skip connection:",
    (block_2_combined_scale4).item(),
)

# ------------------------------------------------------
# Extract conv wts
# ------------------------------------------------------
block_0_int_weight_1 = quant_model.quant_block_0_conv1.quant_weight().int(
    float_datatype=True
)
block_0_int_weight_2 = quant_model.quant_block_0_conv2.quant_weight().int(
    float_datatype=True
)
block_0_int_weight_3 = quant_model.quant_block_0_conv3.quant_weight().int(
    float_datatype=True
)
block_0_int_weight_skip = quant_model.shortcut.quant_weight().int(float_datatype=True)

block_1_int_weight_1 = quant_model.quant_block_1_conv1.quant_weight().int(
    float_datatype=True
)
block_1_int_weight_2 = quant_model.quant_block_1_conv2.quant_weight().int(
    float_datatype=True
)
block_1_int_weight_3 = quant_model.quant_block_1_conv3.quant_weight().int(
    float_datatype=True
)

block_2_int_weight_1 = quant_model.quant_block_2_conv1.quant_weight().int(
    float_datatype=True
)
block_2_int_weight_2 = quant_model.quant_block_2_conv2.quant_weight().int(
    float_datatype=True
)
block_2_int_weight_3 = quant_model.quant_block_2_conv3.quant_weight().int(
    float_datatype=True
)

# ------------------------------------------------------
# Run Brevitas inference
# ------------------------------------------------------
quant_model_out = quant_model(input)
gold_out = (
    quant_model_out.int(float_datatype=True).data.numpy().astype(np.dtype("uint8"))
)
print("Brevitas::", quant_model_out)
# ------------------------------------------------------
# Pytorch layer
# ------------------------------------------------------

# ------------------------------------------------------
# Define your golden reference
# ------------------------------------------------------
min = 0
max = 255


class resnet_conv2_x_int8(nn.Module):
    expansion = 4

    def __init__(self, in_planes=64, planes=64):
        super(resnet_conv2_x_int8, self).__init__()
        self.shortcut = nn.Conv2d(
            in_planes, self.expansion * planes, kernel_size=1, bias=False
        )
        # Bottleneck 0
        self.block_0_conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.block_0_conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros", bias=False
        )
        self.block_0_conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        self.block_0_relu1 = nn.ReLU()
        self.block_0_relu2 = nn.ReLU()
        self.block_0_relu3 = nn.ReLU()
        
        # Bottleneck 1
        self.block_1_conv1 = nn.Conv2d(self.expansion * planes, planes, kernel_size=1, bias=False)
        self.block_1_conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros", bias=False
        )
        self.block_1_conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        self.block_1_relu1 = nn.ReLU()
        self.block_1_relu2 = nn.ReLU()
        self.block_1_relu3 = nn.ReLU()

        # Bottleneck 2
        self.block_2_conv1 = nn.Conv2d(self.expansion * planes, planes, kernel_size=1, bias=False)
        self.block_2_conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros", bias=False
        )
        self.block_2_conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        self.block_2_relu1 = nn.ReLU()
        self.block_2_relu2 = nn.ReLU()
        self.block_2_relu3 = nn.ReLU()

    def forward(self, x):
        # **************** Bottleneck 0 ****************
        block_0_conv1_out = self.block_0_conv1(x) * init_scale * block_0_weight_scale1
        block_0_relu1_out = torch.clamp(
            torch.round(self.block_0_relu1(block_0_conv1_out) / block_0_relu_1), min, max
        )  # convert to int and apply relu
        block_0_conv2_out = self.block_0_conv2(block_0_relu1_out) * block_0_relu_1 * block_0_weight_scale2
        block_0_relu2_out = torch.clamp(
            torch.round(self.block_0_relu2(block_0_conv2_out) / block_0_relu_2), min, max
        )
        block_0_conv3_out = self.block_0_conv3(block_0_relu2_out) * block_0_relu_2 * block_0_weight_scale3
        block_0_rhf_same_scale = torch.clamp(torch.round(block_0_conv3_out / init_scale), -128, 127)


        block_0_lhs_conv = self.shortcut(x) * init_scale * block_0_weight_scale_skip
        block_0_lhs_same_scale = torch.clamp(
            torch.round(block_0_lhs_conv / init_scale),  -128, 127)
          # convert to int and apply relu

        block_0_skip_add = init_scale * (block_0_rhf_same_scale + block_0_lhs_same_scale)
        block_0_final_out =  (
            torch.clamp(torch.round(self.block_0_relu3(block_0_skip_add) / block_0_relu_3), min, max)
        )
        # **************** Bottleneck 1 ****************
        block_1_conv1_out = self.block_1_conv1(block_0_final_out) * block_0_relu_3 * block_1_weight_scale1
        block_1_relu1_out = torch.clamp(
            torch.round(self.block_1_relu1(block_1_conv1_out) / block_1_relu_1), min, max
        )  # convert to int and apply relu
        block_1_conv2_out = self.block_1_conv2(block_1_relu1_out) * block_1_relu_1 * block_1_weight_scale2
        block_1_relu2_out = torch.clamp(
            torch.round(self.block_1_relu2(block_1_conv2_out) / block_1_relu_2), min, max
        )
        block_1_conv3_out = self.block_1_conv3(block_1_relu2_out) * block_1_relu_2 * block_1_weight_scale3
        block_1_rhf_same_scale = torch.clamp(torch.round(block_1_conv3_out / block_0_relu_3), -128, 127)

        block_1_skip_add = block_0_relu_3 * (block_1_rhf_same_scale + block_0_final_out)
        block_1_final_out = (
            torch.clamp(torch.round(self.block_1_relu3(block_1_skip_add) / block_1_relu_3), min, max)
        )

        # **************** Bottleneck 2 ****************
        block_2_conv1_out = self.block_2_conv1(block_1_final_out) * block_1_relu_3 * block_2_weight_scale1
        block_2_relu1_out = torch.clamp(
            torch.round(self.block_2_relu1(block_2_conv1_out) / block_2_relu_1),
            min,
            max,
        )  # convert to int and apply relu
        block_2_conv2_out = self.block_2_conv2(block_2_relu1_out) * block_2_relu_1 * block_2_weight_scale2
        block_2_relu2_out = torch.clamp(
            torch.round(self.block_2_relu2(block_2_conv2_out) / block_2_relu_2), min, max
        )
        block_2_conv3_out = self.block_2_conv3(block_2_relu2_out) * block_2_relu_2 * block_2_weight_scale3
        block_2_rhf_same_scale = torch.clamp(torch.round(block_2_conv3_out / block_1_relu_3), -128, 127)

        block_2_skip_add = block_1_relu_3 * (block_2_rhf_same_scale + block_1_final_out)
        block_2_final_out = block_2_relu_3 * (
            torch.clamp(torch.round(self.block_2_relu3(block_2_skip_add) / block_2_relu_3), min, max)
        )
        return block_2_final_out

py_model=resnet_conv2_x_int8()

py_model.block_0_conv1.weight.data.copy_(block_0_int_weight_1)
py_model.block_0_conv2.weight.data.copy_(block_0_int_weight_2)
py_model.block_0_conv3.weight.data.copy_(block_0_int_weight_3)
py_model.shortcut.weight.data.copy_(block_0_int_weight_skip)

py_model.block_1_conv1.weight.data.copy_(block_1_int_weight_1)
py_model.block_1_conv2.weight.data.copy_(block_1_int_weight_2)
py_model.block_1_conv3.weight.data.copy_(block_1_int_weight_3)

py_model.block_2_conv1.weight.data.copy_(block_2_int_weight_1)
py_model.block_2_conv2.weight.data.copy_(block_2_int_weight_2)
py_model.block_2_conv3.weight.data.copy_(block_2_int_weight_3)


final_out=py_model(int_inp)

print("Pytorch::", final_out)

print("difference::", torch.max(torch.abs(final_out - quant_model_out)))
print("--------------------------------------------------------------")
print("combined_scale block_0 after first conv1x1:", block_0_combined_scale1.item())
print("combined_scale block_0 after second conv3x3:", block_0_combined_scale2.item())
print("combined_scale block_0 after third conv1x1:", block_0_combined_scale3.item())
print(
    "combined_scale block_0 after adding skip connection:",
    (block_0_combined_scale4).item(),
)
print(
    "combined_scale block_0 after skip conv1x1:", block_0_combined_scale_skip.item()
)
print("--------------------------------------------------------------")
print("combined_scale block_1 after first conv1x1:", block_1_combined_scale1.item())
print("combined_scale block_1 after second conv3x3:", block_1_combined_scale2.item())
print("combined_scale block_1 after third conv1x1:", block_1_combined_scale3.item())
print(
    "combined_scale block_1 after adding skip connection:",
    (block_1_combined_scale4).item(),
)
print("--------------------------------------------------------------")
print("combined_scale block_2 after first conv1x1:", block_2_combined_scale1.item())
print("combined_scale block_2 after second conv3x3:", block_2_combined_scale2.item())
print("combined_scale block_2 after third conv1x1:", block_2_combined_scale3.item())
print(
    "combined_scale block_2 after adding skip connection:",
    (block_2_combined_scale4).item(),
)