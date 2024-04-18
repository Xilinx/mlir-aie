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
input = torch.randn(1, 256, 32, 32)


# ------------------------------------------------------
# Define your golden reference
# ------------------------------------------------------
class QuantBottleneck_projected(nn.Module):
    expansion = 4

    def __init__(self, in_planes=256, planes=64):
        super(QuantBottleneck_projected, self).__init__()
        self.quant_id_1 = QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint,
            bit_width=8,
            return_quant_tensor=True,
        )
        self.conv1 = QuantConv2d(
            in_planes,
            planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        self.conv2 = QuantConv2d(
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
        self.conv3 = QuantConv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            bit_width=8,
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
        self.relu2 = QuantReLU(
            act_quant=Uint8ActPerTensorFixedPoint,
            bit_width=8,
            return_quant_tensor=True,
        )
        self.relu3 = QuantReLU(
            act_quant=Uint8ActPerTensorFixedPoint,
            bit_width=8,
            return_quant_tensor=True,
        )

    def forward(self, x):
        out_q = self.quant_id_1(x)
        out = self.conv1(out_q)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.quant_id_1(out)
        out = out + out_q
        out = self.relu3(out)
        return out


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
inp_scale1 = quant_model.quant_id_1.quant_act_scale()
inp_scale2 = quant_model.relu1.quant_act_scale()
inp_scale3 = quant_model.relu2.quant_act_scale()
inp_scale4 = quant_model.relu3.quant_act_scale()

weight_scale1 = quant_model.conv1.quant_weight_scale()
weight_scale2 = quant_model.conv2.quant_weight_scale()
weight_scale3 = quant_model.conv3.quant_weight_scale()

combined_scale1 = -torch.log2(inp_scale1 * weight_scale1 / inp_scale2)
combined_scale2 = -torch.log2(inp_scale2 * weight_scale2 / inp_scale3)
combined_scale3 = -torch.log2(inp_scale3 * weight_scale3 / inp_scale1)
combined_scale4 = -torch.log2(inp_scale1 / inp_scale4)

# ------------------------------------------------------
# Extract conv wts
# ------------------------------------------------------
int_weight1 = quant_model.conv1.quant_weight().int(float_datatype=True)
int_weight2 = quant_model.conv2.quant_weight().int(float_datatype=True)
int_weight3 = quant_model.conv3.quant_weight().int(float_datatype=True)

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
py_conv1 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
py_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode="zeros", bias=False)
py_conv3 = nn.Conv2d(64, 256, kernel_size=1, bias=False)

py_relu1 = nn.ReLU()
py_relu2 = nn.ReLU()
py_relu3 = nn.ReLU()

py_conv1.eval()
py_conv2.eval()
py_conv3.eval()

min = 0
max = 255
py_conv1.weight.data.copy_(int_weight1)
py_conv2.weight.data.copy_(int_weight2)
py_conv3.weight.data.copy_(int_weight3)

conv1_out = py_conv1(int_inp) * inp_scale1 * weight_scale1
relu1_out = torch.clamp(
    torch.round(py_relu1(conv1_out) / inp_scale2), min, max
)  # convert to int and apply relu
conv2_out = py_conv2(relu1_out) * inp_scale2 * weight_scale2
relu2_out = torch.clamp(torch.round(py_relu2(conv2_out) / inp_scale3), min, max)
conv3_out = py_conv3(relu2_out) * inp_scale3 * weight_scale3
same_scale_init = torch.clamp(torch.round(conv3_out / inp_scale1), -128, 127)

skip_add = inp_scale1 * (same_scale_init + int_inp)
final_out = inp_scale4 * (torch.clamp(torch.round(skip_add / inp_scale4), min, max))

print("Pytorch::", final_out)

print("difference::", torch.max(torch.abs(final_out - quant_model_out)))
print("combined_scale after first conv1x1:", combined_scale1.item())
print("combined_scale after second conv3x3:", combined_scale2.item())
print("combined_scale after third conv1x1:", combined_scale3.item())
print("combined_scale after adding skip connection:", (combined_scale4).item())
