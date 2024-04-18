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
class quant_conv2d_relu(nn.Module):
    def __init__(self, in_planes=64, planes=64):
        super(quant_conv2d_relu, self).__init__()
        self.quant_id_1 = QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint,
            bit_width=8,
            return_quant_tensor=True,
        )
        self.quant_conv1 = QuantConv2d(
            in_planes,
            planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        self.quant_id_2 = QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint,
            bit_width=8,
            return_quant_tensor=True,
        )

    def forward(self, x):
        out_q = self.quant_id_1(x)
        out = self.quant_conv1(out_q)
        out = self.quant_id_2(out)
        return out


quant_model = quant_conv2d_relu()
quant_model.eval()

# ------------------------------------------------------
# Converted input to int
# ------------------------------------------------------
input = torch.randint(1, 100, (1, 64, 32, 32)).type(torch.FloatTensor)
q_inp = quant_model.quant_id_1(input)
int_inp = q_inp.int(float_datatype=True)

# ------------------------------------------------------
# Extract scales
# ------------------------------------------------------
init_scale = quant_model.quant_id_1.quant_act_scale()
conv_wts_scale = quant_model.quant_conv1.quant_weight_scale()
init_scale_2 = quant_model.quant_id_2.quant_act_scale()

import math

scale = -torch.log2(init_scale * conv_wts_scale / init_scale_2)
print("combined_scale after first conv1x1:", scale.item())
# ------------------------------------------------------
# Extract conv wts
# ------------------------------------------------------
int_weight = quant_model.quant_conv1.quant_weight().int(float_datatype=True)

# ------------------------------------------------------
# Run Brevitas inference
# ------------------------------------------------------
quant_model_out = quant_model(input)
quant_model_out_int = (
    quant_model_out.int(float_datatype=True).data.numpy().astype(np.dtype("int8"))
)
quant_model_out_int = quant_model_out / (init_scale_2)
# ------------------------------------------------------
# Pytorch layer
# ------------------------------------------------------
conv1 = nn.Conv2d(64, 64, kernel_size=1, bias=False)

conv1.eval()

conv1.weight.data.copy_(int_weight)
min = -128
max = 127

conv1_out_int = conv1(int_inp)
conv1_out_quant = conv1(int_inp) * init_scale * conv_wts_scale
final_output_quant = init_scale_2 * torch.clamp(
    torch.round(conv1_out_quant / init_scale_2), min, max
)
final_output_int = final_output_quant / init_scale_2

print("Brevitas Quant::", quant_model_out)
print("Pytorch Quant::", final_output_quant)
print("Difference::", torch.max(torch.abs(final_output_quant - quant_model_out)))

print("Brevitas INT::", quant_model_out_int)
print("Pytorch INT::", final_output_int)
print("Difference INT::", torch.max(torch.abs(final_output_int - quant_model_out_int)))

print("multiply after first conv1x1:", init_scale * conv_wts_scale)
print("init_scale_2 after first conv1x1:", init_scale_2)
print("combined_scale after first conv1x1:", scale)
# scale=-math.log2(0.25)
# print("combined_scale after first conv1x1:",scale)
