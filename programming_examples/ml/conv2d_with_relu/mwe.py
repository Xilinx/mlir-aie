import torch
import torch.nn as nn
import sys
sys.path.append("../../utils")
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
        self.quant_relu1 = QuantReLU(
            act_quant=Uint8ActPerTensorFixedPoint,
            bit_width=8,
            return_quant_tensor=True,
        )

    def forward(self, x):
        out_q = self.quant_id_1(x)
        out = self.quant_conv1(out_q)
        out = self.quant_relu1(out)
        return out

quant_model = quant_conv2d_relu()
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
conv_wts_scale = quant_model.quant_conv1.quant_weight_scale()
relu_scale = quant_model.quant_relu1.quant_act_scale()
scale = -torch.log2(init_scale * conv_wts_scale / relu_scale)
print("Scale factor after first conv1x1:", scale.item())
# ------------------------------------------------------
# Extract conv wts
# ------------------------------------------------------
int_weight = quant_model.quant_conv1.quant_weight().int(float_datatype=True)

# ------------------------------------------------------
# Run Brevitas inference
# ------------------------------------------------------
quant_model_out = quant_model(input)
gold_out = quant_model_out.int(float_datatype=True).data.numpy().astype(np.dtype("uint8"))
print("Brevitas::", quant_model_out)
# ------------------------------------------------------
# Pytorch layer
# ------------------------------------------------------
conv1= nn.Conv2d(64, 64, kernel_size=1, bias=False)
relu= nn.ReLU()
conv1.eval()
min=0
max=255
conv1.weight.data.copy_(int_weight)

conv1_out=conv1(int_inp) * init_scale * conv_wts_scale
relu_output=relu(conv1_out)
final_output=relu_scale * torch.clamp(torch.round(relu_output/relu_scale), min, max)

print("Pytorch::", final_output)

print("difference::",torch.max(torch.abs(final_output - quant_model_out)))