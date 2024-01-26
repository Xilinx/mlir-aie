import torch
import torch.nn as nn
import sys
import os
import math
import numpy as np
from brevitas.nn import QuantConv2d, QuantIdentity, QuantReLU
from brevitas.quant.fixed_point import (
    Int8ActPerTensorFixedPoint,
    Int8WeightPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)

torch.manual_seed(0)

input = torch.randn(1, 64, 32, 32)
num_classes = 10


class QuantBottleneck_projected(nn.Module):
    expansion = 4

    def __init__(self, in_planes=64, planes=64):
        super(QuantBottleneck_projected, self).__init__()
        # block 0
        self.quant_id_1 = QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.quant_block0_conv1 = QuantConv2d(
            in_planes,
            planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        self.quant_block0_conv2 = QuantConv2d(
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
        self.quant_block0_conv3 = QuantConv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        self.quant_block0_relu1 = QuantReLU(
            act_quant=Uint8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.quant_block0_relu2 = QuantReLU(
            act_quant=Uint8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.quant_block0_relu3 = QuantReLU(
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
        self.quant_block1_conv1 = QuantConv2d(
            self.expansion * in_planes,
            planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        self.quant_block1_conv2 = QuantConv2d(
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
        self.quant_block1_conv3 = QuantConv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        self.quant_block1_relu1 = QuantReLU(
            act_quant=Uint8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.quant_block1_relu2 = QuantReLU(
            act_quant=Uint8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.quant_block1_relu3 = QuantReLU(
            act_quant=Uint8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )

        self.quant_add_1 = QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        # Quant_add_1 shares the scale factors with block0_relu3, however one is signed and the other one is unsigned
        self.quant_add_1.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl = (
            self.quant_block0_relu3.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl
        )
        self.quant_add_1.act_quant.fused_activation_quant_proxy.tensor_quant.int_scaling_impl = (
            self.quant_block0_relu3.act_quant.fused_activation_quant_proxy.tensor_quant.int_scaling_impl
        )

        # block 2
        self.quant_block2_conv1 = QuantConv2d(
            self.expansion * in_planes,
            planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        self.quant_block2_conv2 = QuantConv2d(
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
        self.quant_block2_conv3 = QuantConv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            bit_width=8,
            weight_bit_width=8,
            bias=False,
            weight_quant=Int8WeightPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        self.quant_block2_relu1 = QuantReLU(
            act_quant=Uint8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.quant_block2_relu2 = QuantReLU(
            act_quant=Uint8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        self.quant_block2_relu3 = QuantReLU(
            act_quant=Uint8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )

        self.quant_add_2 = QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
        )
        # Quant_add_1 shares the scale factors with block0_relu3, however one is signed and the other one is unsigned
        self.quant_add_2.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl = (
            self.quant_block1_relu3.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl
        )
        self.quant_add_2.act_quant.fused_activation_quant_proxy.tensor_quant.int_scaling_impl = (
            self.quant_block1_relu3.act_quant.fused_activation_quant_proxy.tensor_quant.int_scaling_impl
        )

    def forward(self, x):
        out_q = self.quant_id_1(x)
        out_rhs = self.quant_block0_conv1(out_q)
        out_rhs = self.quant_block0_relu1(out_rhs)
        out_rhs = self.quant_block0_conv2(out_rhs)
        out_rhs = self.quant_block0_relu2(out_rhs)
        out_rhs = self.quant_block0_conv3(out_rhs)
        out_rhs = self.quant_id_1(out_rhs)
        out_lhs = self.shortcut(out_q)
        out_lhs = self.quant_id_1(out_lhs)
        out_block0 = out_rhs + out_lhs
        out_block0 = self.quant_block0_relu3(out_block0)
        # block 1
        out_rhs1 = self.quant_block1_conv1(out_block0)
        out_rhs1 = self.quant_block1_relu1(out_rhs1)
        out_rhs1 = self.quant_block1_conv2(out_rhs1)
        out_rhs1 = self.quant_block1_relu2(out_rhs1)
        out_rhs1 = self.quant_block1_conv3(out_rhs1)
        out_rhs1 = self.quant_add_1(out_rhs1)
        out_block1 = out_block0 + out_rhs1
        # out_block1=out_block0
        out_block1 = self.quant_block1_relu3(out_block1)

        # block 1
        out_rhs2 = self.quant_block2_conv1(out_block1)
        out_rhs2 = self.quant_block2_relu1(out_rhs2)
        out_rhs2 = self.quant_block2_conv2(out_rhs2)
        out_rhs2 = self.quant_block2_relu2(out_rhs2)
        out_rhs2 = self.quant_block2_conv3(out_rhs2)
        out_rhs2 = self.quant_add_2(out_rhs2)
        out_block2 = out_block1 + out_rhs2
        # out_block1=out_block0
        out_block2 = self.quant_block2_relu3(out_block2)

        return out_block2


quant_bottleneck_model = QuantBottleneck_projected()

quant_id_1 = QuantIdentity(
    act_quant=Int8ActPerTensorFixedPoint, bit_width=8, return_quant_tensor=True
)
quant_bottleneck_model.eval()
quant_id_1.eval()

init_scale = quant_bottleneck_model.quant_id_1.quant_act_scale()
block_0_relu_1 = quant_bottleneck_model.quant_block0_relu1.quant_act_scale()
block_0_relu_2 = quant_bottleneck_model.quant_block0_relu2.quant_act_scale()
block_0_relu_3 = quant_bottleneck_model.quant_block0_relu3.quant_act_scale()

block_0_weight_scale1 = quant_bottleneck_model.quant_block0_conv1.quant_weight_scale()
block_0_weight_scale2 = quant_bottleneck_model.quant_block0_conv2.quant_weight_scale()
block_0_weight_scale3 = quant_bottleneck_model.quant_block0_conv3.quant_weight_scale()
block_0_weight_scale_skip = quant_bottleneck_model.shortcut.quant_weight_scale()

# Block 2

block_1_relu_1 = quant_bottleneck_model.quant_block1_relu1.quant_act_scale()
block_1_relu_2 = quant_bottleneck_model.quant_block1_relu2.quant_act_scale()
block_1_relu_3 = quant_bottleneck_model.quant_block1_relu3.quant_act_scale()

block_1_weight_scale1 = quant_bottleneck_model.quant_block1_conv1.quant_weight_scale()
block_1_weight_scale2 = quant_bottleneck_model.quant_block1_conv2.quant_weight_scale()
block_1_weight_scale3 = quant_bottleneck_model.quant_block1_conv3.quant_weight_scale()
block_1_quant_add_1 = quant_bottleneck_model.quant_add_1.quant_act_scale()

# Block 3
block_2_relu_1 = quant_bottleneck_model.quant_block2_relu1.quant_act_scale()
block_2_relu_2 = quant_bottleneck_model.quant_block2_relu2.quant_act_scale()
block_2_relu_3 = quant_bottleneck_model.quant_block2_relu3.quant_act_scale()

block_2_weight_scale1 = quant_bottleneck_model.quant_block2_conv1.quant_weight_scale()
block_2_weight_scale2 = quant_bottleneck_model.quant_block2_conv2.quant_weight_scale()
block_2_weight_scale3 = quant_bottleneck_model.quant_block2_conv3.quant_weight_scale()
block_2_quant_add_1 = quant_bottleneck_model.quant_add_2.quant_act_scale()

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

print("combined_scale block0 after first conv1x1:", block_0_combined_scale1.item())
print("combined_scale block0 after second conv3x3:", block_0_combined_scale2.item())
print("combined_scale block0 after third conv1x1:", block_0_combined_scale3.item())
print(
    "combined_scale block0 after adding skip connection:",
    (block_0_combined_scale4).item(),
)
print("combined_scale block0 after skip conv1x1:", block_0_combined_scale_skip.item())
print("--------------------------------------------------------------")
print("combined_scale block1 after first conv1x1:", block_1_combined_scale1.item())
print("combined_scale block1 after second conv3x3:", block_1_combined_scale2.item())
print("combined_scale block1 after third conv1x1:", block_1_combined_scale3.item())
print(
    "combined_scale block1 after adding skip connection:",
    (block_1_combined_scale4).item(),
)
print("--------------------------------------------------------------")
print("combined_scale block2 after first conv1x1:", block_2_combined_scale1.item())
print("combined_scale block2 after second conv3x3:", block_2_combined_scale2.item())
print("combined_scale block2 after third conv1x1:", block_2_combined_scale3.item())
print(
    "combined_scale block2 after adding skip connection:",
    (block_2_combined_scale4).item(),
)

q_bottleneck_out = quant_bottleneck_model(input)
dtype_out = np.dtype("uint8")
# gold_out=q_bottleneck_out.int(float_datatype=True).data.numpy().astype(dtype_out)
# print("Golden::Brevitas::",gold_out)
# print(block_1_relu_3)
from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate

calibrate([(torch.rand(32, 64, 32, 32), 1) for _ in range(5)], quant_bottleneck_model)
#
from brevitas.fx import brevitas_symbolic_trace

model = brevitas_symbolic_trace(quant_bottleneck_model)
print(model.graph)
