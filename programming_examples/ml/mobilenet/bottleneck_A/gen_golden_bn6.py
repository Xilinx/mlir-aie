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
import aie.iron as iron
from aie.utils import DefaultNPURuntime
from aie.utils import TraceConfig, HostRuntime, NPUKernel, DefaultNPURuntime
import aie.utils.test as test_utils
from brevitas.nn import QuantConv2d, QuantIdentity, QuantReLU
from brevitas.quant.fixed_point import (
    Int8ActPerTensorFixedPoint,
    Int8WeightPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
vectorSize = 8

sys.path.append("..")
import mb_utils

log_dir = "log/"
data_dir = "data/"

# bn6
tensorInW = 28
tensorInH = 28
tensorInC = 40
tensorOutC = 80

depthwiseStride = 2
depthWiseChannels = 240

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

bneck_1_InC1_vec = math.floor(bneck_1_InC1 / vectorSize)
bneck_1_OutC3_vec = math.floor(bneck_1_OutC3 / vectorSize)


def main():
    print("Running torch reference model for bottleneck_A_subblocks for bn6 ...")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # ------------------------------------------------------
    # Configure this to match your design's buffer size
    # ------------------------------------------------------
    dtype_in = np.dtype("int8")
    dtype_wts = np.dtype("int8")
    dtype_out = np.dtype("int8")

    shape_total_wts = (
        bneck_1_InC1 * bneck_1_OutC1
        + 3 * 3 * bneck_1_OutC2
        + bneck_1_OutC2 * bneck_1_OutC3,
        1,
    )
    shape_in_act = (
        bneck_1_InH1,
        bneck_1_InC1_vec,
        bneck_1_InW1,
        vectorSize,
    )  #'YCXC8' , 'CYX'
    shape_out = (bneck_1_InH3, bneck_1_OutC3_vec, bneck_1_InW3, vectorSize)  # HCWC8
    shape_out_final = (
        bneck_1_OutC3_vec * vectorSize,
        bneck_1_InH3,
        bneck_1_InW3,
    )  # CHW

    # ------------------------------------------------------
    # Initialize activation, weights, scaling factor for int8 model
    # ------------------------------------------------------
    input = torch.randn(1, bneck_1_InC1_vec * vectorSize, bneck_1_InH1, bneck_1_InW1)

    class QuantBottleneckA(nn.Module):
        def __init__(self, in_planes=16, bn1_expand=16, bn1_project=16):
            super(QuantBottleneckA, self).__init__()
            self.quant_id_1 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

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
            out = self.quant_id_1(x)
            out = self.bn1_quant_conv1(out)
            out = self.bn1_quant_relu1(out)
            out = self.bn1_quant_conv2(out)
            out = self.bn1_quant_relu2(out)
            out = self.bn1_quant_conv3(out)
            out = self.bn1_quant_id_2(out)
            return out

    quant_bottleneck_model = QuantBottleneckA(
        in_planes=bneck_1_InC1, bn1_expand=bneck_1_OutC1, bn1_project=bneck_1_OutC3
    )
    quant_bottleneck_model.eval()

    q_bottleneck_out = quant_bottleneck_model(input)
    golden_output = (
        q_bottleneck_out.int(float_datatype=True).data.numpy().astype(dtype_out)
    )
    print("Golden::Brevitas::", golden_output)
    q_inp = quant_bottleneck_model.quant_id_1(input)
    int_inp = q_inp.int(float_datatype=True)

    init_scale = quant_bottleneck_model.quant_id_1.act_quant.scale()
    block_1_relu_1 = quant_bottleneck_model.bn1_quant_relu1.act_quant.scale()
    block_1_relu_2 = quant_bottleneck_model.bn1_quant_relu2.act_quant.scale()
    block_1_final_scale = quant_bottleneck_model.bn1_quant_id_2.act_quant.scale()

    block_1_weight_scale1 = quant_bottleneck_model.bn1_quant_conv1.weight_quant.scale()
    block_1_weight_scale2 = quant_bottleneck_model.bn1_quant_conv2.weight_quant.scale()
    block_1_weight_scale3 = quant_bottleneck_model.bn1_quant_conv3.weight_quant.scale()
    block_1_combined_scale1 = -torch.log2(
        init_scale * block_1_weight_scale1 / block_1_relu_1
    )
    block_1_combined_scale2 = -torch.log2(
        block_1_relu_1 * block_1_weight_scale2 / block_1_relu_2
    )
    block_1_combined_scale3 = -torch.log2(
        block_1_relu_2 * block_1_weight_scale3 / block_1_final_scale
    )

    print("********************BN1*******************************")
    print("combined_scale after conv1x1:", block_1_combined_scale1.item())
    print("combined_scale after conv3x3:", block_1_combined_scale2.item())
    print("combined_scale after conv1x1:", block_1_combined_scale3.item())
    print("********************BN1*******************************")

    # print("combined_scale after conv1x1:", ( block_0_relu_2 * block_0_weight_scale3).item())

    # ------------------------------------------------------
    # Reorder input data-layout
    # ------------------------------------------------------

    block_1_int_weight_1 = quant_bottleneck_model.bn1_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_1_int_weight_2 = quant_bottleneck_model.bn1_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_1_int_weight_3 = quant_bottleneck_model.bn1_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    print("Writing golden output txt file.")
    golden_output.tofile(log_dir + "/golden_output.txt", sep=",", format="%d")
    ds = DataShaper()
    before_input = int_inp.squeeze().data.numpy().astype(dtype_in)

    print("Writing input txt file.")
    before_input.tofile(log_dir + "before_ifm_mem_fmt_1x1.txt", sep=",", format="%d")
    ifm_mem_fmt = ds.reorder_mat(before_input, "YCXC8", "CYX")
    ifm_mem_fmt.tofile(log_dir + "after_ifm_mem_fmt.txt", sep=",", format="%d")

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

    total_wts = np.concatenate((bn1_wts1, bn1_wts2, bn1_wts3), axis=None)

    print("Writing weights txt files.")
    total_wts.tofile(log_dir + "after_weights_mem_fmt_final.txt", sep=",", format="%d")

    print("{}+{}+{}".format(bn1_wts1.shape, bn1_wts2.shape, bn1_wts3.shape))
    print(shape_total_wts)
    print(total_wts.shape)

    print("Done.")


if __name__ == "__main__":
    main()
