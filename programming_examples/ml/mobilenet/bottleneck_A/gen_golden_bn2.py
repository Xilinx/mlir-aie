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

# bn2
tensorInW = 56
tensorInH = 56
tensorInC = 24
tensorOutC = tensorInC

depthwiseStride = 1
depthWiseChannels = 72

bneck_2_InW1 = tensorInW
bneck_2_InH1 = tensorInH
bneck_2_InC1 = tensorInC
bneck_2_OutC1 = depthWiseChannels

bneck_2_InW2 = bneck_2_InW1
bneck_2_InH2 = bneck_2_InH1
bneck_2_OutC2 = bneck_2_OutC1

bneck_2_InW3 = bneck_2_InW2
bneck_2_InH3 = bneck_2_InH2
bneck_2_OutC3 = bneck_2_InC1

bneck_2_InC1_vec = math.floor(bneck_2_InC1 / vectorSize)
bneck_2_OutC3_vec = math.floor(bneck_2_OutC3 / vectorSize)


def main():
    print("Running torch reference model for bottleneck_A_subblocks for bn2...")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # ------------------------------------------------------
    # Configure this to match your design's buffer size
    # ------------------------------------------------------
    dtype_in = np.dtype("int8")
    dtype_wts = np.dtype("int8")
    dtype_out = np.dtype("int8")

    shape_total_wts = (
        bneck_2_InC1 * bneck_2_OutC1
        + 3 * 3 * bneck_2_OutC2
        + bneck_2_OutC2 * bneck_2_OutC3,
        1,
    )
    shape_in_act = (
        bneck_2_InH1,
        bneck_2_InC1_vec,
        bneck_2_InW1,
        vectorSize,
    )  #'YCXC8' , 'CYX'
    shape_out = (bneck_2_InH3, bneck_2_OutC3_vec, bneck_2_InW3, vectorSize)  # HCWC8
    shape_out_final = (
        bneck_2_OutC3_vec * vectorSize,
        bneck_2_InH3,
        bneck_2_InW3,
    )  # CHW

    # ------------------------------------------------------
    # Initialize activation, weights, scaling factor for int8 model
    # ------------------------------------------------------
    input = torch.randn(1, bneck_2_InC1_vec * vectorSize, bneck_2_InH1, bneck_2_InW1)

    class QuantBottleneckA(nn.Module):
        def __init__(self, in_planes=16, bn2_expand=16, bn2_project=16):
            super(QuantBottleneckA, self).__init__()
            self.quant_id_1 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

            self.bn2_quant_conv1 = QuantConv2d(
                in_planes,
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
                stride=depthwiseStride,
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

        def forward(self, x):
            out_q = self.quant_id_1(x)
            out = self.bn2_quant_conv1(out_q)
            out = self.bn2_quant_relu1(out)
            out = self.bn2_quant_conv2(out)
            out = self.bn2_quant_relu2(out)
            out = self.bn2_quant_conv3(out)
            out = self.quant_id_1(out)
            out = out + out_q
            out = self.bn2_add(out)
            return out

    quant_bottleneck_model = QuantBottleneckA(
        in_planes=bneck_2_InC1, bn2_expand=bneck_2_OutC1, bn2_project=bneck_2_OutC3
    )
    sys.path.append("..")
    from mb_utils import ExpandChannels
    from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
    import torchvision
    import torch.utils.data as data_utils
    from torchvision import transforms

    # Define the image preprocessing pipeline
    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(56),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ExpandChannels(target_channels=24),  # Expand to 80 channels
        ]
    )

    # test_dataset = torchvision.datasets.ImageNet(
    #     root=data_dir, train=False, transform=transform, download=True)

    # # Create a subset and DataLoader for the single image
    # indices = torch.arange(32)
    # val_sub = data_utils.Subset(test_dataset, indices)
    # calib_loader = torch.utils.data.DataLoader(dataset=val_sub, batch_size=32, shuffle=False)

    src_data = "/group/xrlabs2/imagenet/calibration"
    datset = torchvision.datasets.ImageFolder(src_data, transform)
    indices = torch.arange(4)
    val_sub = data_utils.Subset(datset, indices)
    calib_loader = torch.utils.data.DataLoader(
        dataset=val_sub, batch_size=32, shuffle=False
    )
    calibrate(calib_loader, quant_bottleneck_model)

    quant_bottleneck_model.eval()

    q_bottleneck_out = quant_bottleneck_model(input)
    golden_output = (
        q_bottleneck_out.int(float_datatype=True).data.numpy().astype(dtype_out)
    )
    print("Golden::Brevitas::", golden_output)
    q_inp = quant_bottleneck_model.quant_id_1(input)
    int_inp = q_inp.int(float_datatype=True)

    block_2_inp_scale1 = quant_bottleneck_model.quant_id_1.act_quant.scale()

    block_2_relu_1 = quant_bottleneck_model.bn2_quant_relu1.act_quant.scale()
    block_2_relu_2 = quant_bottleneck_model.bn2_quant_relu2.act_quant.scale()
    block_2_skip_add = quant_bottleneck_model.bn2_add.act_quant.scale()

    block_2_weight_scale1 = quant_bottleneck_model.bn2_quant_conv1.weight_quant.scale()
    block_2_weight_scale2 = quant_bottleneck_model.bn2_quant_conv2.weight_quant.scale()
    block_2_weight_scale3 = quant_bottleneck_model.bn2_quant_conv3.weight_quant.scale()
    block_2_combined_scale1 = -torch.log2(
        block_2_inp_scale1 * block_2_weight_scale1 / block_2_relu_1
    )
    block_2_combined_scale2 = -torch.log2(
        block_2_relu_1 * block_2_weight_scale2 / block_2_relu_2
    )
    block_2_combined_scale3 = -torch.log2(
        block_2_relu_2 * block_2_weight_scale3 / block_2_inp_scale1
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

    # print("combined_scale after conv1x1:", ( block_0_relu_2 * block_0_weight_scale3).item())

    # ------------------------------------------------------
    # Reorder input data-layout
    # ------------------------------------------------------

    block_2_int_weight_1 = quant_bottleneck_model.bn2_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_2_int_weight_2 = quant_bottleneck_model.bn2_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_2_int_weight_3 = quant_bottleneck_model.bn2_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    print("Writing golden output txt file.")
    golden_output.tofile(log_dir + "golden_output.txt", sep=",", format="%d")
    ds = DataShaper()
    before_input = int_inp.squeeze().data.numpy().astype(dtype_in)

    print("Writing input txt file.")
    before_input.tofile(log_dir + "before_ifm_mem_fmt_1x1.txt", sep=",", format="%d")
    ifm_mem_fmt = ds.reorder_mat(before_input, "YCXC8", "CYX")
    ifm_mem_fmt.tofile(log_dir + "after_ifm_mem_fmt.txt", sep=",", format="%d")

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

    total_wts = np.concatenate((bn2_wts1, bn2_wts2, bn2_wts3), axis=None)

    print("Writing weights txt files.")
    total_wts.tofile(log_dir + "after_weights_mem_fmt_final.txt", sep=",", format="%d")
    print(total_wts.shape)

    print("Done.")


if __name__ == "__main__":
    main()
