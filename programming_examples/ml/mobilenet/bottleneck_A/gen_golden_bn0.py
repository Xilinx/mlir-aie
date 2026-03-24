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

# bn0
tensorInW = 112
tensorInH = 112
tensorInC = 16
tensorOutC = tensorInC

# depthwiseStride = 1
# depthWiseChannels = 72

bneck_0_InW2 = tensorInW
bneck_0_InH2 = tensorInH
bneck_0_InC2 = tensorInC
bneck_0_OutC2 = bneck_0_InC2

bneck_0_InW3 = bneck_0_InW2
bneck_0_InH3 = bneck_0_InH2
bneck_0_OutC3 = tensorOutC

bneck_0_InC2_vec = math.floor(bneck_0_InC2 / vectorSize)
bneck_0_OutC3_vec = math.floor(bneck_0_OutC3 / vectorSize)


def main():
    print("Running torch reference model for bottleneck_A_subblocks for bn1 ...")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # ------------------------------------------------------
    # Configure this to match your design's buffer size
    # ------------------------------------------------------
    dtype_in = np.dtype("int8")
    dtype_wts = np.dtype("int8")
    dtype_out = np.dtype("int8")

    shape_total_wts = (3 * 3 * bneck_0_OutC2 + bneck_0_OutC2 * bneck_0_OutC3, 1)
    shape_in_act = (
        bneck_0_InH2,
        bneck_0_InC2_vec,
        bneck_0_InW2,
        vectorSize,
    )  #'YCXC8' , 'CYX'
    shape_out = (bneck_0_InH3, bneck_0_OutC3_vec, bneck_0_InW3, vectorSize)  # HCWC8
    shape_out_final = (
        bneck_0_OutC3_vec * vectorSize,
        bneck_0_InH3,
        bneck_0_InW3,
    )  # CHW

    # ------------------------------------------------------
    # Initialize activation, weights, scaling factor for int8 model
    # ------------------------------------------------------
    input = torch.randn(1, bneck_0_InC2_vec * vectorSize, bneck_0_InH2, bneck_0_InW2)

    class QuantBottleneck0(nn.Module):
        def __init__(self, in_planes=16, bn0_expand=16, bn0_project=16):
            super(QuantBottleneck0, self).__init__()
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
            self.bn0_quant_id_2.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl = (
                self.quant_id_1.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl
            )
            self.bn0_quant_id_2.act_quant.fused_activation_quant_proxy.tensor_quant.int_scaling_impl = (
                self.quant_id_1.act_quant.fused_activation_quant_proxy.tensor_quant.int_scaling_impl
            )

        def forward(self, x):
            out_q = self.quant_id_1(x)
            out = self.bn0_quant_conv2(out_q)
            out = self.bn0_quant_relu2(out)
            out = self.bn0_quant_conv3(out)
            out = self.bn0_quant_id_2(out)
            out = out + out_q
            out = self.bn0_add(out)
            return out

    quant_bottleneck_model = QuantBottleneck0(
        in_planes=bneck_0_InC2, bn0_expand=bneck_0_InC2, bn0_project=bneck_0_OutC3
    )

    # import pathlib
    sys.path.append("..")
    from mb_utils import ExpandChannels
    from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
    import torchvision
    import torch.utils.data as data_utils
    from torchvision import transforms

    # Define the image preprocessing pipeline
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ExpandChannels(target_channels=16),  # Expand to 80 channels
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

    # calibrate([(torch.rand(1, bneck_0_InC2, bneck_0_InH2, bneck_0_InW2), 1) for _ in range(5)], quant_bottleneck_model)

    quant_bottleneck_model.eval()

    q_bottleneck_out = quant_bottleneck_model(input)
    golden_output = (
        q_bottleneck_out.int(float_datatype=True).data.numpy().astype(dtype_out)
    )
    print("Golden::Brevitas::", golden_output)
    q_inp = quant_bottleneck_model.quant_id_1(input)
    int_inp = q_inp.int(float_datatype=True)

    block_0_inp_scale = quant_bottleneck_model.quant_id_1.act_quant.scale()

    block_0_relu_2 = quant_bottleneck_model.bn0_quant_relu2.act_quant.scale()
    block_0_skip_add = quant_bottleneck_model.bn0_add.act_quant.scale()

    block_0_weight_scale2 = quant_bottleneck_model.bn0_quant_conv2.weight_quant.scale()
    block_0_weight_scale3 = quant_bottleneck_model.bn0_quant_conv3.weight_quant.scale()

    block_0_combined_scale2 = -torch.log2(
        block_0_inp_scale * block_0_weight_scale2 / block_0_relu_2
    )
    block_0_combined_scale3 = -torch.log2(
        block_0_relu_2 * block_0_weight_scale3 / block_0_inp_scale
    )
    block_0_combined_scale_skip = -torch.log2(
        block_0_inp_scale / block_0_skip_add
    )  # After addition | clip -128-->127

    print("********************BN0*******************************")
    print("combined_scale after conv3x3:", block_0_combined_scale2.item())
    print("combined_scale after conv1x1:", block_0_combined_scale3.item())
    print("combined_scale after skip add:", block_0_combined_scale_skip.item())
    print("********************BN0*******************************")

    # ------------------------------------------------------
    # Reorder input data-layout
    # ------------------------------------------------------

    block_0_int_weight_2 = quant_bottleneck_model.bn0_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_0_int_weight_3 = quant_bottleneck_model.bn0_quant_conv3.quant_weight().int(
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

    # **************************** bn0 ****************************

    bn0_wts2 = ds.reorder_mat(
        block_0_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn0_wts3 = ds.reorder_mat(
        block_0_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )

    bn0_wts2.tofile(
        log_dir + "bn0_layer2_after_weights_mem_fmt_final.txt",
        sep=",",
        format="%d",
    )
    bn0_wts3.tofile(
        log_dir + "bn0_layer3_after_weights_mem_fmt_final.txt",
        sep=",",
        format="%d",
    )
    total_wts = np.concatenate((bn0_wts2, bn0_wts3), axis=None)

    print("Writing weights txt files.")
    total_wts.tofile(
        log_dir + "bn0_after_weights_mem_fmt_final.txt", sep=",", format="%d"
    )
    print(total_wts.shape)

    print("Done.")


if __name__ == "__main__":
    main()
