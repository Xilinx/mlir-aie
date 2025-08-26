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
from aie.utils.xrt import setup_aie, extract_trace, write_out_trace, execute
import aie.utils.test as test_utils
from brevitas.nn import QuantConv2d, QuantIdentity, QuantReLU
from brevitas.quant.fixed_point import (
    Int8ActPerTensorFixedPoint,
    Int8WeightPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)

sys.path.append("..")
from mb_utils import convert_to_numpy

torch.use_deterministic_algorithms(True)
import json


# Function to read scale factors from JSON file
def read_scale_factors(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# Function to write scale factors to JSON file
def write_scale_factors(file_path, scale_factors):
    with open(file_path, "w") as file:
        json.dump(scale_factors, file, indent=4)


# Read the existing scale factors
file_path = "scale_factors_fused_bn8+bn9.json"
scale_factors = read_scale_factors(file_path)

torch.manual_seed(0)
vectorSize = 8

tensorInW = 7
tensorInH = 7  # 7 NOLF set to 6 for debug to avoid needing dynamic objFIFO
tensorInC = 80  # NOLF to avoid L1 overflow due to weights

# config for bn8
bneck_7_OutC3 = tensorInC

bneck_8_tensorInW = tensorInW
bneck_8_tensorInH = tensorInH
bneck_8_tensorInC = tensorInC
bneck_8_tensorOutC = tensorInC
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


# config for bn9
bneck_9_tensorInW = bneck_8_InW3
bneck_9_tensorInH = bneck_8_InH3
bneck_9_tensorInC = bneck_8_OutC3
bneck_9_tensorOutC = tensorInC
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

tensorOutW = bneck_9_InW3
tensorOutH = bneck_9_InH3
tensorOutC = bneck_9_OutC3

InC_vec = math.floor(tensorInC / vectorSize)
OutC_vec = math.floor(tensorOutC / vectorSize)


def main(opts):
    design = "mobilenet_bottleneck_A_bn8_bn9"
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
    dtype_in = np.dtype("int8")
    dtype_wts = np.dtype("int8")
    dtype_out = np.dtype("int8")

    shape_total_wts = (
        (
            bneck_8_InC1 * bneck_8_OutC1
            + 3 * 3 * bneck_8_OutC2
            + bneck_8_OutC2 * bneck_8_OutC3
        )
        + (
            bneck_9_InC1 * bneck_9_OutC1
            + 3 * 3 * bneck_9_OutC2
            + bneck_9_OutC2 * bneck_9_OutC3
        ),
        1,
    )
    print("Total weights size: ", shape_total_wts)

    shape_in_act = (tensorInH, InC_vec, tensorInW, vectorSize)  #'YCXC8' , 'CYX'
    shape_out = (tensorOutH, OutC_vec, tensorOutW, vectorSize)  # HCWC8
    size_out = tensorOutH * OutC_vec * tensorOutW * vectorSize
    shape_out_final = (OutC_vec * vectorSize, tensorOutH, tensorOutW)  # CHW

    # ------------------------------------------------------
    # Initialize activation, weights, scaling factor for int8 model
    # ------------------------------------------------------
    input = torch.randn(1, InC_vec * vectorSize, tensorInH, tensorInW)
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
        dtype_out,
        enable_trace=enable_trace,
        trace_size=trace_size,
    )

    class QuantBottleneck0(nn.Module):
        def __init__(
            self,
            bn7_project=40,
            bn8_expand=92,
            bn8_project=40,
            bn9_expand=92,
            bn9_project=40,
        ):
            super(QuantBottleneck0, self).__init__()
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

        def forward(self, x):
            out_q = self.bn7_add(x)

            # bn8

            out = self.bn8_quant_conv1(out_q)
            out = self.bn8_quant_relu1(out)
            out = self.bn8_quant_conv2(out)
            out = self.bn8_quant_relu2(out)
            out = self.bn8_quant_conv3(out)
            out = self.bn7_add(out)
            out = out + out_q
            out_q = self.bn8_add(out)

            # bn9

            out = self.bn9_quant_conv1(out_q)
            out = self.bn9_quant_relu1(out)
            out = self.bn9_quant_conv2(out)
            out = self.bn9_quant_relu2(out)
            out = self.bn9_quant_conv3(out)
            out = self.bn8_add(out)
            out = out + out_q
            out = self.bn9_add(out)
            return out  # return out_q

    quant_bottleneck_model = QuantBottleneck0(
        bn7_project=bneck_7_OutC3,
        bn8_expand=bneck_8_OutC1,
        bn8_project=bneck_8_OutC3,
        bn9_expand=bneck_9_OutC1,
        bn9_project=bneck_9_OutC3,
    )

    from mb_utils import ExpandChannels
    from brevitas_examples.imagenet_classification.ptq.ptq_common import calibrate
    import torchvision
    import torch.utils.data as data_utils
    from torchvision import transforms

    # Define the image preprocessing pipeline
    transform = transforms.Compose(
        [
            transforms.Resize(128),
            transforms.CenterCrop(tensorInW),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ExpandChannels(target_channels=tensorInC),  # Expand to 80 channels
        ]
    )
    data_dir = "data"

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

    # calibrate([(torch.rand(1, bneck_8_InC2, bneck_8_InH2, bneck_8_InW2), 1) for _ in range(5)], quant_bottleneck_model)

    quant_bottleneck_model.eval()

    q_bottleneck_out = quant_bottleneck_model(input)
    golden_output = (
        q_bottleneck_out.int(float_datatype=True).data.numpy().astype(dtype_out)
    )
    print("Golden::Brevitas::", golden_output)
    q_inp = quant_bottleneck_model.bn7_add(input)
    int_inp = q_inp.int(float_datatype=True)

    block_7_skip_add = quant_bottleneck_model.bn7_add.act_quant.scale()
    block_8_inp_scale1 = block_7_skip_add
    block_8_relu_1 = quant_bottleneck_model.bn8_quant_relu1.act_quant.scale()
    block_8_relu_2 = quant_bottleneck_model.bn8_quant_relu2.act_quant.scale()
    block_8_skip_add = quant_bottleneck_model.bn8_add.act_quant.scale()
    block_8_weight_scale1 = quant_bottleneck_model.bn8_quant_conv1.weight_quant.scale()
    block_8_weight_scale2 = quant_bottleneck_model.bn8_quant_conv2.weight_quant.scale()
    block_8_weight_scale3 = quant_bottleneck_model.bn8_quant_conv3.weight_quant.scale()

    block_8_combined_scale1 = -torch.log2(
        block_8_inp_scale1 * block_8_weight_scale1 / block_8_relu_1
    )
    block_8_combined_scale2 = -torch.log2(
        block_8_relu_1 * block_8_weight_scale2 / block_8_relu_2
    )
    block_8_combined_scale3 = -torch.log2(
        block_8_relu_2 * block_8_weight_scale3 / block_8_inp_scale1
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
    scale_factors["BN8"]["skip_add"] = int(block_8_combined_scale_skip.item())

    block_9_inp_scale1 = block_8_skip_add
    block_9_relu_1 = quant_bottleneck_model.bn9_quant_relu1.act_quant.scale()
    block_9_relu_2 = quant_bottleneck_model.bn9_quant_relu2.act_quant.scale()
    block_9_skip_add = quant_bottleneck_model.bn9_add.act_quant.scale()
    block_9_weight_scale1 = quant_bottleneck_model.bn9_quant_conv1.weight_quant.scale()
    block_9_weight_scale2 = quant_bottleneck_model.bn9_quant_conv2.weight_quant.scale()
    block_9_weight_scale3 = quant_bottleneck_model.bn9_quant_conv3.weight_quant.scale()
    block_9_combined_scale1 = -torch.log2(
        block_9_inp_scale1 * block_9_weight_scale1 / block_9_relu_1
    )
    block_9_combined_scale2 = -torch.log2(
        block_9_relu_1 * block_9_weight_scale2 / block_9_relu_2
    )
    block_9_combined_scale3 = -torch.log2(
        block_9_relu_2 * block_9_weight_scale3 / block_9_inp_scale1
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
    scale_factors["BN9"]["skip_add"] = int(block_9_combined_scale_skip.item())

    write_scale_factors(file_path, scale_factors)

    atol = block_9_skip_add
    # ------------------------------------------------------
    # Reorder input data-layout
    # ------------------------------------------------------

    block_8_int_weight_1 = quant_bottleneck_model.bn8_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_8_int_weight_2 = quant_bottleneck_model.bn8_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_8_int_weight_3 = quant_bottleneck_model.bn8_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    block_9_int_weight_1 = quant_bottleneck_model.bn9_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_9_int_weight_2 = quant_bottleneck_model.bn9_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_9_int_weight_3 = quant_bottleneck_model.bn9_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    golden_output.tofile(log_folder + "/golden_output.txt", sep=",", format="%d")
    ds = DataShaper()
    before_input = int_inp.squeeze().data.numpy().astype(dtype_in)
    before_input.tofile(
        log_folder + "/before_ifm_mem_fmt_1x1.txt", sep=",", format="%d"
    )
    ifm_mem_fmt = ds.reorder_mat(before_input, "YCXC8", "CYX")
    ifm_mem_fmt.tofile(log_folder + "/after_ifm_mem_fmt.txt", sep=",", format="%d")

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

    total_wts = np.concatenate((bn8_total_wts, bn9_total_wts), axis=None)

    total_wts.tofile(
        log_folder + "/after_weights_mem_fmt_final.txt", sep=",", format="%d"
    )
    print(total_wts.shape)
    print(input.shape)
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
    print(ofm_mem_fmt_out)
    # ------------------------------------------------------
    # Compare the AIE output and the golden reference
    # ------------------------------------------------------
    print("\nAvg NPU time: {}us.".format(int((npu_time_total / num_iter) / 1000)))
    golden = convert_to_numpy(golden_output)
    ofm_mem_fmt_out = convert_to_numpy(ofm_mem_fmt_out)
    max_diff_int = np.max((golden) - (ofm_mem_fmt_out))
    print("atol: {} max difference (int): {}".format(atol, max_diff_int))
    mismatch_indices = np.where(golden != ofm_mem_fmt_out)

    # Extract mismatch values
    mismatch_values_golden = golden[mismatch_indices]
    mismatch_values_ofm = ofm_mem_fmt_out[mismatch_indices]

    # Print mismatch indices and corresponding values
    print("golden shape: ", golden.shape)
    print("Output shape: ", ofm_mem_fmt_out.shape)
    print("Mismatch indices and corresponding values:")
    for idx, (golden_value, ofm_value) in zip(
        zip(*mismatch_indices), zip(mismatch_values_golden, mismatch_values_ofm)
    ):
        print(f"Index: {idx}, Golden value: {golden_value}, OFM value: {ofm_value}")
    if np.allclose(
        ofm_mem_fmt_out,
        golden_output,
        rtol=0,
        atol=1,
    ):
        print("\nPASS!\n")
        exit(0)
    else:
        print("\nFailed.\n")
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
