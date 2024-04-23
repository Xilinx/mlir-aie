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

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)


def main(opts):
    design = "resnet_conv2_x_int8"
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
    dtype_out = np.dtype("uint8")

    shape_in_act = (32, 8, 32, 8)
    shape_total_wts = (212992, 1)
    shape_out = (32, 32, 32, 8)

    # ------------------------------------------------------
    # Initialize activation, weights, scaling factor for int8 model
    # ------------------------------------------------------
    int_inp = torch.randint(1, 10, (1, 64, 32, 32)).type(torch.FloatTensor)
    block_0_int_weight_1 = torch.randint(10, 20, (64, 64, 1, 1)).type(torch.FloatTensor)
    block_0_int_weight_2 = torch.randint(10, 20, (64, 64, 3, 3)).type(torch.FloatTensor)
    block_0_int_weight_3 = torch.randint(10, 20, (256, 64, 1, 1)).type(
        torch.FloatTensor
    )
    block_0_int_weight_skip = torch.randint(10, 20, (256, 64, 1, 1)).type(
        torch.FloatTensor
    )

    block_1_int_weight_1 = torch.randint(20, 30, (64, 256, 1, 1)).type(
        torch.FloatTensor
    )
    block_1_int_weight_2 = torch.randint(20, 30, (64, 64, 3, 3)).type(torch.FloatTensor)
    block_1_int_weight_3 = torch.randint(20, 30, (256, 64, 1, 1)).type(
        torch.FloatTensor
    )

    block_2_int_weight_1 = torch.randint(30, 40, (64, 256, 1, 1)).type(
        torch.FloatTensor
    )
    block_2_int_weight_2 = torch.randint(30, 40, (64, 64, 3, 3)).type(torch.FloatTensor)
    block_2_int_weight_3 = torch.randint(30, 40, (256, 64, 1, 1)).type(
        torch.FloatTensor
    )

    init_scale = 0.5
    block_0_relu_1 = 0.5
    block_0_relu_2 = 0.5
    block_0_relu_3 = 0.5

    block_0_weight_scale1 = 0.5
    block_0_weight_scale2 = 0.5
    block_0_weight_scale3 = 0.5
    block_0_weight_scale_skip = 0.5

    block_1_relu_1 = 0.5
    block_1_relu_2 = 0.5
    block_1_relu_3 = 0.5

    block_1_weight_scale1 = 0.5
    block_1_weight_scale2 = 0.5
    block_1_weight_scale3 = 0.5
    block_1_quant_add_1 = 0.5

    block_2_relu_1 = 0.5
    block_2_relu_2 = 0.5
    block_2_relu_3 = 0.5

    block_2_weight_scale1 = 0.5
    block_2_weight_scale2 = 0.5
    block_2_weight_scale3 = 0.5
    block_2_quant_add_1 = 0.5

    block_0_combined_scale1 = -math.log2(
        init_scale * block_0_weight_scale1 / block_0_relu_1
    )  # RHS after first conv1x1 | clip 0-->255
    block_0_combined_scale2 = -math.log2(
        block_0_relu_1 * block_0_weight_scale2 / block_0_relu_2
    )  # RHS after second conv3x3 | clip 0-->255
    block_0_combined_scale3 = -math.log2(
        block_0_relu_2 * block_0_weight_scale3 / init_scale
    )  # RHS after third conv1x1 | clip -128-->+127
    block_0_combined_scale_skip = -math.log2(
        init_scale * block_0_weight_scale_skip / init_scale
    )  # LHS after conv1x1 | clip -128-->+127
    block_0_combined_scale4 = -math.log2(
        init_scale / block_0_relu_3
    )  # After addition | clip 0-->255

    block_1_combined_scale1 = -math.log2(
        block_0_relu_3 * block_1_weight_scale1 / block_1_relu_1
    )  # RHS after first conv1x1 | clip 0-->255
    block_1_combined_scale2 = -math.log2(
        block_1_relu_1 * block_1_weight_scale2 / block_1_relu_2
    )  # RHS after second conv3x3 | clip 0-->255
    block_1_combined_scale3 = -math.log2(
        block_1_relu_2 * block_1_weight_scale3 / block_1_quant_add_1
    )  # RHS after third conv1x1 | clip -128-->+127
    block_1_combined_scale4 = -math.log2(
        block_1_quant_add_1 / block_1_relu_3
    )  # After addition | clip 0-->255

    block_2_combined_scale1 = -math.log2(
        block_1_relu_3 * block_2_weight_scale1 / block_2_relu_1
    )  # RHS after first conv1x1 | clip 0-->255
    block_2_combined_scale2 = -math.log2(
        block_2_relu_1 * block_2_weight_scale2 / block_2_relu_2
    )  # RHS after second conv3x3 | clip 0-->255
    block_2_combined_scale3 = -math.log2(
        block_2_relu_2 * block_2_weight_scale3 / block_2_quant_add_1
    )  # RHS after third conv1x1 | clip -128-->+127
    block_2_combined_scale4 = -math.log2(
        block_2_quant_add_1 / block_2_relu_3
    )  # After addition | clip 0-->255

    min = 0
    max = 255

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

    # ------------------------------------------------------
    # Define your golden reference
    # ------------------------------------------------------
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
                planes,
                planes,
                kernel_size=3,
                padding=1,
                padding_mode="zeros",
                bias=False,
            )
            self.block_0_conv3 = nn.Conv2d(
                planes, self.expansion * planes, kernel_size=1, bias=False
            )

            self.block_0_relu1 = nn.ReLU()
            self.block_0_relu2 = nn.ReLU()
            self.block_0_relu3 = nn.ReLU()

            # Bottleneck 1
            self.block_1_conv1 = nn.Conv2d(
                self.expansion * planes, planes, kernel_size=1, bias=False
            )
            self.block_1_conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                padding=1,
                padding_mode="zeros",
                bias=False,
            )
            self.block_1_conv3 = nn.Conv2d(
                planes, self.expansion * planes, kernel_size=1, bias=False
            )

            self.block_1_relu1 = nn.ReLU()
            self.block_1_relu2 = nn.ReLU()
            self.block_1_relu3 = nn.ReLU()

            # Bottleneck 2
            self.block_2_conv1 = nn.Conv2d(
                self.expansion * planes, planes, kernel_size=1, bias=False
            )
            self.block_2_conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                padding=1,
                padding_mode="zeros",
                bias=False,
            )
            self.block_2_conv3 = nn.Conv2d(
                planes, self.expansion * planes, kernel_size=1, bias=False
            )

            self.block_2_relu1 = nn.ReLU()
            self.block_2_relu2 = nn.ReLU()
            self.block_2_relu3 = nn.ReLU()

        def forward(self, x):
            # **************** Bottleneck 0 ****************
            block_0_conv1_out = (
                self.block_0_conv1(x) * init_scale * block_0_weight_scale1
            )
            block_0_relu1_out = torch.clamp(
                torch.round(self.block_0_relu1(block_0_conv1_out) / block_0_relu_1),
                min,
                max,
            )  # convert to int and apply relu
            block_0_conv2_out = (
                self.block_0_conv2(block_0_relu1_out)
                * block_0_relu_1
                * block_0_weight_scale2
            )
            block_0_relu2_out = torch.clamp(
                torch.round(self.block_0_relu2(block_0_conv2_out) / block_0_relu_2),
                min,
                max,
            )
            block_0_conv3_out = (
                self.block_0_conv3(block_0_relu2_out)
                * block_0_relu_2
                * block_0_weight_scale3
            )
            block_0_rhf_same_scale = torch.clamp(
                torch.round(block_0_conv3_out / init_scale), -128, 127
            )

            block_0_lhs_conv = self.shortcut(x) * init_scale * block_0_weight_scale_skip
            block_0_lhs_same_scale = torch.clamp(
                torch.round(block_0_lhs_conv / init_scale), -128, 127
            )
            # convert to int and apply relu

            block_0_skip_add = init_scale * (
                block_0_rhf_same_scale + block_0_lhs_same_scale
            )
            block_0_final_out = torch.clamp(
                torch.round(self.block_0_relu3(block_0_skip_add) / block_0_relu_3),
                min,
                max,
            )
            # **************** Bottleneck 1 ****************
            block_1_conv1_out = (
                self.block_1_conv1(block_0_final_out)
                * block_0_relu_3
                * block_1_weight_scale1
            )
            block_1_relu1_out = torch.clamp(
                torch.round(self.block_1_relu1(block_1_conv1_out) / block_1_relu_1),
                min,
                max,
            )  # convert to int and apply relu
            block_1_conv2_out = (
                self.block_1_conv2(block_1_relu1_out)
                * block_1_relu_1
                * block_1_weight_scale2
            )
            block_1_relu2_out = torch.clamp(
                torch.round(self.block_1_relu2(block_1_conv2_out) / block_1_relu_2),
                min,
                max,
            )
            block_1_conv3_out = (
                self.block_1_conv3(block_1_relu2_out)
                * block_1_relu_2
                * block_1_weight_scale3
            )
            block_1_rhf_same_scale = torch.clamp(
                torch.round(block_1_conv3_out / block_0_relu_3), -128, 127
            )

            block_1_skip_add = block_0_relu_3 * (
                block_1_rhf_same_scale + block_0_final_out
            )
            block_1_final_out = torch.clamp(
                torch.round(self.block_1_relu3(block_1_skip_add) / block_1_relu_3),
                min,
                max,
            )

            # **************** Bottleneck 2 ****************
            block_2_conv1_out = (
                self.block_2_conv1(block_1_final_out)
                * block_1_relu_3
                * block_2_weight_scale1
            )
            block_2_relu1_out = torch.clamp(
                torch.round(self.block_2_relu1(block_2_conv1_out) / block_2_relu_1),
                min,
                max,
            )  # convert to int and apply relu
            block_2_conv2_out = (
                self.block_2_conv2(block_2_relu1_out)
                * block_2_relu_1
                * block_2_weight_scale2
            )
            block_2_relu2_out = torch.clamp(
                torch.round(self.block_2_relu2(block_2_conv2_out) / block_2_relu_2),
                min,
                max,
            )
            block_2_conv3_out = (
                self.block_2_conv3(block_2_relu2_out)
                * block_2_relu_2
                * block_2_weight_scale3
            )
            block_2_rhf_same_scale = torch.clamp(
                torch.round(block_2_conv3_out / block_1_relu_3), -128, 127
            )

            block_2_skip_add = block_1_relu_3 * (
                block_2_rhf_same_scale + block_1_final_out
            )
            block_2_final_out = block_2_relu_3 * (
                torch.clamp(
                    torch.round(self.block_2_relu3(block_2_skip_add) / block_2_relu_3),
                    min,
                    max,
                )
            )
            return block_2_final_out

    # ------------------------------------------------------
    # Pytorch baseline
    # ------------------------------------------------------
    model = resnet_conv2_x_int8()
    model.eval()
    model.block_0_conv1.weight.data.copy_(block_0_int_weight_1)
    model.block_0_conv2.weight.data.copy_(block_0_int_weight_2)
    model.block_0_conv3.weight.data.copy_(block_0_int_weight_3)
    model.shortcut.weight.data.copy_(block_0_int_weight_skip)

    model.block_1_conv1.weight.data.copy_(block_1_int_weight_1)
    model.block_1_conv2.weight.data.copy_(block_1_int_weight_2)
    model.block_1_conv3.weight.data.copy_(block_1_int_weight_3)

    model.block_2_conv1.weight.data.copy_(block_2_int_weight_1)
    model.block_2_conv2.weight.data.copy_(block_2_int_weight_2)
    model.block_2_conv3.weight.data.copy_(block_2_int_weight_3)

    golden_output = model(int_inp)

    # ------------------------------------------------------
    # Reorder input data-layout
    # ------------------------------------------------------
    ds = DataShaper()
    before_input = int_inp.squeeze().data.numpy().astype(dtype_in)
    before_input.tofile(
        log_folder + "/before_ifm_mem_fmt_1x1.txt", sep=",", format="%d"
    )
    ifm_mem_fmt = ds.reorder_mat(before_input, "YCXC8", "CYX")
    ifm_mem_fmt.tofile(log_folder + "/after_ifm_mem_fmt_1x1.txt", sep=",", format="%d")

    block0_wts1 = ds.reorder_mat(
        block_0_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    block0_wts2 = ds.reorder_mat(
        block_0_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    block0_wts3 = ds.reorder_mat(
        block_0_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    block0_wts_skip = ds.reorder_mat(
        block_0_int_weight_skip.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )

    total_wts = np.concatenate(
        (block0_wts1, block0_wts2, block0_wts3, block0_wts_skip), axis=None
    )

    block1_wts1 = ds.reorder_mat(
        block_1_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    block1_wts2 = ds.reorder_mat(
        block_1_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    block1_wts3 = ds.reorder_mat(
        block_1_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )

    total_wts2 = np.concatenate(
        (total_wts, block1_wts1, block1_wts2, block1_wts3), axis=None
    )

    block2_wts1 = ds.reorder_mat(
        block_2_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    block2_wts2 = ds.reorder_mat(
        block_2_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    block2_wts3 = ds.reorder_mat(
        block_2_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )

    total_wts3 = np.concatenate(
        (total_wts2, block2_wts1, block2_wts2, block2_wts3), axis=None
    )

    total_wts3.tofile(log_folder + "/weights_mem_fmt_final.txt", sep=",", format="%d")

    # ------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------
    for i in range(num_iter):
        start = time.time_ns()
        aie_output = execute(app, ifm_mem_fmt, total_wts) * block_2_relu_3
        stop = time.time_ns()

        if enable_trace:
            aie_output, trace = extract_trace(
                aie_output, shape_out, dtype_out, trace_size
            )
            write_out_trace(trace, trace_file)

        npu_time = stop - start
        npu_time_total = npu_time_total + npu_time

    # ------------------------------------------------------
    # Reorder output data-layout
    # ------------------------------------------------------
    temp_out = aie_output.reshape(32, 32, 32, 8)
    temp_out = ds.reorder_mat(temp_out, "CDYX", "YCXD")
    ofm_mem_fmt = temp_out.reshape(256, 32, 32)
    ofm_mem_fmt.tofile(
        log_folder + "/after_ofm_mem_fmt_final.txt", sep=",", format="%d"
    )
    ofm_mem_fmt_out = torch.from_numpy(ofm_mem_fmt).unsqueeze(0)

    # ------------------------------------------------------
    # Compare the AIE output and the golden reference
    # ------------------------------------------------------
    print("\nAvg NPU time: {}us.".format(int((npu_time_total / num_iter) / 1000)))

    assert np.allclose(
        ofm_mem_fmt_out.detach().numpy(),
        golden_output.detach().numpy(),
        rtol=0,
        atol=block_2_relu_3,
    )

    print("\nPASS!\n")


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
