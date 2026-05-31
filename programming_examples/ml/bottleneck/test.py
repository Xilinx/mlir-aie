#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
"""ResNet-style int8 bottleneck (3 convs + skip add) host test."""

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn

from aie.utils.hostruntime.argparse import add_runtime_args
from aie.utils.ml import run_conv_torch_test

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)


def main(opts):
    inp_scale1 = inp_scale2 = inp_scale3 = inp_scale4 = 0.5
    weight_scale1 = weight_scale2 = weight_scale3 = 0.5

    int_inp = torch.randint(1, 100, (1, 256, 32, 32)).type(torch.FloatTensor)
    int_weight1 = torch.randint(50, 100, (64, 256, 1, 1)).type(torch.FloatTensor)
    int_weight2 = torch.randint(50, 100, (64, 64, 3, 3)).type(torch.FloatTensor)
    int_weight3 = torch.randint(50, 100, (256, 64, 1, 1)).type(torch.FloatTensor)

    class BottleneckInt8(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
            self.conv2 = nn.Conv2d(
                64, 64, kernel_size=3, padding=1, padding_mode="zeros", bias=False
            )
            self.conv3 = nn.Conv2d(64, 256, kernel_size=1, bias=False)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()

        def forward(self, x):
            c1 = self.conv1(x) * inp_scale1 * weight_scale1
            r1 = torch.clamp(torch.round(self.relu1(c1) / inp_scale2), 0, 255)
            c2 = self.conv2(r1) * inp_scale2 * weight_scale2
            r2 = torch.clamp(torch.round(self.relu2(c2) / inp_scale3), 0, 255)
            c3 = self.conv3(r2) * inp_scale3 * weight_scale3
            same_scale = torch.clamp(torch.round(c3 / inp_scale1), -128, 127)
            skip = inp_scale1 * (same_scale + int_inp)
            return inp_scale4 * torch.clamp(torch.round(skip / inp_scale4), 0, 255)

    model = BottleneckInt8()
    model.conv1.weight.data.copy_(int_weight1)
    model.conv2.weight.data.copy_(int_weight2)
    model.conv3.weight.data.copy_(int_weight3)

    run_conv_torch_test(
        xclbin_path=opts.xclbin,
        insts_path=opts.instr,
        golden_model=model,
        int_inp=int_inp,
        int_weights=[int_weight1, int_weight2, int_weight3],
        out_shape_in_layout=(32, 32, 32, 8),
        out_shape_final=(256, 32, 32),
        out_scale=inp_scale4,
        atol=inp_scale4,
        dtype_out=np.uint8,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_runtime_args(p, with_io_sizes=True)
    main(p.parse_args(sys.argv[1:]))
