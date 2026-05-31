#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
"""Conv2d 1x1 int8 host test: build golden, run on NPU, compare."""

import argparse
import sys

import torch
import torch.nn as nn

from aie.utils.hostruntime.argparse import add_runtime_args
from aie.utils.ml import run_conv_torch_test

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)


def main(opts):
    width = int(opts.width)
    height = int(opts.height)
    ci = int(opts.in_channels)
    co = int(opts.out_channels)
    co8 = co // 8

    conv_scale = 7.6294e-06  # int8 x int8 → int32 → float
    int8_scale = 0.0078

    int_inp = torch.randint(1, 20, (1, ci, height, width)).type(torch.FloatTensor)
    int_weight = torch.randint(50, 80, (co, ci, 1, 1)).type(torch.FloatTensor)

    class Conv2dInt(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(ci, co, kernel_size=1, bias=False)

        def forward(self, x):
            out_int = self.conv(x)
            out_quant = out_int * conv_scale
            return int8_scale * torch.clamp(
                torch.round(out_quant / int8_scale), -128, 127
            )

    model = Conv2dInt()
    model.conv.weight.data.copy_(int_weight)

    run_conv_torch_test(
        xclbin_path=opts.xclbin,
        insts_path=opts.instr,
        kernel_name=opts.kernel,
        golden_model=model,
        int_inp=int_inp,
        int_weights=[int_weight],
        out_shape_in_layout=(height, co8, width, 8),
        out_shape_final=(co, height, width),
        out_scale=int8_scale,
        atol=2 * int8_scale,
        trace_size=opts.trace_size,
        trace_file="log/trace_conv2d.txt" if opts.trace_size else None,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_runtime_args(p, with_io_sizes=True)
    p.add_argument("-wd", "--width", default=32, help="conv tile width")
    p.add_argument("-ht", "--height", default=32, help="conv tile height")
    p.add_argument("-ic", "--in_channels", default=64, help="input channels")
    p.add_argument("-oc", "--out_channels", default=64, help="output channels")
    main(p.parse_args(sys.argv[1:]))
