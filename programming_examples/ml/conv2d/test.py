#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
"""Conv2d 1x1 int8 host test (optionally fused with ReLU).

Build a torch golden model, run the compiled NPU design, compare. The
`--fuse_relu` flag must match the flag the design was compiled with
(see Makefile) — it picks the uint8 output dtype and the ReLU branch
in the reference model.
"""

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
    width = int(opts.width)
    height = int(opts.height)
    ci = int(opts.in_channels)
    co = int(opts.out_channels)
    co8 = co // 8

    if opts.fuse_relu:
        conv_scale = 0.0039
        out_scale = 0.0078
        clip_lo, clip_hi = 0, 255
        out_dtype = np.uint8
        int_inp_hi = 100
        int_wt_lo, int_wt_hi = 50, 100
    else:
        conv_scale = 7.6294e-06
        out_scale = 0.0078
        clip_lo, clip_hi = -128, 127
        out_dtype = np.int8
        int_inp_hi = 20
        int_wt_lo, int_wt_hi = 50, 80

    int_inp = torch.randint(1, int_inp_hi, (1, ci, height, width)).type(torch.FloatTensor)
    int_weight = torch.randint(int_wt_lo, int_wt_hi, (co, ci, 1, 1)).type(torch.FloatTensor)

    class Conv2dInt(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(ci, co, kernel_size=1, bias=False)
            self.relu = nn.ReLU() if opts.fuse_relu else None

        def forward(self, x):
            out_float = self.conv(x) * conv_scale
            if self.relu is not None:
                out_float = self.relu(out_float)
            return out_scale * torch.clamp(
                torch.round(out_float / out_scale), clip_lo, clip_hi
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
        out_scale=out_scale,
        atol=2 * out_scale,
        dtype_out=out_dtype,
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
    p.add_argument("--fuse_relu", action="store_true",
                   help="reference model includes ReLU + uses uint8 out dtype")
    main(p.parse_args(sys.argv[1:]))
