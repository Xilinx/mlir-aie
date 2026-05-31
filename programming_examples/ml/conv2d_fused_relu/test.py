#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
"""Conv2d 1x1 + fused ReLU int8 host test."""

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
    conv_scale = 0.0039
    relu_scale = 0.0078

    int_inp = torch.randint(1, 100, (1, 64, 32, 32)).type(torch.FloatTensor)
    int_weight = torch.randint(50, 100, (64, 64, 1, 1)).type(torch.FloatTensor)

    class Conv2dReluInt(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(64, 64, kernel_size=1, bias=False)
            self.relu = nn.ReLU()

        def forward(self, x):
            out_float = self.conv(x) * conv_scale
            return relu_scale * torch.clamp(
                torch.round(self.relu(out_float) / relu_scale), 0, 255
            )

    model = Conv2dReluInt()
    model.conv.weight.data.copy_(int_weight)

    run_conv_torch_test(
        xclbin_path=opts.xclbin,
        insts_path=opts.instr,
        golden_model=model,
        int_inp=int_inp,
        int_weights=[int_weight],
        out_shape_in_layout=(32, 8, 32, 8),
        out_shape_final=(64, 32, 32),
        out_scale=relu_scale,
        atol=2 * relu_scale,
        dtype_out=np.uint8,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_runtime_args(p, with_io_sizes=True)
    main(p.parse_args(sys.argv[1:]))
