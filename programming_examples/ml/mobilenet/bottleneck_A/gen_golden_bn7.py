#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
"""bn7 brevitas fixture generator.  Delegates to ``_gen_golden_template``."""

from pathlib import Path

from ._gen_golden_template import BottleneckSpec, generate_bottleneck_fixtures

_SPEC = BottleneckSpec(
    name="bn7",
    tensor_in_w=14,
    tensor_in_h=14,
    tensor_in_c=80,
    tensor_out_c=80,
    depthwise_stride=1,
    depthwise_channels=200,
    has_skip=True,
    calibrate_imagenet=False,
)


def main():
    generate_bottleneck_fixtures(_SPEC, log_dir=Path(__file__).parent / "log")


if __name__ == "__main__":
    main()
