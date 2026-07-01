#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""bn6 brevitas fixture generator.  Delegates to ``_gen_golden_template``."""

from pathlib import Path

from ._gen_golden_template import BottleneckSpec, generate_bottleneck_fixtures

_SPEC = BottleneckSpec(
    name="bn6",
    tensor_in_w=28,
    tensor_in_h=28,
    tensor_in_c=40,
    tensor_out_c=80,
    depthwise_stride=2,
    depthwise_channels=240,
    has_skip=False,
    calibrate_imagenet=False,
)


def main():
    generate_bottleneck_fixtures(_SPEC, log_dir=Path(__file__).parent / "log")


if __name__ == "__main__":
    main()
