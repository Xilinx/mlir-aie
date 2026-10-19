#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""bn8 brevitas fixture generator.  Delegates to ``_gen_golden_template``."""

from ._gen_golden_template import BottleneckSpec, generate_bottleneck_fixtures

_SPEC = BottleneckSpec(
    name="bn8",
    tensor_in_w=14,
    tensor_in_h=14,
    tensor_in_c=80,
    tensor_out_c=80,
    depthwise_stride=1,
    depthwise_channels=184,
    has_skip=True,
    calibrate_images=False,
)


def main():
    generate_bottleneck_fixtures(_SPEC)


if __name__ == "__main__":
    main()
