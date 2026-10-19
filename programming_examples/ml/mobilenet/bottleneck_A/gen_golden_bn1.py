#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""bn1 brevitas fixture generator.  Delegates to ``_gen_golden_template``.

The full recipe (3-conv bottleneck, per-variant scale extraction, fixture
filenames) lives in
``bottleneck_A._gen_golden_template.generate_bottleneck_fixtures``;
this script just supplies the per-block shape constants and structural
flags.  See ``_gen_golden_template.py`` for the rationale.
"""

from ._gen_golden_template import BottleneckSpec, generate_bottleneck_fixtures

_SPEC = BottleneckSpec(
    name="bn1",
    tensor_in_w=112,
    tensor_in_h=112,
    tensor_in_c=16,
    tensor_out_c=24,
    depthwise_stride=2,
    depthwise_channels=64,
    has_skip=False,
    calibrate_images=True,
)


def main():
    generate_bottleneck_fixtures(_SPEC)


if __name__ == "__main__":
    main()
