# test_decode_coded_e2e.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# REQUIRES: xrt_python_bindings

"""decode_coded returns every bf16 bit pattern, special values included.

The factory's sample draws normal weights; these cases instead give the
device kernel all 65536 patterns (NaN payloads, both infinities, -0 and the
subnormals among them), once inside the windows and once as exceptions,
and compare its output with the original bits.
"""

import numpy as np
from aie.iron import kernels
from aie.iron.algorithms import kernel_design as kd
from aie.utils import coded
from ml_dtypes import bfloat16

_K = 2048
_M = 4


def _run(patterns: np.ndarray) -> None:
    calls = patterns.shape[0] // _M
    rows = patterns.reshape(calls, _M, _K).view(bfloat16)
    a = coded.encode_rows(rows)
    harness = kd.KernelDesign(kernels.decode_coded(dim_m=_M, dim_k=_K), calls=calls)
    got = harness.run([a])
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint16).reshape(-1), patterns.reshape(-1)
    )


def test_every_bf16_pattern_decodes_inside_its_window():
    # In bit order a row of 2048 holds 8 consecutive high bytes of one sign,
    # so every pattern lands in a window and none is an exception.
    _run(np.arange(1 << 16, dtype=np.uint16).reshape(-1, _K))


def test_special_values_decode_as_exceptions():
    specials = np.array(
        [0x7FC1, 0xFFC0, 0x7F80, 0xFF80, 0x8000, 0x0001, 0x807F, 0x7F7F],
        np.uint16,
    )
    rows = np.full((_M * 2, _K), 0x3F80, np.uint16)
    rows[:, 0] = specials[: _M * 2]
    rows[:, -1] = specials[::-1][: _M * 2]
    rows[0, 1 : 1 + specials.size] = specials
    _run(rows)
