# kernel_harness.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Test-side entry point to the generic kernel design builder.

The implementation lives in :mod:`aie.iron.algorithms.kernel_design`, so
programming examples and kernel authors reach it without importing a test
harness. This module re-exports it -- the public API plus the few private
helpers the case table and the benchmark read -- so ``from aie.utils import
kernel_harness as kh`` keeps working.
"""

from aie.iron.algorithms.kernel_design import (
    HostArg,
    cycles_per_call,
    design,
    dtype_name,
    host_args,
    host_layout,
    is_matmul,
    is_matvec,
    output_size,
    sample_inputs,
    upload,
)
from aie.iron.algorithms.kernel_design import (
    _arg_types as _arg_types,
)
from aie.iron.algorithms.kernel_design import (
    _elems as _elems,
)
from aie.iron.algorithms.kernel_design import (
    _is_bfp as _is_bfp,
)
from aie.iron.algorithms.kernel_design import (
    _shape_dtype as _shape_dtype,
)

__all__ = [
    "HostArg",
    "cycles_per_call",
    "design",
    "dtype_name",
    "host_args",
    "host_layout",
    "is_matmul",
    "is_matvec",
    "output_size",
    "sample_inputs",
    "upload",
]
