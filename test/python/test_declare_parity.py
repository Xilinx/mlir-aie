# test_declare_parity.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""A generator's signature and a class body declare one kernel alike — no NPU required."""

from __future__ import annotations

import numpy as np
from aie.utils.compile.jit._introspect import declare
from aie.utils.compile.jit.markers import CompileTime, DispatchTime, In, Out
from aie.utils.compile.jit.operator import Operator
from aie.utils.compile.jit.params import param, per_call


def scale(
    a: In,
    b: Out,
    *,
    M: CompileTime[int],
    rows: CompileTime[int] = 4,
    n: DispatchTime[np.int32],
):
    pass


class Scale(Operator):
    M: int = param()
    rows: int = param(default=4)
    a = In(M)
    b = Out(M)
    n = per_call(np.int32)


def record(d):
    return (
        [(name, p.default, p.tunable) for name, p in d.params.items()],
        [(m.name, type(m)) for m in d.operands],
        [(m.name, m.dtype) for m in d.per_calls],
    )


def test_a_generator_and_a_class_body_make_one_record():
    assert record(declare(scale)) == record(Scale.declaration)
    assert declare(scale).signature.parameters["M"].annotation == CompileTime[int]


def double(x):
    return 2 * x


def mapped(a: In, *, f: CompileTime[object] = double):
    pass


def test_a_function_compile_time_default_is_the_value():
    assert declare(mapped).params["f"].default(None) is double
