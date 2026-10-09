# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

# An op built on the user's behalf is located at the user's statement, however
# many frames of the aie package lie between them.

import inspect

import numpy as np

from aie.dialects.aie import buffer, tile
from aie.extras.context import mlir_mod_ctx
from aie.iron import Buffer
from aie.iron.device import Tile

line_type = np.ndarray[(4,), np.dtype[np.int32]]


def located(module, names):
    ops = [op for op in module.body.operations if op.operation.name in names]
    assert [op.operation.name for op in ops] == names, ops
    return [
        (op.location.filename, op.location.start_line, op.location.start_col)
        for op in ops
    ]


def check(build):
    with mlir_mod_ctx() as ctx:
        t = tile(0, 2)
        line = build(t)
    names = ["arith.constant", "memref.load", "arith.constant", "memref.store"]
    load, store = located(ctx.module, names)[1::2]
    assert load == (__file__, line, 11), load
    assert store == (__file__, line, 4), store


def dialect(t):
    b = buffer(t, line_type, name="b")
    line = inspect.currentframe().f_lineno + 1
    b[0] = b[1]
    return line


def iron(t):
    placed = Tile(0, 2)
    placed.op = t
    b = Buffer(line_type, name="b", tile=placed)
    b.resolve()
    line = inspect.currentframe().f_lineno + 1
    b[0] = b[1]
    return line


check(dialect)
check(iron)
