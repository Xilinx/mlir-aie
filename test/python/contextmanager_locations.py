# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

# An op built by one of the package's @contextmanagers takes the user's `with`
# statement, not the contextlib frame that enters the manager, whether it is
# located by its own builder or by a traced Worker body.

import contextlib
import os

from aie.extras import types as T
from aie.extras.dialects.arith import constant
from aie.helpers.dialects.scf import if_
from aie.ir import CallSiteLoc, Context, InsertionPoint, Location, Module, NameLoc
from aie.iron import Program, Runtime, Worker
from aie.iron.device import NPU1Col1

SOURCE = open(__file__).read().splitlines()


def walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from walk(child.operation)


def located_here(op, statement):
    loc = op.location
    if isinstance(loc, NameLoc):
        loc = loc.child_loc
    # The statement is the callee; its callers follow.
    if isinstance(loc, CallSiteLoc):
        loc = loc.callee
    assert loc.filename == __file__, (op.name, loc)
    assert statement in SOURCE[loc.start_line - 1], (op.name, loc)


def check_builder():
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            cond = constant(1, T.bool())
            with if_(cond, hasElse=False):
                pass
        if_op = next(op for op in walk(module.operation) if op.name == "scf.if")
        located_here(if_op, "with if_(cond, hasElse=False)")


def body():
    cond = constant(1, T.bool())
    with if_(cond, hasElse=False):
        constant(7, T.i32())


def check_traced_body():
    worker = Worker(body, [], while_true=False)
    module = Program(
        NPU1Col1(), Runtime(lambda: None), workers=[worker]
    ).resolve_program()
    ops = list(walk(module.operation))
    located_here(next(op for op in ops if op.name == "scf.if"), "with if_(cond")
    for op in ops:
        loc = str(op.location)
        assert os.path.basename(contextlib.__file__) not in loc, (op.name, loc)


check_builder()
check_traced_body()
print("PASS: contextmanager locations")
