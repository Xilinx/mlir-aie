# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

from functools import partial

from aie.extras import types as T
from aie.extras.dialects.arith import constant
from aie.iron import Program, Runtime, Worker
from aie.iron.device import NPU1Col1

SOURCE = open(__file__).read().splitlines()


def walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from walk(child.operation)


def emit(value):
    constant(value[0], T.i32())


class Emitter:
    def __call__(self, value):
        emit(value)


def check_callable_bodies():
    for body in (partial(emit), Emitter()):
        worker = Worker(body, [[41]], while_true=False)
        runtime = Runtime(body, [[43]])
        module = Program(NPU1Col1(), runtime, workers=[worker]).resolve_program()
        found = set()
        for op in walk(module.operation):
            if op.name == "arith.constant":
                value = op.attributes["value"].value
                if value in (41, 43):
                    loc = op.location.child_loc
                    assert loc.filename == __file__, loc
                    assert "constant(value[0]" in SOURCE[loc.start_line - 1], loc
                    found.add(value)
        assert found == {41, 43}, found


def check_synthesized_body():
    worker = Worker(None)
    module = Program(
        NPU1Col1(), Runtime(lambda: None), workers=[worker]
    ).resolve_program()
    core = next(op for op in walk(module.operation) if op.name == "aie.core")
    loops = [op for op in walk(core) if op.name == "scf.for"]
    assert len(loops) == 2
    for op in walk(core):
        loc = op.location
        assert loc.filename == __file__, (op.name, loc)
        assert "Worker(None)" in SOURCE[loc.start_line - 1], (op.name, loc)


check_callable_bodies()
check_synthesized_body()
print("PASS: callable and synthesized body locations")
