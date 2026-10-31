# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

import os
import tempfile
from functools import partial

from aie.extras import types as T
from aie.extras.dialects.arith import constant
from aie.helpers.sourceloc import traced_body
from aie.ir import CallSiteLoc, Context, InsertionPoint, Location, Module
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
    loops = [op for op in walk(core) if op.name == "scf.while"]
    assert len(loops) == 2
    for op in walk(core):
        # A synthesized body takes the Worker's site, named after the Worker
        # (here, after the variable it is assigned to).
        assert op.location.name_str == "iron:worker", (op.name, op.location)
        loc = op.location.child_loc
        # The statement is the callee of the calls that led to it.
        if isinstance(loc, CallSiteLoc):
            loc = loc.callee
        assert loc.filename == __file__, (op.name, loc)
        assert "Worker(None)" in SOURCE[loc.start_line - 1], (op.name, loc)


def failing_body():
    constant(7, T.i32())
    raise RuntimeError("body failed")


def check_failing_body():
    """Ops a body built before it raised are still marked as Python's."""
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            try:
                with traced_body(failing_body):
                    failing_body()
            except RuntimeError:
                pass
            else:
                raise AssertionError("expected the body to raise")
        (op,) = module.body.operations
        assert op.location.name_str == "py:failing_body", op.location


def check_relative_body():
    """A body compiled under a relative name is located by its absolute path."""
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "rel_body.py"), "w") as f:
            f.write("def rel_body():\n    constant(9, T.i32())\n")
        os.chdir(tmp)
        try:
            expected = os.path.abspath("rel_body.py")
            with open("rel_body.py") as f:
                code = compile(f.read(), "rel_body.py", "exec")
            design = {"constant": constant, "T": T}
            exec(code, design)
            with Context(), Location.unknown():
                module = Module.create()
                with InsertionPoint(module.body):
                    with traced_body(design["rel_body"]):
                        design["rel_body"]()
                (op,) = module.body.operations
                filename = op.location.child_loc.filename
        finally:
            os.chdir(cwd)
    assert filename == expected, filename


check_callable_bodies()
check_synthesized_body()
check_failing_body()
check_relative_body()
print("PASS: callable, synthesized, failing and relative body locations")
