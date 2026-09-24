# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s

import inspect
from functools import partial

from aie.extras import types as T
from aie.extras.dialects.arith import constant
from aie.helpers.astloc import with_statement_locations
from aie.helpers.sourceloc import current_body
from aie.ir import Context, InsertionPoint, Location, Module
from aie.iron import Program, Runtime, Worker
from aie.iron.device import NPU1Col1


def walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from walk(child.operation)


def check_callable_bodies():
    observed = []

    def emit(value):
        observed.append(current_body()[0])
        constant(value[0], T.i32())

    class Emitter:
        def __call__(self, value):
            emit(value)

    for body, name in ((partial(emit), "partial"), (Emitter(), "Emitter")):
        observed.clear()
        worker = Worker(body, [[41]], while_true=False)
        runtime = Runtime(body, [[43]])
        assert f"Worker({name} on " in repr(worker)
        module = Program(NPU1Col1(), runtime, workers=[worker]).resolve_program()
        assert sorted(observed) == sorted(["sequence", name]), observed
        with module.context:
            expected = {
                41: worker._source_site.to_location(name),
                43: runtime._source_site.to_location(),
            }
            found = {}
            for op in walk(module.operation):
                if op.name == "arith.constant":
                    value = op.attributes["value"].value
                    if value in expected:
                        assert op.location == expected[value], op.location
                        found[value] = op.location
            assert found.keys() == expected.keys(), found


def check_synthesized_body():
    worker = Worker(None)
    assert with_statement_locations(worker.core_fn) is worker.core_fn
    module = Program(
        NPU1Col1(), Runtime(lambda: None), workers=[worker]
    ).resolve_program()
    core = next(op for op in walk(module.operation) if op.name == "aie.core")
    with module.context:
        expected = worker._source_site.to_location()
        named = worker._source_site.to_location(worker.core_fn.__name__)
        loops = [op for op in walk(core) if op.name == "scf.for"]
        assert len(loops) == 2
        for op in walk(core):
            assert op.location in (expected, named), (op.name, op.location)


def match_body(value):
    match value:
        case 0:
            constant(101, T.i32())
            constant(102, T.i32())
        case n if n > 0:
            for _ in range(2):
                match n:
                    case 1:
                        constant(103, T.i32())
                    case _:
                        constant(104, T.i32())
        case _:
            constant(105, T.i32())
    constant(106, T.i32())


def check_match_locations():
    traced = with_statement_locations(match_body)
    assert traced is not match_body
    source, first_line = inspect.getsourcelines(match_body)
    expected_lines = {
        value: first_line
        + next(i for i, line in enumerate(source) if f"constant({value}," in line)
        for value in range(101, 107)
    }
    with Context(), Location.unknown():
        module = Module.create()
        ambient = Location.file("ambient.py", 1, 0)
        with InsertionPoint(module.body), ambient:
            for value in (0, 1, 2, -1):
                traced(value)
                assert Location.current == ambient
        values = []
        for view in module.body.operations:
            op = view.operation
            value = op.attributes["value"].value
            values.append(value)
            loc = op.location
            assert loc.filename == __file__, loc
            assert loc.start_line == expected_lines[value], (value, loc)
        assert values == [101, 102, 106, 103, 103, 106, 104, 104, 106, 105, 106]


check_callable_bodies()
check_synthesized_body()
check_match_locations()
print("PASS: callable bodies, synthesized cores, and match statement locations")
