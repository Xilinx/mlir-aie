# test_trace.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""aie.iron.trace: each core body's objectFIFO accesses, recorded in program order as a design is built — no NPU required."""

import os

import numpy as np
from aie.dialects._aie_enum_gen import ObjectFifoLinkMode, ObjectFifoPort
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import else_, forever_, if_, range_
from aie.iron.device import NPU2
from aie.iron.trace import Access, Scope, record

THIS_FILE = os.path.abspath(__file__)
SOURCE = open(THIS_FILE).read().splitlines()
line = np.ndarray[(64,), np.dtype[np.int32]]


def design(while_true: bool = True, forever: bool = False) -> tuple:
    """Return a design, its fifo read by the core through a memtile, its output fifo and its Worker.

    The core holds a window of two objects, releasing one an iteration, and
    releases its output on either side of an ``if_``.
    """
    of_in = ObjectFifo(line, depth=2, name="in")
    of_mid = of_in.cons().forward()
    of_out = ObjectFifo(line, depth=2, name="out")

    def core_fn(a, b):
        if forever:
            for _ in forever_():
                a.acquire(1)
                a.release(1)
            return
        for _ in range_(4):
            a.acquire(2)
            b.acquire(1)
            with if_(True) as branch:
                b.release(1)
            with else_(branch):
                b.release(1)
            a.release(1)
        a.release(1)

    worker = Worker(core_fn, [of_mid.cons(), of_out.prod()], while_true=while_true)

    def sequence(a_in, b_out, in_handle, out_handle):
        in_handle.fill(a_in)
        out_handle.drain(b_out, wait=True)

    rt = Runtime(sequence, [line, line, of_in.prod(), of_out.cons()])
    return Program(NPU2(), rt, workers=[worker]), of_mid, of_out, worker


def test_a_core_body_is_recorded_in_program_order_nested_in_its_control_flow():
    program, of_mid, of_out, worker = design()
    with record() as recording:
        program.resolve_program()
    (core,) = recording.cores
    assert core.kind == "core" and core.owner is worker
    read, write = ObjectFifoPort.Consume, ObjectFifoPort.Produce
    loop, drain = core.body
    assert drain == Access(of_mid, read, "release", 1)
    assert loop.kind == "for" and loop.owner == (0, 4, 1)
    window, slot, then, otherwise, advance = loop.body
    assert window == Access(of_mid, read, "acquire", 2)
    assert slot == Access(of_out, write, "acquire", 1)
    assert then == Scope("if", body=[Access(of_out, write, "release", 1)])
    assert otherwise == Scope("else", body=[Access(of_out, write, "release", 1)])
    assert advance == Access(of_mid, read, "release", 1)


def _bounds_design(core_fn) -> tuple:
    """Return a design whose core body is exactly ``core_fn``, for tests that only check a loop's recorded bounds."""
    worker = Worker(core_fn, fn_args=[], while_true=False)
    program = Program(NPU2(), Runtime(lambda: None), workers=[worker])
    return program, worker


def test_a_fors_keyword_bounds_are_recorded():
    def core_fn():
        for _ in range_(0, 8, step=2):
            pass

    program, _ = _bounds_design(core_fn)
    with record() as recording:
        program.resolve_program()
    (loop,) = recording.cores[0].body
    assert loop.owner == (0, 8, 2)


def test_a_fors_all_keyword_bounds_are_recorded():
    def core_fn():
        for _ in range_(start=0, stop=8, step=2):
            pass

    program, _ = _bounds_design(core_fn)
    with record() as recording:
        program.resolve_program()
    (loop,) = recording.cores[0].body
    assert loop.owner == (0, 8, 2)


def test_a_fors_numpy_integer_bound_is_recorded_as_a_python_int():
    def core_fn():
        for _ in range_(np.int32(4)):
            pass

    program, _ = _bounds_design(core_fn)
    with record() as recording:
        program.resolve_program()
    (loop,) = recording.cores[0].body
    assert loop.owner == (0, 4, 1)
    assert all(isinstance(bound, int) for bound in loop.owner)


def test_a_fors_runtime_bound_is_recorded_as_none():
    def core_fn():
        for i in range_(4):
            for _ in range_(i):
                pass

    program, _ = _bounds_design(core_fn)
    with record() as recording:
        program.resolve_program()
    (outer,) = recording.cores[0].body
    (inner,) = outer.body
    assert outer.owner == (0, 4, 1)
    assert inner.owner == (0, None, 1)


def test_the_workers_own_loop_around_its_body_is_not_the_bodys():
    program, _, _, _ = design(while_true=False)
    with record() as recording:
        program.resolve_program()
    assert [s.kind for s in recording.cores[0].body if isinstance(s, Scope)] == ["for"]


def test_a_forever_the_body_writes_itself_is_recorded():
    program, of_mid, _, _ = design(while_true=False, forever=True)
    with record() as recording:
        program.resolve_program()
    (forever,) = recording.cores[0].body
    read = ObjectFifoPort.Consume
    assert forever == Scope(
        "forever",
        body=[Access(of_mid, read, "acquire", 1), Access(of_mid, read, "release", 1)],
    )


def test_a_link_the_design_draws_while_recorded_is_recorded():
    with record() as recording:
        program, of_mid, _, _ = design()
        program.resolve_program()
    (made,) = recording.links
    assert [d.object_fifo for d in made.dsts] == [of_mid]
    assert made.mode == ObjectFifoLinkMode.Space


def test_nothing_is_recorded_outside_a_record_block_and_the_build_is_unchanged():
    program, _, _, _ = design()
    recorded = None
    with record() as recording:
        recorded = str(program.resolve_program())
    program, _, _, _ = design()
    assert str(program.resolve_program()) == recorded
    assert recording.cores


def test_a_handle_names_the_fifo_it_is_an_end_of():
    of = ObjectFifo(line, depth=2, name="ends")
    assert of.prod().object_fifo is of and of.cons().object_fifo is of


def test_an_access_carries_the_user_statement_that_made_it():
    program, _, _, _ = design()
    with record() as recording:
        program.resolve_program()
    loop = recording.cores[0].body[0]
    window = loop.body[0]
    assert window.site.filename == THIS_FILE
    assert SOURCE[window.site.line - 1].strip() == "a.acquire(2)"
    assert SOURCE[loop.site.line - 1].strip() == "for _ in range_(4):"
    core = recording.cores[0]
    assert SOURCE[core.site.line - 1].strip().startswith("worker = Worker(core_fn")


def test_the_runtime_sequence_s_fills_and_drains_are_recorded():
    with record() as recording:
        program, _, of_out, _ = design()
        program.resolve_program()
    (fill,) = [t for t in recording.transfers if t.kind == "fill"]
    (drain,) = [t for t in recording.transfers if t.kind == "drain"]
    assert drain.fifo is of_out and drain.elements == 64
    assert fill.fifo is not of_out and fill.elements == 64
