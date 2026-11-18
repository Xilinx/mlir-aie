# test_trace.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""aie.iron.trace: each core body's objectFIFO accesses, recorded in program order as a design is built — no NPU required."""

import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import else_, forever_, if_, range_
from aie.iron.device import NPU2
from aie.iron.trace import Access, Block, record

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
    with record() as recorder:
        program.resolve_program()
    (core,) = recorder.cores
    assert core.kind == "core" and core.owner is worker
    loop, drain = core.body
    assert drain == Access(of_mid, False, False, 1)
    assert loop.kind == "for" and loop.owner == (4,)
    window, slot, then, otherwise, advance = loop.body
    assert window == Access(of_mid, False, True, 2)
    assert slot == Access(of_out, True, True, 1)
    assert then == Block("if", then.owner, [Access(of_out, True, False, 1)])
    assert otherwise == Block("else", None, [Access(of_out, True, False, 1)])
    assert advance == Access(of_mid, False, False, 1)


def test_the_workers_own_loop_around_its_body_is_not_the_bodys():
    program, _, _, _ = design(while_true=False)
    with record() as recorder:
        program.resolve_program()
    assert [b.kind for b in recorder.cores[0].body if isinstance(b, Block)] == ["for"]


def test_a_forever_the_body_writes_itself_is_recorded():
    program, of_mid, _, _ = design(while_true=False, forever=True)
    with record() as recorder:
        program.resolve_program()
    (forever,) = recorder.cores[0].body
    assert forever == Block(
        "forever",
        None,
        [Access(of_mid, False, True, 1), Access(of_mid, False, False, 1)],
    )


def test_a_link_the_design_draws_while_recorded_is_recorded():
    with record() as recorder:
        program, of_mid, _, _ = design()
        program.resolve_program()
    (made,) = recorder.links
    assert made.dsts == (of_mid,) and made.mode == "space"


def test_nothing_is_recorded_outside_a_record_block_and_the_build_is_unchanged():
    program, _, _, _ = design()
    recorded = None
    with record() as recorder:
        recorded = str(program.resolve_program())
    program, _, _, _ = design()
    assert str(program.resolve_program()) == recorded
    assert recorder.cores
