# test_jit.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# REQUIRES: ryzen_ai
# RUN: %pytest %s

"""Fold @iron.jit designs with iron.Reconfiguration, single- and multi-design.

What: the IRON front door -- ordinary ``@iron.jit``-decorated Python designs
folded by ``iron.Reconfiguration`` and dispatched via ``pyxrt.runlist``. Covers
the single-design fold (one entrypoint, plus ``main:init`` for ctrlpkt) and the
multi-design fold (two designs, one ELF, each keeping its own output).

How: ``r.add(design, inp, out)`` per design then ``r.compile()`` yields a
``FullElf``; every entrypoint is dispatched in one runlist and each output read
back with a device->host sync. Written generically against ``elf.entrypoints``
and ``elf.init`` so it runs for every delivery method.

Why: this is the path applications use to fold and run reconfigurable designs
from Python, and the single-design fold shape is not covered elsewhere.
"""

from pathlib import Path

import numpy as np
import pytest

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.compile.jit.compilabledesign import NPU_CACHE_HOME

from harness import dispatch_runlist, read_i32

_TILE = 16
_N = 64


def _add_const_program(add_value):
    tile_ty = np.ndarray[(_TILE,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(_N,), np.dtype[np.int32]]
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(a, b):
        for _ in range_(_N // _TILE):
            e = a.acquire(1)
            o = b.acquire(1)
            for i in range_(_TILE):
                o[i] = e[i] + add_value
            a.release(1)
            b.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod()])

    def sequence(inp, out, in_h, out_h):
        in_h.fill(inp)
        out_h.drain(out, wait=True)

    rt = Runtime(sequence, [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


# Distinct entrypoint names -- Reconfiguration rejects a fold with duplicates.
@iron.jit(name="add5", add_value=5)
def _design_add5(inp: In, out: Out, *, add_value: CompileTime[int]):
    return _add_const_program(add_value)


@iron.jit(name="add3", add_value=3)
def _design_add3(inp: In, out: Out, *, add_value: CompileTime[int]):
    return _add_const_program(add_value)


@iron.jit(name="add7", add_value=7)
def _design_add7(inp: In, out: Out, *, add_value: CompileTime[int]):
    return _add_const_program(add_value)


@pytest.mark.parametrize("reconfig_method", ["loadpdi", "write32", "ctrlpkt"])
def test_single_design_fold(reconfig_method):
    """One design folded alone: entrypoints are just its own (plus main:init for
    ctrlpkt), and the single output computes correctly."""
    inp = iron.arange(_N, dtype=np.int32, device="npu")
    out = iron.zeros(_N, dtype=np.int32, device="npu")

    r = iron.Reconfiguration(
        f"jit_single_{reconfig_method}",
        method=reconfig_method,
        output_dir=str(Path(NPU_CACHE_HOME) / f"jit_single_{reconfig_method}"),
    )
    r.add(_design_add5, inp, out)
    elf = r.compile()

    assert "main:add5" in elf.entrypoints
    assert (elf.init == "main:init") == (reconfig_method == "ctrlpkt")

    per_ep = {"main:add5": (inp, out)}
    if elf.init:
        per_ep[elf.init] = ()
    dispatch_runlist(elf, per_ep)

    np.testing.assert_array_equal(read_i32(out), inp.numpy() + 5)


def test_multi_design_fold():
    """Two @iron.jit designs fold into one ELF (ctrlpkt); each output keeps its
    own design's result."""
    inp = iron.arange(_N, dtype=np.int32, device="npu")
    out_a = iron.zeros(_N, dtype=np.int32, device="npu")
    out_b = iron.zeros(_N, dtype=np.int32, device="npu")

    r = iron.Reconfiguration(
        "jit_multi_ctrlpkt",
        method="ctrlpkt",
        output_dir=str(Path(NPU_CACHE_HOME) / "jit_multi_ctrlpkt"),
    )
    r.add(_design_add3, inp, out_a)
    r.add(_design_add7, inp, out_b)
    elf = r.compile()

    per_ep = {"main:add3": (inp, out_a), "main:add7": (inp, out_b)}
    if elf.init:
        per_ep[elf.init] = ()
    dispatch_runlist(elf, per_ep)

    np.testing.assert_array_equal(read_i32(out_a), inp.numpy() + 3)
    np.testing.assert_array_equal(read_i32(out_b), inp.numpy() + 7)
