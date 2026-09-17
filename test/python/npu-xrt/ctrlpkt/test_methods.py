# test_methods.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# REQUIRES: ryzen_ai
# RUN: %pytest %s

"""Reconfiguration delivery methods dispatched via ``pyxrt.runlist``.

What: two named designs, each adding a distinct constant, fold into one full ELF
via ``iron.Reconfiguration`` and run through a single batched ``pyxrt.runlist``;
each design's own output buffer must carry its own result. Covered for all three
delivery methods (``loadpdi``, ``write32``, ``ctrlpkt``) and, for ctrlpkt, with
per-design external C++ kernels.

How: ``aiecc --get-full-elf --reconfig-method=M`` folds the designs; the ELF
exposes one entrypoint per design (plus ``main:init`` for the ctrlpkt overlay).
``dispatch_runlist`` submits every entrypoint in one atomic ordered runlist, then
``read_i32`` reads each output back with an explicit device->host sync.

Why: the fold + dispatch path is the toolchain's product; this exercises all three
methods on real hardware so the delivery selector is proven end to end.
"""

from pathlib import Path

import numpy as np
import pytest

import aie.iron as iron
from aie.iron import ExternalFunction, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.compile.jit.compilabledesign import NPU_CACHE_HOME
from aie.utils.compile.jit.markers import CompileTime

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


@iron.jit(name="add_a", add_value=5)
def _design_a(inp: In, out: Out, *, add_value: CompileTime[int]):
    return _add_const_program(add_value)


@iron.jit(name="add_b", add_value=9)
def _design_b(inp: In, out: Out, *, add_value: CompileTime[int]):
    return _add_const_program(add_value)


@pytest.mark.parametrize("reconfig_method", ["loadpdi", "write32", "ctrlpkt"])
def test_runlist_two_designs(reconfig_method):
    """Two designs fold into one ELF; a runlist dispatches them and each output
    buffer holds its own design's result. All three delivery methods batch."""
    inp = iron.arange(_N, dtype=np.int32, device="npu")
    out_a = iron.zeros(_N, dtype=np.int32, device="npu")
    out_b = iron.zeros(_N, dtype=np.int32, device="npu")

    r = iron.Reconfiguration(
        f"runlist_{reconfig_method}",
        method=reconfig_method,
        output_dir=str(Path(NPU_CACHE_HOME) / f"recfg_{reconfig_method}"),
    )
    r.add(_design_a, inp, out_a)
    r.add(_design_b, inp, out_b)
    elf = r.compile()

    dispatch_runlist(
        elf,
        {"main:init": (), "main:add_a": (inp, out_a), "main:add_b": (inp, out_b)},
    )

    np.testing.assert_array_equal(read_i32(out_a), inp.numpy() + 5)
    np.testing.assert_array_equal(read_i32(out_b), inp.numpy() + 9)


_EXT_KERNEL_SRC_TMPL = """extern "C" {{
    void {sym}(int* input, int* output, int tile_size) {{
        for (int i = 0; i < tile_size; i++) {{
            output[i] = input[i] + {delta};
        }}
    }}
}}"""


def _add_ext_program(func, suffix):
    tile_ty = np.ndarray[(_TILE,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(_N,), np.dtype[np.int32]]
    of_in = ObjectFifo(tile_ty, name=f"ein{suffix}")
    of_out = ObjectFifo(tile_ty, name=f"eout{suffix}")

    def core_body(a, b, func_to_apply):
        for _ in range_(_N // _TILE):
            e = a.acquire(1)
            o = b.acquire(1)
            func_to_apply(e, o, _TILE)
            a.release(1)
            b.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), func])

    def sequence(inp, out, in_h, out_h):
        in_h.fill(inp)
        out_h.drain(out, wait=True)

    rt = Runtime(sequence, [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


_func_ext_c = ExternalFunction(
    "add_c_ext",
    source_string=_EXT_KERNEL_SRC_TMPL.format(sym="add_c_ext", delta=3),
    arg_types=[
        np.ndarray[(_TILE,), np.dtype[np.int32]],
        np.ndarray[(_TILE,), np.dtype[np.int32]],
        np.int32,
    ],
)

_func_ext_d = ExternalFunction(
    "add_d_ext",
    source_string=_EXT_KERNEL_SRC_TMPL.format(sym="add_d_ext", delta=7),
    arg_types=[
        np.ndarray[(_TILE,), np.dtype[np.int32]],
        np.ndarray[(_TILE,), np.dtype[np.int32]],
        np.int32,
    ],
)


@iron.jit(name="ext_design_c", func=_func_ext_c)
def _design_ext_c(inp: In, out: Out, *, func: CompileTime[object]):
    return _add_ext_program(func, suffix="_c")


@iron.jit(name="ext_design_d", func=_func_ext_d)
def _design_ext_d(inp: In, out: Out, *, func: CompileTime[object]):
    return _add_ext_program(func, suffix="_d")


def test_runlist_two_external_designs():
    """Two designs, each with its own distinct C++ external kernel, fold together
    and dispatch via one runlist (ctrlpkt), each computing its own result."""
    inp = iron.arange(_N, dtype=np.int32, device="npu")
    out_c = iron.zeros(_N, dtype=np.int32, device="npu")
    out_d = iron.zeros(_N, dtype=np.int32, device="npu")

    r = iron.Reconfiguration(
        "runlist_ext_ctrlpkt",
        method="ctrlpkt",
        output_dir=str(Path(NPU_CACHE_HOME) / "recfg_ext_ctrlpkt"),
    )
    r.add(_design_ext_c, inp, out_c)
    r.add(_design_ext_d, inp, out_d)
    elf = r.compile()

    dispatch_runlist(
        elf,
        {
            "main:init": (),
            "main:ext_design_c": (inp, out_c),
            "main:ext_design_d": (inp, out_d),
        },
    )

    np.testing.assert_array_equal(read_i32(out_c), inp.numpy() + 3)
    np.testing.assert_array_equal(read_i32(out_d), inp.numpy() + 7)
