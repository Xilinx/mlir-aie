# test_reconfig_runlist.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# REQUIRES: ryzen_ai
# RUN: %pytest %s

"""Reference harness: dispatch a folded multi-design full ELF via ``pyxrt.runlist``.

The mlir-aie toolchain's job ends at the fold: ``aiecc --get-full-elf
--reconfig-method=M`` compiles several designs into ONE full ELF whose entrypoints
(``main:<design>``, plus ``main:init`` for ctrlpkt) each reconfigure the array and
compute. *Running* an ordered sequence of those entrypoints is the application's
job, and the primitive already exists: ``pyxrt.runlist`` -- a single-context
batched submit that executes its runs atomically in order. There is no library
dispatch API; this test IS the harness, using pyxrt directly.

Two named designs each add a distinct constant, so each design's own output buffer
must carry that design's result after the chain runs. Because the harness chose the
method, it knows to prepend the ``main:init`` overlay-standup run and give every
kernel an inert control-packet slot for ctrlpkt -- the runlist itself is
method-blind.

Device finding (2026-09-07): ALL THREE methods -- loadpdi, write32, ctrlpkt --
batch correctly in one ``pyxrt.runlist`` (each entry's distinct config computes its
own result in a single batched submit). An earlier probe wrongly concluded ctrlpkt
could not batch; that was a readback artifact, not a device limit: dispatching via a
raw ``pyxrt.runlist`` bypasses the iron runtime's post-dispatch device-dirty
marking, so an iron tensor's lazy ``.to("cpu")`` returned stale host zeros for an
output the runlist had actually written correctly. The fix is to read each output
from its buffer object with an explicit device->host sync (below), which is the
honest pattern anyway when the application dispatches with raw pyxrt.
"""

from pathlib import Path

import numpy as np
import pytest
import pyxrt  # pyright: ignore[reportMissingImports]

import aie.iron as iron
from aie.iron import ExternalFunction, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.compile.jit.compilabledesign import NPU_CACHE_HOME
from aie.utils.compile.jit.markers import CompileTime
from aie.utils.hostruntime.xrtruntime.device import acquire_device

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
    """Two named designs fold into one ELF; a pyxrt.runlist dispatches them in
    order and each design's own output buffer holds its own result."""
    inp = iron.arange(_N, dtype=np.int32, device="npu")
    out_a = iron.zeros(_N, dtype=np.int32, device="npu")
    out_b = iron.zeros(_N, dtype=np.int32, device="npu")

    # Toolchain: fold the two idiomatic designs into one full ELF.
    r = iron.Reconfiguration(
        f"runlist_{reconfig_method}",
        method=reconfig_method,
        output_dir=str(Path(NPU_CACHE_HOME) / f"recfg_{reconfig_method}"),
    )
    r.add(_design_a, inp, out_a)
    r.add(_design_b, inp, out_b)
    elf = r.compile()

    # Harness: dispatch the ELF's entrypoints via a raw pyxrt.runlist.
    dev = acquire_device()
    ctx = pyxrt.hw_context(dev, pyxrt.elf(str(elf.path)))

    # ctrlpkt stands up a resident overlay once via main:init, and every kernel
    # carries an extra (inert) control-packet slot. The harness knows this from
    # the descriptor (``elf.needs_ctrl_bo``); the runlist itself is method-blind.
    dummy = (
        iron.zeros(1024, dtype=np.int32, device="npu") if elf.needs_ctrl_bo else None
    )

    def _bos(*tensors):
        bos = [t.buffer_object() for t in tensors]
        if elf.needs_ctrl_bo:
            bos.append(dummy.buffer_object())
        return bos

    # elf.entrypoints is init-first when present; map each to its buffers.
    # main:init has a fixed, overlay-defined signature -- only the (inert)
    # control-packet buffer, no design tensors -- so its own entry is empty
    # and _bos() below supplies just the dummy ctrl BO.
    per_ep = {
        "main:init": (),
        "main:add_a": (inp, out_a),
        "main:add_b": (inp, out_b),
    }

    # Batch every entry into ONE pyxrt.runlist submit -- correct for all three
    # methods (loadpdi / write32 / ctrlpkt).
    runlist = pyxrt.runlist(ctx)
    keep = []  # keep kernels + runs alive until wait() returns
    for name in elf.entrypoints:
        kernel = pyxrt.ext.kernel(ctx, name)
        run = pyxrt.run(kernel)
        for i, bo in enumerate(_bos(*per_ep[name])):
            run.set_arg(i, bo)
        runlist.add(run)  # NOT run.start() -- UB for a run inside a runlist
        keep.append((kernel, run))
    runlist.execute()
    runlist.wait()

    # Read each output from its buffer object with an explicit device->host sync.
    # Raw-pyxrt dispatch bypasses the iron runtime's device-dirty marking, so the
    # tensor's lazy .to("cpu") would return stale host bytes; sync the BO directly.
    def _read(t):
        bo = t.buffer_object()
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return np.frombuffer(bo.map(), dtype=np.int32)

    np.testing.assert_array_equal(_read(out_a), inp.numpy() + 5)
    np.testing.assert_array_equal(_read(out_b), inp.numpy() + 9)

    # Release the context before the next parametrization to avoid churn.
    del runlist, keep, ctx


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
    """Multi-external device proof: two designs, EACH with its own distinct C++
    external kernel (different add constants), fold via ``Reconfiguration`` and
    dispatch via the same ``pyxrt.runlist`` pattern as ``test_runlist_two_designs``.

    ``test_two_external_designs_fold`` in ``test/python/reconfiguration.py`` proves
    the multi-external fold offline (ELF exists, both kernel objects staged); this
    proves the folded ELF actually computes correctly on real NPU hardware.

    Method: ctrlpkt. This is the richest path (it prepends ``main:init`` to stand
    up a resident overlay and adds a control-packet slot to every kernel), so it
    is the strongest available device proof that per-design external kernels
    survive the flat multi-external build and the ctrlpkt fold/dispatch
    machinery together.
    """
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

    dev = acquire_device()
    ctx = pyxrt.hw_context(dev, pyxrt.elf(str(elf.path)))

    dummy = (
        iron.zeros(1024, dtype=np.int32, device="npu") if elf.needs_ctrl_bo else None
    )

    def _bos(*tensors):
        bos = [t.buffer_object() for t in tensors]
        if elf.needs_ctrl_bo:
            bos.append(dummy.buffer_object())
        return bos

    # main:init has a fixed, overlay-defined signature -- only the (inert)
    # control-packet buffer, no design tensors.
    per_ep = {
        "main:init": (),
        "main:ext_design_c": (inp, out_c),
        "main:ext_design_d": (inp, out_d),
    }

    runlist = pyxrt.runlist(ctx)
    keep = []
    for name in elf.entrypoints:
        kernel = pyxrt.ext.kernel(ctx, name)
        run = pyxrt.run(kernel)
        for i, bo in enumerate(_bos(*per_ep[name])):
            run.set_arg(i, bo)
        runlist.add(run)
        keep.append((kernel, run))
    runlist.execute()
    runlist.wait()

    def _read(t):
        bo = t.buffer_object()
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return np.frombuffer(bo.map(), dtype=np.int32)

    np.testing.assert_array_equal(_read(out_c), inp.numpy() + 3)
    np.testing.assert_array_equal(_read(out_d), inp.numpy() + 7)

    del runlist, keep, ctx
