# vector_compact/vector_compact_trace.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
"""Stream-compaction with AIE hardware event tracing — IRON + ``@iron.jit``.

This is the trace-enabled twin of ``vector_compact.py``.  It runs the *same*
``bf16_vector_compact`` (butterfly / prefix-scan) and ``bf16_scalar_compact``
(scalar baseline) kernels, but wires the AIE core's hardware trace unit through
``rt.enable_trace()`` so the ``event0()`` / ``event1()`` window bracketing each
kernel body can be read back as cycle counts.

The trace plumbing mirrors ``basic/event_trace/aie_trace.py`` as closely as the
two-pointer compaction signature allows:

  * one ``Worker`` with ``trace=1`` (the compute core),
  * ``rt.enable_trace(trace_size=..., workers=[worker], coretile_events=[...])``
    with the same event IDs (INSTR_EVENT_0 / INSTR_EVENT_1 are the two markers
    the kernel emits via ``event0()`` / ``event1()``; the rest are the standard
    vector / stall / DMA-port events the sibling example traces).

Because the left-pack carries a running write offset across tiles, a single AIE
core processes the whole N=1024 tensor in one call (same as the non-trace
design).  The butterfly and scalar kernels are *separate designs* (one external
symbol each) so each gets its own clean event0->event1 window in its own trace
buffer; ``run_trace.py`` runs both and combines the results.

  * standalone (butterfly):  ``python3 vector_compact_trace.py``
  * compile-only:            ``... --xclbin-path=PATH --insts-path=PATH``
  * scalar variant:          import ``build_trace_design('bf16_scalar_compact')``
"""

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.trace.events import (
    CoreEvent,
    PortEvent,
    WireBundle,
)

_KERNEL_SRC = str(Path(__file__).parent / "vector_compact_kernel.cc")
_N = 1024  # fixed for this version (matches N_ELEMS default in the kernel)
_TRACE_SIZE = 8192  # bytes; same default as basic/event_trace


def build_trace_design(kernel_name, trace_size=_TRACE_SIZE):
    """Return a trace-enabled @iron.jit design calling the named C symbol.

    ``kernel_name`` is one of ``bf16_vector_compact`` (butterfly) or
    ``bf16_scalar_compact`` (scalar baseline).  The compute is identical to
    ``vector_compact.py``; the only addition is ``trace=1`` on the worker plus
    an ``rt.enable_trace()`` block configured exactly like the sibling
    ``basic/event_trace/aie_trace.py`` example.
    """

    @iron.jit
    def design(a: In, c: Out):
        N = _N
        tensor_ty = np.ndarray[(N,), np.dtype[bfloat16]]

        compact_fn = ExternalFunction(
            kernel_name,
            source_file=_KERNEL_SRC,
            arg_types=[tensor_ty, tensor_ty],
            compile_flags=[f"-DN_ELEMS={N}", "-DTILE=32"],
        )

        in_fifo = ObjectFifo(tensor_ty, name="in_fifo")
        out_fifo = ObjectFifo(tensor_ty, name="out_fifo")

        def core_fn(in_fifo, out_fifo, compact_fn):
            for _ in range_(1):
                elem_in = in_fifo.acquire(1)
                elem_out = out_fifo.acquire(1)
                compact_fn(elem_in, elem_out)
                in_fifo.release(1)
                out_fifo.release(1)

        # trace=1 turns this worker's core tile into a trace producer.
        worker = Worker(
            core_fn,
            fn_args=[in_fifo.cons(), out_fifo.prod(), compact_fn],
            trace=1,
        )

        rt = Runtime()
        with rt.sequence(tensor_ty, tensor_ty) as (a_in, c_out):
            # Same per-tile-class event list as basic/event_trace/aie_trace.py.
            # INSTR_EVENT_0 / INSTR_EVENT_1 are the markers the kernel emits via
            # event0() / event1(); the cycle delta between them is the
            # region-of-interest cost we report.
            rt.enable_trace(
                trace_size=trace_size,
                workers=[worker],
                coretile_events=[
                    CoreEvent.INSTR_EVENT_0,
                    CoreEvent.INSTR_EVENT_1,
                    CoreEvent.INSTR_VECTOR,
                    CoreEvent.MEMORY_STALL,
                    CoreEvent.STREAM_STALL,
                    CoreEvent.LOCK_STALL,
                    PortEvent(CoreEvent.PORT_RUNNING_0, WireBundle.DMA, 0, True),
                    PortEvent(CoreEvent.PORT_RUNNING_1, WireBundle.DMA, 0, False),
                ],
            )
            rt.start(worker)
            rt.fill(in_fifo.prod(), a_in)
            rt.drain(out_fifo.cons(), c_out, wait=True)

        return Program(iron.get_current_device(), rt).resolve_program()

    return design


# The butterfly (vector) kernel is the default design used by run_design_cli and
# the compile-only Makefile path.  run_trace.py builds the scalar one on demand
# via build_trace_design("bf16_scalar_compact").
vector_compact_trace = build_trace_design("bf16_vector_compact")


def _reference(x_np):
    """numpy reference: left-pack survivors (x >= 0), zero-pad the rest."""
    survivors = x_np[x_np >= 0.0]
    k = len(survivors)
    ref = np.zeros(_N, dtype=bfloat16)
    ref[:k] = survivors
    return ref, k


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Vector Stream Compaction (trace)")
    add_compile_args(p)
    p.add_argument("--trace-size", type=int, default=_TRACE_SIZE)
    return p


def _compile_kwargs(opts):
    # trace_size is a CompileTime knob inside enable_trace's design closure; the
    # design captures _TRACE_SIZE directly, so nothing extra is needed here.
    return dict()


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    x_np = rng.uniform(-1.0, 1.0, size=(_N,)).astype(bfloat16)
    ref, k = _reference(x_np)
    print(f"survivor count (x >= 0): {k} / {_N}")

    a_t = iron.tensor(x_np, dtype=bfloat16, device="npu")
    c_t = iron.zeros(_N, dtype=bfloat16, device="npu")
    vector_compact_trace(a_t, c_t)
    out = np.array(c_t.numpy(), copy=True)

    ok = np.array_equal(out[:k], ref[:k]) and bool(
        np.all(out[k:].astype(np.float32) == 0.0)
    )
    print("PASS!" if ok else "FAIL!")


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        vector_compact_trace,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
