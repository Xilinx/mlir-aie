# _pipeline.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""One Worker per stage, fed and drained through fifos: the loop the single-core templates run.

A ``Stage`` is what one core does: acquire its input fifos, acquire its
outputs, call its body, release. ``pipeline`` builds the Workers and
writes the runtime sequence that fills and drains the host buffers.
``transform``, ``for_each``, ``reduce`` and the kernel-validation builder are
each a few lines on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from aie.iron.controlflow import range_
from aie.iron.dataflow import ObjectFifo
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.iron.program import Program
from aie.iron.runtime import Runtime
from aie.iron.worker import Worker
from aie.utils import get_current_device


@dataclass
class Stage:
    """A Worker's loop.

    Every iteration acquires ``count`` objects from each ``(fifo, count)`` in
    ``inputs`` and one from each output fifo, calls ``body(ins, outs, held,
    constants, iteration)`` and releases them. ``held`` fifos are acquired
    once before the loop (a kernel's tensor parameters); ``constants`` are
    the kernels, buffers and callables the Worker is handed, resolved, in
    the same order. With ``outputs_span_iterations`` the outputs are
    acquired once around the loop instead. ``prologue(constants)`` runs
    before the loop and ``initialize(outs, constants)`` right after the
    outputs are acquired.
    """

    body: Callable
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    held: list = field(default_factory=list)
    constants: list = field(default_factory=list)
    iterations: int = 1
    outputs_span_iterations: bool = False
    prologue: Callable | None = None
    initialize: Callable | None = None
    tile: Tile | None = None
    stack_size: int | None = None
    trace: bool = False
    worker: Worker | None = field(default=None, init=False)

    def _core(self):
        ni, no, nh = len(self.inputs), len(self.outputs), len(self.held)
        counts = [n for _, n in self.inputs]
        spans = self.outputs_span_iterations

        def core(*args):
            f_in, f_out = args[:ni], args[ni : ni + no]
            f_held, constants = args[ni + no : ni + no + nh], args[ni + no + nh :]
            if self.prologue is not None:
                self.prologue(constants)
            held = [f.acquire(1) for f in f_held]
            outs: list = []

            def acquire_outputs():
                outs[:] = [f.acquire(1) for f in f_out]
                if self.initialize is not None:
                    self.initialize(outs, constants)

            if spans:
                acquire_outputs()
            loop = (
                range_(self.iterations)
                if self.iterations > 1
                else range(self.iterations)
            )
            for iteration in loop:
                ins = [f.acquire(n) for f, n in zip(f_in, counts)]
                if not spans:
                    acquire_outputs()
                self.body(ins, outs, held, constants, iteration)
                for f, n in zip(f_in, counts):
                    f.release(n)
                if not spans:
                    for f in f_out:
                        f.release(1)
            if spans:
                for f in f_out:
                    f.release(1)
            for f in f_held:
                f.release(1)

        return core

    def build(self) -> Worker:
        self.worker = Worker(
            self._core(),
            [f.cons() for f, _ in self.inputs]
            + [f.prod() for f in self.outputs]
            + [f.cons() for f in self.held]
            + list(self.constants),
            tile=self.tile,
            stack_size=self.stack_size,
            trace=1 if self.trace else 0,
        )
        return self.worker


@dataclass
class KernelParams:
    """A kernel's trailing parameters: scalars inline, tensors through fifos held for the loop."""

    count: int = 0
    scalars: dict = field(default_factory=dict)
    fifos: list = field(default_factory=list)
    types: list = field(default_factory=list)

    def resolve(self, held) -> list:
        """Return the parameter list for one call; ``held`` holds the acquired fifo objects in order."""
        held = iter(held)
        return [
            self.scalars[i] if i in self.scalars else next(held)
            for i in range(self.count)
        ]


def kernel_params(func, params, first: int) -> KernelParams:
    """Sort ``params`` by the kernel's argument types from index ``first`` on.

    A numpy scalar type is passed as an MLIR constant; anything else is a
    tensor (a real one or a descriptor with ``shape`` and ``dtype``) sent
    through its own ``param<i>`` fifo. A callable that is not an
    ``ExternalFunction`` takes no parameters.
    """
    if not isinstance(func, ExternalFunction):
        return KernelParams()
    result = KernelParams(count=len(params))
    for i, (param, arg_type) in enumerate(zip(params, func.arg_types()[first:])):
        if isinstance(arg_type, type) and issubclass(arg_type, np.generic):
            result.scalars[i] = param
        else:
            ty = np.ndarray[param.shape, np.dtype[param.dtype]]
            result.types.append(ty)
            result.fifos.append(ObjectFifo(ty, name=f"param{i}"))
    return result


def pipeline(stages, host_types, transfers, *, trace_size=0):
    """Build the stages' Workers and the sequence that moves their host buffers.

    ``host_types`` are the design's host buffers in argument order and
    ``transfers`` the ``(fifo, "fill" | "drain", host index)`` triples that
    connect them; fills are issued first, then drains, each in the order
    given. A positive ``trace_size`` traces the stages that asked for it.
    """
    device = get_current_device()
    if device is None:
        raise RuntimeError(
            "iron.algorithms requires an active NPU device. Call "
            "iron.set_current_device() or ensure DefaultNPURuntime is initialized first."
        )
    workers = [stage.build() for stage in stages]
    endpoints = [
        (fifo.prod() if kind == "fill" else fifo.cons(), kind, index)
        for fifo, kind, index in transfers
    ]
    n_host = len(host_types)

    def sequence(*args):
        hosts, handles = args[:n_host], args[n_host:]
        for handle, (_, kind, index) in zip(handles, endpoints):
            if kind == "fill":
                handle.fill(hosts[index])
        for handle, (_, kind, index) in zip(handles, endpoints):
            if kind == "drain":
                handle.drain(hosts[index], wait=True)

    rt = Runtime(sequence, list(host_types) + [e for e, _, _ in endpoints])
    prog = Program(device, rt, workers=workers)
    if trace_size > 0:
        prog.enable_trace(trace_size)
    return prog.resolve_program()
