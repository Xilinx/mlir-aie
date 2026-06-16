# runtime.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc.
"""Runtime: orchestrates host-side data movement and worker execution for an IRON program.

A :class:`Runtime` owns a single runtime *sequence* — the host-side program that
streams data into and out of the AIE array and starts the Workers. The sequence
body is supplied as a plain callable, mirroring how a :class:`Worker` takes a
``core_fn``::

    rt = Runtime()

    def sequence(A, B, C):
        of_in.prod().fill(A)
        of_out.cons().drain(C, wait=True)

    rt.sequence(sequence, [A_ty, B_ty, C_ty])

The body runs *inside* the ``aie.runtime_sequence`` builder, so ordinary Python
control flow (``range_``/``if_`` from :mod:`aie.iron.controlflow`) and arithmetic
on the scalar arguments lower to ``scf``/``arith`` ops, enabling a single compiled
design to serve many runtime shapes. Data-movement verbs live on the objects they
act on — :meth:`ObjectFifoHandle.fill`/:meth:`drain`, :meth:`Buffer.write` — and
completion is grouped with :class:`TaskGroup`.
"""

from __future__ import annotations
import bisect
from collections import defaultdict, deque
import itertools
import logging
import numpy as np
from typing import Callable

logger = logging.getLogger(__name__)

from ...utils import trace as trace_utils

from ... import ir  # type: ignore

from ...dialects.aiex import runtime_sequence, sync_scratchpad_parameters_from_host
from ..dataflow import ObjectFifoHandle
from ..dataflow.objectfifo import ObjectFifoLink
from ..resolvable import Resolvable
from ..scratchpad_parameter import ScratchpadParameter
from ..worker import Worker
from .data import RuntimeData, RuntimeScalar
from ._context import active_sequence, active_sequence_scope


class _RuntimeBdIdAllocator:
    """Simple runtime-sequence BD allocator for direct npu.dma_memcpy_nd emission."""

    def __init__(self, max_bd_ids_per_key: int = 16):
        self._max_bd_ids_per_key = max_bd_ids_per_key
        self._next_id = defaultdict(int)
        self._free_ids = defaultdict(deque)

    def allocate(self, key) -> int:
        if self._free_ids[key]:
            return self._free_ids[key].popleft()
        bd_id = self._next_id[key]
        if bd_id >= self._max_bd_ids_per_key:
            raise ValueError(
                f"Runtime BD allocator exhausted available IDs for key {key}."
            )
        self._next_id[key] += 1
        return bd_id

    def free(self, key, bd_id: int) -> None:
        free_ids = self._free_ids[key]
        if bd_id in free_ids:
            return
        bisect.insort(free_ids, bd_id)


class IronRuntimeError(Exception):
    """Raised by the IRON Runtime when resolution encounters an unrecoverable state."""


class ActiveSequence:
    """Per-build state for the runtime sequence currently being emitted.

    Created by :meth:`Runtime.resolve` for the duration of the sequence body and
    published via :mod:`aie.iron.runtime._context` so the data-movement verbs and
    :class:`TaskGroup` can reach it without holding a reference to the
    :class:`Runtime`.

    The sequence body runs exactly once. Data-movement verbs reference their
    ObjectFifo by symbol name (a forward reference MLIR resolves later), so DMA
    ops are emitted in program order without the fifo op needing to exist yet.
    After the body, :meth:`resolve_touched_fifos` creates the ``aie.objectfifo``
    ops for fifos the runtime touched, once all their endpoints are known.

    Responsibilities: BD-id allocation, the implicit default :class:`TaskGroup`,
    tracking runtime-touched fifos, and end-of-body finalization.
    """

    def __init__(self, runtime: "Runtime", device, device_ip: ir.InsertionPoint):
        self._runtime = runtime
        self._device = device
        self._device_ip = device_ip
        self._bd_id_allocator = _RuntimeBdIdAllocator()
        self._task_group_index = itertools.count()
        self._task_groups = []
        self._default_group = None  # created lazily on first ungrouped transfer
        self._touched_fifos = []

    def resolve_touched_fifos(self):
        """Create the device-scope ops for every fifo the runtime touched.

        Runs after the body, once all runtime endpoints are established. Worker-
        reachable fifos are already resolved by the Program; these are the
        runtime-facing ones (and their split/join/forward link peers). The
        underlying ops are idempotent, so overlap with the Program is harmless.
        """
        with self._device_ip:
            for of in self._touched_fifos:
                self._dedup_runtime_consumers(of)
            for of in self._touched_fifos:
                self._resolve_fifo(of)

    def _resolve_fifo(self, of) -> None:
        """Create a fifo's op (and its link group's tiles) if not already done."""
        if getattr(of, "_op", None) is not None:
            return
        for grp_of in self._link_group(of):
            for tile in grp_of.tiles():
                self._device.resolve_tile(tile)
            if grp_of._delegate_tile is not None:
                self._device.resolve_tile(grp_of._delegate_tile)
        of.resolve()

    @staticmethod
    def _link_group(of):
        """All fifos reachable from ``of`` through split/join/forward links."""
        seen = {}
        frontier = [of]
        while frontier:
            cur = frontier.pop()
            if id(cur) in seen:
                continue
            seen[id(cur)] = cur
            handles = ([cur._prod] if cur._prod else []) + list(cur._cons)
            for h in handles:
                ep = h.endpoint
                if isinstance(ep, ObjectFifoLink):
                    for peer in (*ep._srcs, *ep._dsts):
                        frontier.append(peer._object_fifo)
        return list(seen.values())

    @staticmethod
    def _dedup_runtime_consumers(of) -> None:
        """Collapse the repeated runtime consumer handles a body creates.

        Calling ``of.cons()`` returns a fresh handle each time, so a body that
        drains the same fifo in a loop accumulates identical runtime consumers.
        Keep one; drop the rest (matching the historical runtime behavior). A
        genuinely different RuntimeEndpoint on the same fifo is an error.
        """
        from .endpoint import RuntimeEndpoint

        runtime_cons = None
        to_remove = []
        for c in of._cons:
            if isinstance(c.endpoint, RuntimeEndpoint):
                if runtime_cons is None:
                    runtime_cons = c
                elif (
                    c.depth == runtime_cons.depth
                    and c.dims_from_stream == runtime_cons.dims_from_stream
                ):
                    to_remove.append(c)
                else:
                    raise IronRuntimeError(
                        f"Found two different RuntimeEndpoints for consumers of "
                        f"the same ObjectFifo: {of}"
                    )
        for r in to_remove:
            of._cons.remove(r)

    # -- task-group bookkeeping (called by TaskGroup) --------------------------

    def next_task_group_id(self) -> int:
        return next(self._task_group_index)

    def register_task_group(self, tg) -> None:
        self._task_groups.append(tg)

    def reclaim_bd(self, task) -> None:
        if task.uses_direct_npu_dma():
            self._bd_id_allocator.free(task.bd_allocation_key, task.bd_id)

    # -- transfer emission (called by ObjectFifoHandle.fill/drain) -------------

    def record_touched_fifo(self, handle: ObjectFifoHandle) -> None:
        """Record that the runtime touches this fifo, for resolution after the body."""
        of = handle._object_fifo
        if of not in self._touched_fifos:
            self._touched_fifos.append(of)

    def emit_transfer(self, task, group) -> None:
        """Emit a DMA transfer in program order, attaching it to its task group.

        The transfer references its fifo by symbol name, so the fifo op need not
        exist yet; it is created by :meth:`resolve_touched_fifos` after the body.
        """
        if group is None:
            if self._default_group is None:
                # Imported lazily to avoid a module-load cycle (taskgroup imports
                # the active-sequence context which this module installs).
                from .taskgroup import TaskGroup

                self._default_group = TaskGroup()
            group = self._default_group
        if task.uses_direct_npu_dma():
            task.resolve(bd_id=self._bd_id_allocator.allocate(task.bd_allocation_key))
        else:
            task.resolve()
        group._add(task)

    # -- finalization ---------------------------------------------------------

    def finalize(self) -> None:
        """Resolve the implicit default group and check for leaked groups."""
        if self._default_group is not None:
            self._default_group.resolve()
        unresolved = [tg for tg in self._task_groups if not tg.resolved]
        if unresolved:
            names = ", ".join(str(tg) for tg in unresolved)
            raise IronRuntimeError(
                f"Runtime sequence ended with unresolved task group(s): {names}. "
                f"Call group.resolve() for every TaskGroup you create."
            )


class Runtime(Resolvable):
    """A Runtime contains the operations and structure of all operations that
    need to be taken care of by the host/runtime in order to run a program.
    """

    def __init__(self) -> None:
        """Initialize a runtime object."""
        self._sequence_fn: Callable | None = None
        self._sequence_arg_types: list = []
        self._sequence_items: list = []
        self._rt_data = []
        self._fifos = set()
        # Lower-level explicit-routing primitives (peers of ObjectFifo for
        # designs that hand-wire flows + DMA programs instead of letting
        # ObjectFifo manage them).
        self._flows = []
        self._locks = []
        self._tile_dmas = []
        self._scratchpad_parameters: list[ScratchpadParameter] = []
        self._trace_size = None
        self._trace_workers = None
        self._ddr_id = 4
        self._coretile_events = None
        self._coremem_events = None
        self._memtile_events = None
        self._shimtile_events = None
        self._egress_shim_col = 0
        self._active = None  # ActiveSequence while the body is being emitted

    # -- lower-level explicit-routing registration ----------------------------

    def add_flow(self, flow) -> None:
        """Register an explicit :class:`Flow` (or :class:`PacketFlow`) so the
        Program resolves it alongside the ObjectFifos."""
        self._flows.append(flow)

    def add_lock(self, lock) -> None:
        """Register an explicit :class:`Lock` shared between a Worker and a
        :class:`TileDma`."""
        self._locks.append(lock)

    def add_tile_dma(self, tile_dma) -> None:
        """Register an explicit :class:`TileDma` program."""
        self._tile_dmas.append(tile_dma)

    @property
    def flows(self):
        return list(self._flows)

    @property
    def locks(self):
        return list(self._locks)

    @property
    def tile_dmas(self):
        return list(self._tile_dmas)

    # -- sequence definition --------------------------------------------------

    def sequence(self, fn: Callable, arg_types: list) -> None:
        """Define the runtime sequence body.

        Mirrors :class:`Worker`: ``fn`` is a plain callable, registered here and
        invoked later (during :meth:`Program.resolve_program`) — exactly once —
        inside the ``aie.runtime_sequence`` builder, with one argument per entry
        in ``arg_types``. Because the body executes in the builder, ordinary
        Python control flow (``range_`` / ``if_`` from :mod:`aie.iron.controlflow`)
        and arithmetic on the scalar arguments lower to ``scf`` / ``arith`` ops in
        place, so a single compiled design can serve many runtime shapes.

        Inside ``fn``, drive data movement through the objects themselves —
        ``of.prod().fill(...)``, ``of.cons().drain(...)``, ``rtp[i] = value`` —
        and group completions with :class:`TaskGroup`. Tensor arguments arrive as
        buffer handles; scalar arguments arrive as live SSA values supporting
        arithmetic and comparison (so loop bounds / transfer sizes can be derived
        from them).

        The body runs once and is invoked solely to emit MLIR, so it should be a
        pure description of data movement: avoid host-visible side effects (the
        only effects that matter are the IRON calls it makes).

        Args:
            fn (Callable): The sequence body. Called as ``fn(*args)`` with the
                resolved block arguments, in declaration order.
            arg_types (list): One type per sequence argument — an ``np.ndarray``
                type for a buffer, or an MLIR scalar type (e.g. ``T.i32``) /
                zero-arg callable producing one for a scalar parameter.

        Raises:
            TypeError: If an argument type is not recognized.
        """
        items = []
        for t in arg_types:
            if isinstance(t, ir.Type):
                items.append(RuntimeScalar(t))
            elif getattr(t, "__origin__", None) is np.ndarray:
                items.append(RuntimeData(t))
            elif callable(t):
                items.append(RuntimeScalar(t))
            else:
                raise TypeError(
                    f"Unsupported sequence argument type: {type(t).__name__}. "
                    f"Expected np.ndarray type, ir.Type, or callable."
                )
        self._sequence_fn = fn
        self._sequence_arg_types = list(arg_types)
        self._sequence_items = items
        self._rt_data = [i for i in items if isinstance(i, RuntimeData)]

    def inline_ops(self, inline_func: Callable, inline_args: list):
        """Emit arbitrary lower-level ops at this point in the sequence.

        Runs ``inline_func(*inline_args)`` immediately within the runtime-sequence
        builder, in call order relative to surrounding data movement. Any
        :class:`Buffer` in ``inline_args`` is resolved first. Useful for tracing
        or hand-wired DMA programs.

        Args:
            inline_func (Callable): Emits ops within the MLIR context.
            inline_args (list): State the function needs (e.g. Buffers).
        """
        from .task import InlineOpRuntimeTask

        active_sequence()  # enforce in-sequence use
        InlineOpRuntimeTask(inline_func, inline_args).resolve()

    def set_barrier(self, barrier, value: int):
        """Set a :class:`WorkerRuntimeBarrier` value at this point in the sequence.

        Emits immediately, in call order relative to surrounding data movement.

        Args:
            barrier (WorkerRuntimeBarrier): The barrier to set.
            value (int): The value to set it to.
        """
        from ..worker import _BarrierSetOp

        active_sequence()  # enforce in-sequence use
        _BarrierSetOp(barrier, value).resolve()

    def enable_trace(
        self,
        trace_size: int = None,
        workers: list | None = None,
        ddr_id: int = 4,
        coretile_events: list | None = None,
        coremem_events: list | None = None,
        memtile_events: list | None = None,
        shimtile_events: list | None = None,
        egress_shim_col: int = 0,
    ):
        """Enable hardware tracing for this program.

        Configures the AIE trace units and routes trace packets to DDR via the
        shim DMA. Trace setup is emitted at the start of the sequence body.

        Args:
            trace_size (int): Size of the trace buffer in bytes.
            workers (list[Worker] | None, optional): Specific workers to trace.
                If None, all workers with ``trace`` set are traced.
            ddr_id (int, optional): XRT inout buffer index (0-4) for trace data.
            coretile_events / coremem_events / memtile_events / shimtile_events
                (list | None, optional): Up to 8 trace events per tile type.
            egress_shim_col (int, optional): Shim column used to egress trace.
        """
        self._trace_size = trace_size
        self._trace_workers = workers
        self._ddr_id = ddr_id
        self._coretile_events = coretile_events
        self._coremem_events = coremem_events
        self._memtile_events = memtile_events
        self._shimtile_events = shimtile_events
        self._egress_shim_col = egress_shim_col

    def sync_parameters(self):
        """Emit ``aiex.sync_scratchpad_parameters_from_host`` in the sequence.

        Call after all parameters have been written on the host side and before
        starting workers that read them.
        """
        self.inline_ops(lambda: sync_scratchpad_parameters_from_host(), [])

    # -- discovery (used by Program) ------------------------------------------

    @property
    def workers(self) -> list[Worker]:
        # Kept for source compatibility with Program; Workers are supplied to
        # Program directly now, so the Runtime tracks none itself.
        return []

    @property
    def fifos(self) -> list[ObjectFifoHandle]:
        """ObjectFifoHandles the Runtime is known to touch.

        Populated by :meth:`discover_fifos`; used by the Program to resolve
        runtime-facing fifos in deterministic order.
        """
        return list(self._fifos)

    def discover_fifos(self, workers: list[Worker]) -> None:
        """Find fifos the runtime sequence will touch, by walking the link graph.

        A runtime sequence fills the producer / drains the consumer of fifos that
        are reachable from the Workers through :class:`ObjectFifoLink`
        (split/join/forward). Walking that graph lets the Program resolve them up
        front in a stable order; any fifo not reachable this way (a worker-less
        forward design) is resolved lazily on first verb touch.
        """
        seen = set()
        frontier = []
        for w in workers:
            for handle in w.fifos:
                frontier.append(handle._object_fifo)
        while frontier:
            of = frontier.pop()
            if id(of) in seen:
                continue
            seen.add(id(of))
            self._fifos.add(of.prod())
            handles = ([of._prod] if of._prod else []) + list(of._cons)
            for handle in handles:
                endpoint = handle.endpoint
                if isinstance(endpoint, ObjectFifoLink):
                    for h in (*endpoint._srcs, *endpoint._dsts):
                        frontier.append(h._object_fifo)

    # -- resolution -----------------------------------------------------------

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
        device=None,
    ) -> None:
        """Emit the runtime sequence into the current device.

        Builds the ``aie.runtime_sequence`` op, binds each declared argument to
        its block argument, then runs the user body inside the builder so its
        control flow and data movement lower in place.
        """
        if self._sequence_fn is None:
            raise IronRuntimeError(
                "Runtime.sequence(fn, arg_types) must be called before resolve()."
            )

        rt_dtypes = []
        for item in self._sequence_items:
            if isinstance(item, RuntimeData):
                rt_dtypes.append(item.arr_type)
            elif isinstance(item, RuntimeScalar):
                rt_dtypes.append(item.mlir_type)
            else:
                rt_dtypes.append(item)

        active = ActiveSequence(self, device, ir.InsertionPoint.current)

        # Emit the runtime sequence. The body runs once: data-movement verbs emit
        # their DMA ops referencing each fifo by symbol name, so the fifo op need
        # not exist yet. Afterwards, resolve_touched_fifos() creates those fifo
        # ops at device scope (by which point every runtime endpoint is known, so
        # split/join/forward link groups resolve correctly).
        @runtime_sequence(*rt_dtypes)
        def sequence(*args):
            if self._trace_size:
                trace_utils.start_trace(
                    trace_size=self._trace_size,
                    ddr_id=self._ddr_id,
                    routing="single",
                    egress_shim_col=self._egress_shim_col,
                )
            # Tensor args expose a RuntimeData handle (carries shape/tap); scalar
            # args are the live SSA value (arithmetic in the body).
            body_args = []
            for item, arg_val in zip(self._sequence_items, args):
                item.op = arg_val
                body_args.append(item if isinstance(item, RuntimeData) else arg_val)
            with active_sequence_scope(active):
                self._active = active
                try:
                    self._sequence_fn(*body_args)
                    active.finalize()
                finally:
                    self._active = None

        active.resolve_touched_fifos()
