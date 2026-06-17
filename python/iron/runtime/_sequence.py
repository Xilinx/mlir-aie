# _sequence.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc.
"""Internal machinery for emitting an IRON runtime sequence.

The user-facing entry point is :meth:`Program.resolve_program`, which calls
:func:`resolve_sequence` here. The sequence body (a plain callable, like a
:class:`Worker`'s ``core_fn``) runs once inside the ``aie.runtime_sequence``
builder; the data-movement verbs (``ObjectFifoHandle.fill``/``drain``,
``Buffer.write``, ``TaskGroup``) reach the in-flight build state through
:mod:`aie.iron.runtime._context` rather than holding a reference to any
user-visible object. Nothing here is part of the public API.
"""

from __future__ import annotations

import bisect
import itertools
from collections import defaultdict, deque

from ... import ir  # type: ignore
from ...dialects.aiex import runtime_sequence
from ...utils import trace as trace_utils
from ..dataflow.objectfifo import ObjectFifoHandle, ObjectFifoLink
from .data import RuntimeData


class IronRuntimeError(Exception):
    """Raised while emitting a runtime sequence on an unrecoverable state."""


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


def discover_fifos(workers) -> set:
    """Find the fifos a runtime sequence will touch, by walking the link graph.

    A runtime sequence fills the producer / drains the consumer of fifos
    reachable from the Workers through :class:`ObjectFifoLink` (split/join/
    forward). Returns the set of producer handles so the Program can resolve
    them up front in a stable order; any fifo not reachable this way (a
    worker-less forward design) is resolved lazily on first verb touch.
    """
    fifos = set()
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
        fifos.add(of.prod())
        handles = ([of._prod] if of._prod else []) + list(of._cons)
        for handle in handles:
            endpoint = handle.endpoint
            if isinstance(endpoint, ObjectFifoLink):
                for h in (*endpoint._srcs, *endpoint._dsts):
                    frontier.append(h._object_fifo)
    return fifos


class ActiveSequence:
    """Per-build state for the runtime sequence currently being emitted.

    Created by :func:`resolve_sequence` for the duration of the sequence body
    and published via :mod:`aie.iron.runtime._context` so the data-movement
    verbs and :class:`TaskGroup` can reach it without holding a reference to a
    user-visible object.

    The sequence body runs exactly once. Data-movement verbs reference their
    ObjectFifo by symbol name (a forward reference MLIR resolves later), so DMA
    ops are emitted in program order without the fifo op needing to exist yet.
    After the body, :meth:`resolve_touched_fifos` creates the ``aie.objectfifo``
    ops for fifos the runtime touched, once all their endpoints are known.

    Responsibilities: BD-id allocation, the implicit default :class:`TaskGroup`,
    tracking runtime-touched fifos, and end-of-body finalization.
    """

    def __init__(self, device, device_ip: ir.InsertionPoint):
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
        """Finish the implicit default group and check for leaked groups."""
        if self._default_group is not None:
            self._default_group.finish()
        unfinished = [tg for tg in self._task_groups if not tg.finished]
        if unfinished:
            names = ", ".join(str(tg) for tg in unfinished)
            raise IronRuntimeError(
                f"Runtime sequence ended with unfinished task group(s): {names}. "
                f"Call group.finish() for every TaskGroup you create."
            )


def resolve_sequence(sequence_fn, sequence_items, device, trace=None):
    """Emit the ``aie.runtime_sequence`` for ``sequence_fn``.

    Builds the op with one block argument per entry in ``sequence_items``, runs
    the body once inside the builder so its control flow and data movement lower
    in place, then creates the device-scope ops for the fifos it touched.

    Each item is either a :class:`RuntimeData` (a tensor handle, passed into the
    body so verbs can read its shape/tap) or a bare scalar MLIR type / zero-arg
    callable producing one (the body receives the raw SSA ``Value``).

    Args:
        sequence_fn: The user's sequence callable.
        sequence_items: The classified arg items (RuntimeData or scalar types).
        device: The IRON device, used to resolve runtime-touched fifo tiles.
        trace: Optional :class:`TraceBuffer`; when set, the trace egress-DMA
            setup is emitted at the start of the sequence body.
    """
    from ._context import active_sequence_scope

    rt_dtypes = []
    for item in sequence_items:
        if isinstance(item, RuntimeData):
            # A handle reused across builds carries an SSA value from the prior
            # build; clear it so the set-once guard stays a within-build invariant.
            item.reset_op()
            rt_dtypes.append(item.arr_type)
        elif isinstance(item, ir.Type):  # bare MLIR scalar type
            rt_dtypes.append(item)
        else:  # zero-arg callable producing an MLIR type (e.g. T.i32)
            rt_dtypes.append(item())

    active = ActiveSequence(device, ir.InsertionPoint.current)

    @runtime_sequence(*rt_dtypes)
    def sequence(*args):
        if trace is not None:
            trace_utils.start_trace(
                trace_size=trace.trace_size,
                ddr_id=trace.ddr_id,
                routing="single",
                egress_shim_col=trace.egress_shim_col,
            )
        # Tensor args expose a RuntimeData handle (carries shape/tap); scalar
        # args are the live SSA value (arithmetic in the body).
        body_args = []
        for item, arg_val in zip(sequence_items, args):
            if isinstance(item, RuntimeData):
                item.op = arg_val
                body_args.append(item)
            else:
                body_args.append(arg_val)
        with active_sequence_scope(active):
            sequence_fn(*body_args)
            active.finalize()

    active.resolve_touched_fifos()
