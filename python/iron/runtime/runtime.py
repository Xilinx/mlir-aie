# runtime.py -*- Python -*-
#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Runtime: orchestrates host-side data movement and worker execution for an IRON program.

The runtime sequence is written as a *callback body* -- a plain Python function
whose parameters are the runtime I/O buffers (and, optionally, runtime scalars).
The body runs eagerly inside ``@runtime_sequence`` at resolve time, mirroring how
``Worker.core_fn`` runs inside ``@core``. Because the body executes with live MLIR
values in scope, it can use native ``range_``/``if_`` control flow with
``fill``/``drain`` verbs nested inside -- the dynamic path lowers these to
``scf.for``/``scf.if`` (EmitC C++ TXN), and the static path (Python ``range``/int
bounds) elaborates to a flat binary sequence.
"""

from __future__ import annotations
from collections import defaultdict
import itertools
import logging
import numpy as np
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

from ...utils import trace as trace_utils

from ... import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]

from ...dialects.aiex import (
    sync_scratchpad_parameters_from_host,  # pyright: ignore[reportAttributeAccessIssue]
)
from ...dialects._aie_ops_gen import (  # pyright: ignore[reportMissingImports]
    RuntimeSequenceOp,
)
from ...helpers.util import try_convert_np_type_to_mlir_type
from ...dialects._aiex_ops_gen import (  # pyright: ignore[reportMissingImports]
    dma_await_task,
    dma_free_task,
)
from ..dataflow import ObjectFifoHandle
from ..device import AnyShimTile
from ..resolvable import Resolvable
from ..scratchpad_parameter import ScratchpadParameter
from ..worker import WorkerRuntimeBarrier, _BarrierSetOp
from .dmatask import DMATask
from .data import RuntimeData
from .endpoint import RuntimeEndpoint
from .taskgroup import TaskGroup
from .task import InlineOpRuntimeTask
from ._context import active_sequence, active_sequence_scope


def _iter_flat(obj):
    """Yield obj and, recursively, the elements of any nested list/tuple.

    Matches the traversal InlineOpRuntimeTask uses to find Buffer args, so
    fifos nested in the inline_args structure are registered the same way.
    """
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_flat(item)
    else:
        yield obj


class IronRuntimeError(Exception):
    """Raised by the IRON Runtime when resolution encounters an unrecoverable state."""


class ActiveSequence:
    """The state of a runtime sequence body while it is being emitted.

    The body's data-movement verbs (``fifo.fill``/``fifo.drain``) and
    ``TaskGroup`` reach this object through the active-sequence ContextVar
    (see [`_context`][iron.runtime._context]) rather than a threaded ``rt``
    reference, so the body signature carries only the runtime buffers.

    The body runs exactly once, inside the ``runtime_sequence`` op: each verb
    both binds its ObjectFifo's shim endpoint and emits the shim DMA. The DMA
    references the fifo by symbol name (a legal MLIR forward reference), so it
    does not require the fifo to be resolved yet -- the Program resolves fifos
    and cores afterward, with every runtime endpoint already bound.
    """

    def __init__(self, runtime: "Runtime"):
        self._runtime = runtime
        # Actions accumulated per task group: (dma_await_task | dma_free_task, [task]).
        self._task_group_actions: dict[TaskGroup, list] = defaultdict(list)
        # The implicit group for fill/drain calls that pass no explicit group.
        self._default_task_group = TaskGroup(next(runtime._task_group_index))
        self._open_task_groups: list[TaskGroup] = []
        self._used_default = False
        self._used_explicit = False

    def note_fifo(self, handle: ObjectFifoHandle) -> None:
        """Record that ``handle`` is driven from the runtime (its shim endpoint)."""
        self._runtime._fifos.add(handle)

    def register_task_group(self, tg: TaskGroup) -> None:
        self._open_task_groups.append(tg)

    def finish_task_group(self, tg: TaskGroup) -> None:
        """Close a task group: await its waited tasks, then free the rest.

        Waits are ordered before frees within the group, matching the
        hardware-safe order the old flat-list runtime used.
        """
        if tg in self._open_task_groups:
            self._open_task_groups.remove(tg)
        actions = self._task_group_actions.get(tg)
        if not actions:
            return
        wait_tasks = [(fn, a) for (fn, a) in actions if fn == dma_await_task]
        free_tasks = [(fn, a) for (fn, a) in actions if fn == dma_free_task]
        if len(wait_tasks) + len(free_tasks) != len(actions):
            unknown = [
                (fn, a)
                for (fn, a) in actions
                if fn != dma_await_task and fn != dma_free_task
            ]
            raise IronRuntimeError(
                f"Unknown action type detected: {','.join(str(a) for a in unknown)}"
            )
        for fn, a in wait_tasks + free_tasks:
            fn(*a)
        self._task_group_actions[tg] = []

    def emit_transfer(self, task: DMATask, task_group: TaskGroup | None) -> None:
        """Emit a DMA transfer and record its await/free action for group close."""
        task.resolve()
        if task_group is not None:
            self._used_explicit = True
            group = task_group
        else:
            self._used_default = True
            group = self._default_task_group
        action = dma_await_task if task.will_wait() else dma_free_task
        self._task_group_actions[group].append((action, [task.task]))

    def finalize(self) -> None:
        """Close bookkeeping after the body runs."""
        explicit_open = [tg for tg in self._open_task_groups if tg is not self._default_task_group]
        if explicit_open:
            tgs = ", ".join(str(t) for t in explicit_open)
            raise IronRuntimeError(f"Failed to close task groups: {tgs}")
        if (
            self._runtime._strict_task_groups
            and self._used_default
            and self._used_explicit
        ):
            raise IronRuntimeError(
                "Mixing explicit task groups and the default task group is "
                "prohibited. Please assign all tasks to a task group."
            )
        # Flush any transfers left in the default group (no explicit finish).
        if self._task_group_actions[self._default_task_group]:
            self.finish_task_group(self._default_task_group)


class Runtime(Resolvable):
    """The host-side sequence of data-movement and worker-start operations that
    execute an IRON design.

    A Runtime describes what the host does at runtime: filling input
    [`ObjectFifo`][iron.ObjectFifo]s with data and draining results back to host
    buffers. The sequence is a callback registered with
    [`sequence`][iron.runtime.runtime.Runtime.sequence]; its body reads the
    runtime buffers as parameters and moves data with ``fifo.fill(...)`` /
    ``fifo.drain(...)``.
    """

    def __init__(
        self,
        strict_task_groups: bool = True,
    ) -> None:
        """Initialize a runtime object.

        Args:
            strict_task_groups (bool): Disallows mixing the default group and explicit task groups during resolution.
                This can catch common errors, but can be set to False to disable the checks.

        """
        self._seq_fn: Callable | None = None
        self._rt_data: list[RuntimeData] = []
        self._fifos: set[ObjectFifoHandle] = set()
        # Lower-level explicit-routing primitives (peers of ObjectFifo for
        # designs that hand-wire flows + DMA programs instead of letting
        # ObjectFifo manage them).
        self._flows = []
        self._locks = []
        self._tile_dmas = []
        self._scratchpad_parameters: list[ScratchpadParameter] = []
        self._trace_size = None
        self._trace_workers = None
        self._strict_task_groups = strict_task_groups
        self._task_group_index = itertools.count()
        self._reuse_output_buffer = False
        self._egress_shim_col = 0
        self._coretile_events = None
        self._coremem_events = None
        self._memtile_events = None
        self._shimtile_events = None

    def add_flow(self, flow) -> None:
        """Register an explicit [`Flow`][iron.Flow] (or
        [`PacketFlow`][iron.PacketFlow]) so the Program resolves it alongside
        the ObjectFifos."""
        self._flows.append(flow)

    def add_lock(self, lock) -> None:
        """Register an explicit [`Lock`][iron.Lock] shared between a Worker and
        a [`TileDma`][iron.TileDma]."""
        self._locks.append(lock)

    def add_tile_dma(self, tile_dma) -> None:
        """Register an explicit [`TileDma`][iron.TileDma] program."""
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

    def sequence(
        self,
        seq_fn: Callable,
        input_types: Sequence[type[np.ndarray]],
    ) -> None:
        """Register the runtime sequence body.

        The body runs inside ``@runtime_sequence`` at resolve time, receiving one
        live value per declared input type as its positional arguments. Because
        it executes with an active MLIR insertion point, it can use native
        ``range_``/``if_`` control flow and move data with ``fifo.fill(...)`` /
        ``fifo.drain(...)``.

        Args:
            seq_fn (Callable): The sequence body. Its parameters are bound, in
                order, to the runtime buffers/scalars described by
                ``input_types``.
            input_types (Sequence[type[np.ndarray]]): The declared runtime input
                types, one per body parameter (tensor types such as
                ``np.ndarray[(M, K), np.dtype[np.int16]]`` and scalar types such
                as ``np.int32``).
        """
        self._seq_fn = seq_fn
        self._rt_data = [RuntimeData(t) for t in input_types]

    def enable_trace(
        self,
        trace_size: int | None = None,
        workers: list | None = None,
        reuse_output_buffer: bool = False,
        coretile_events: list | None = None,
        coremem_events: list | None = None,
        memtile_events: list | None = None,
        shimtile_events: list | None = None,
        egress_shim_col: int = 0,
    ):
        """Enable hardware tracing for this program.

        Configures the AIE trace units and routes trace packets to DDR via the shim DMA.

        Args:
            trace_size (int): Size of the trace buffer in bytes.
            workers (list[Worker] | None, optional): Specific workers to trace. If None,
                all workers with ``trace`` set will be traced. Defaults to None.
            reuse_output_buffer (bool, optional): When False (default), trace
                lowering appends a dedicated trace-buffer argument to the
                runtime_sequence; it lands at the tail so enabling trace never
                perturbs the data arguments' indices. When True, trace data is
                written into the tail of the last output buffer, saving a host
                buffer. Defaults to False.
            coretile_events (list | None, optional): List of up to 8 core tile trace events.
                See [the AIEX dialect reference](../AIEXDialect.md) for available
                events under (type)EventAIE such as CoreEventAIE.
                Defaults to None (uses hardware defaults).
            coremem_events (list | None, optional): List of up to 8 core memory trace events.
                Defaults to None (uses hardware defaults).
            memtile_events (list | None, optional): List of up to 8 mem tile trace events.
                Defaults to None (uses hardware defaults).
            shimtile_events (list | None, optional): List of up to 8 shim tile trace events.
                Defaults to None (uses hardware defaults).
            egress_shim_col (int, optional): Column of the shim tile used to
                egress trace packets to DDR. Defaults to 0.
        """
        self._trace_size = trace_size
        self._trace_workers = workers
        self._reuse_output_buffer = reuse_output_buffer
        self._coretile_events = coretile_events
        self._coremem_events = coremem_events
        self._memtile_events = memtile_events
        self._shimtile_events = shimtile_events
        self._egress_shim_col = egress_shim_col

    def inline_ops(self, inline_func: Callable, inline_args: list):
        """Emit arbitrary lower-level ops in the runtime sequence body.

        An escape hatch for hardware control that has no high-level IRON verb
        yet (PDI loading, custom BD writes, compression control). Call it from
        within the sequence body. Any ObjectFifoHandle passed in ``inline_args``
        is registered with the Runtime (so the Program resolves its shim
        allocation) and, if it has no endpoint yet, is bound to a shim tile.

        Args:
            inline_func (Callable): The function to execute within an MLIR context.
            inline_args (list): The state the function needs to execute.
        """
        active = active_sequence()
        for arg in _iter_flat(inline_args):
            if isinstance(arg, ObjectFifoHandle):
                if arg.endpoint is None:
                    arg.endpoint = RuntimeEndpoint(AnyShimTile)
                self._fifos.add(arg)
        if not active.registering:
            InlineOpRuntimeTask(inline_func, inline_args).resolve()

    def set_barrier(self, barrier: WorkerRuntimeBarrier, value: int):
        """Set the value of a worker barrier. Call from within the sequence body.

        Args:
            barrier (WorkerRuntimeBarrier): The WorkerRuntimeBarrier to set.
            value (int): The value to set the barrier to.
        """
        if not active_sequence().registering:
            _BarrierSetOp(barrier, value).resolve()

    def sync_parameters(self):
        """Emit `aiex.sync_scratchpad_parameters_from_host`. Call from within the
        sequence body after all parameters have been written on the host side and
        before starting workers that read them.
        """
        if not active_sequence().registering:
            _SyncParametersTask().resolve()

    @property
    def fifos(self) -> list[ObjectFifoHandle]:
        """The ObjectFifoHandles driven from the runtime by fill()/drain()."""
        return list(self._fifos)

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        """Build the ``runtime_sequence`` op and run the sequence body inside it.

        The body runs exactly once. Each ``fill``/``drain`` verb binds its
        ObjectFifo's shim endpoint and emits the shim DMA (referencing the fifo
        by symbol name, a forward reference). The Program calls this before it
        resolves ObjectFifos and cores, so by the time a fifo is resolved every
        runtime endpoint -- including those on link siblings -- is already bound.
        """
        if self._seq_fn is None:
            raise IronRuntimeError(
                "Runtime.sequence(fn, input_types) must be called before the "
                "Runtime is resolved."
            )
        rt_dtypes = [
            try_convert_np_type_to_mlir_type(rt_data.arr_type)
            for rt_data in self._rt_data
        ]
        active = ActiveSequence(self)

        seq_op = RuntimeSequenceOp(sym_name="sequence")
        entry_block = seq_op.body.blocks.append(*rt_dtypes)
        with ir.InsertionPoint(entry_block):
            for rt_data, rt_data_val in zip(self._rt_data, entry_block.arguments):
                rt_data.op = rt_data_val

            if self._trace_size is not None and self._trace_size > 0:
                trace_utils.start_trace(
                    trace_size=self._trace_size,
                    reuse_output_buffer=self._reuse_output_buffer,
                    routing="single",
                    egress_shim_col=self._egress_shim_col,
                )

            with active_sequence_scope(active):
                self._seq_fn(*self._rt_data)
                active.finalize()

        self._dedup_runtime_consumers()

    def _dedup_runtime_consumers(self) -> None:
        """Prune duplicate runtime consumer handles on shared ObjectFifos.

        A loop that drains one ObjectFifo obtains a fresh consumer handle per
        ``cons()`` call, so identical runtime consumers accumulate. Collapse
        them, matching the flat-list runtime's ``__exit__`` cleanup.
        """
        for of_handle in self._fifos:
            if of_handle._is_prod:
                continue
            fifo_obj = of_handle._object_fifo
            runtime_cons = None
            to_remove = []
            for c in fifo_obj._cons:
                if isinstance(c.endpoint, RuntimeEndpoint):
                    if not runtime_cons:
                        runtime_cons = c
                    elif (
                        c.depth == runtime_cons.depth
                        and c.dims_from_stream == runtime_cons.dims_from_stream
                    ):
                        to_remove.append(c)
                    else:
                        raise ValueError(
                            f"Found two different RuntimeEndpoints for consumers "
                            f"of the same ObjectFifo: {fifo_obj}"
                        )
            for r in to_remove:
                fifo_obj._cons.remove(r)


class _SyncParametersTask(Resolvable):
    """Emits `aiex.sync_scratchpad_parameters_from_host` during runtime sequence resolution."""

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        sync_scratchpad_parameters_from_host(loc=loc, ip=ip)
