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
from ...helpers.util import (
    try_convert_np_type_to_mlir_type,
    np_dtype_to_mlir_type,
    flatten_fn_args,
)
from ...extras.dialects.arith import constant  # pyright: ignore[reportMissingImports]
from ...dialects._aiex_ops_gen import (  # pyright: ignore[reportMissingImports]
    dma_await_task,
    dma_free_task,
)
from ..dataflow import ObjectFifoHandle
from ..resolvable import Resolvable
from ..scratchpad_parameter import ScratchpadParameter
from .dmatask import DMATask
from .data import RuntimeData
from .endpoint import RuntimeEndpoint
from .taskgroup import TaskGroup
from ._context import active_sequence, active_sequence_scope


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
        explicit_open = [
            tg for tg in self._open_task_groups if tg is not self._default_task_group
        ]
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
        seq_fn: Callable,
        inputs: Sequence,
        fn_args: "Sequence | None" = None,
        *,
        strict_task_groups: bool = True,
    ) -> None:
        """Create a runtime from its sequence body, declared inputs, and fn_args.

        Mirrors [`Worker`][iron.Worker]``(core_fn, fn_args)``: ``seq_fn`` runs
        inside ``@runtime_sequence`` at resolve time. It is called as
        ``seq_fn(*inputs, *fn_args)`` -- the runtime I/O buffers/scalars followed
        by the shared objects it operates on (ObjectFifo handles, Buffers, ...).
        Because it executes with an active MLIR insertion point, it can use native
        ``range_``/``if_`` control flow and move data with ``fifo.fill(...)`` /
        ``fifo.drain(...)``.

        Each ``inputs`` entry is either:

        * a **type** -- a tensor type (``np.ndarray[(M, K), np.dtype[np.int16]]``)
          becomes a runtime buffer passed to the body as a ``RuntimeData``; a
          scalar type (``np.int32``) becomes a **runtime** SSA scalar passed as
          its live value. The ``scf`` control flow it drives survives to the
          dynamic (EmitC) lowering.
        * a concrete **int value** -- passed to the body as a folded
          ``arith.constant`` of that value. A ``range_``/``if_`` bounded by it has
          a constant bound, so ``aie-unroll-runtime-sequence-loops`` unrolls/folds
          it to the static binary path.

        ``fn_args`` are the shared objects the body drives, registered eagerly the
        way [`Worker`][iron.Worker] registers its fn_args: an ObjectFifoHandle's
        shim endpoint is bound now (using its ``prod(tile=...)``/``cons(tile=...)``
        tile), and Buffers are recorded. Binding endpoints at construction (not
        when the body runs) lets the Program resolve fifos and cores first and
        emit the sequence body last -- so verbs that read worker-side state
        (``barrier.set``, ``inline_ops`` over a worker Buffer) see it resolved.

        Args:
            seq_fn (Callable): The sequence body. Its parameters are bound, in
                order, to ``inputs`` followed by ``fn_args``.
            inputs (Sequence): The declared runtime inputs, one per leading body
                parameter -- each a tensor/scalar type (runtime) or a concrete int
                value (folded constant).
            fn_args (Sequence | None): Shared objects (ObjectFifoHandles, Buffers,
                ...) the body operates on, bound to the trailing body parameters.
                Defaults to None (empty).
            strict_task_groups (bool): Disallows mixing the default group and explicit task groups during resolution.
                This can catch common errors, but can be set to False to disable the checks.

        """
        self._seq_fn: Callable = seq_fn
        # A concrete int input is a folded constant; anything else is a type.
        self._const_inputs = [
            v if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else None
            for v in inputs
        ]
        self._rt_data: list[RuntimeData] = [
            None if c is not None else RuntimeData(t)
            for c, t in zip(self._const_inputs, inputs)
        ]
        self._fn_args = list(fn_args) if fn_args is not None else []
        self._fifos: set[ObjectFifoHandle] = set()
        self._register_fn_args()
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

    def _register_fn_args(self) -> None:
        """Bind shared objects in fn_args now, before the Program resolves.

        Mirrors Worker.__init__: an ObjectFifoHandle gets its shim endpoint bound
        (from the handle's prod()/cons() tile) and is recorded, so the fifo has
        both ends known when the Program resolves it -- letting the sequence body
        emit last (after workers), which the body's worker-reading verbs need.

        A fn_args entry may be a nested list/tuple of handles (e.g. one per
        column); the flattened leaves are registered while the body still
        receives the structured argument.
        """
        for arg in flatten_fn_args(self._fn_args):
            if isinstance(arg, ObjectFifoHandle):
                if arg.endpoint is None:
                    arg.endpoint = RuntimeEndpoint(arg._shim_tile)
                self._fifos.add(arg)

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
        # A runtime_sequence block arg per runtime (type) input; folded-constant
        # inputs contribute no block arg.
        rt_dtypes = [
            try_convert_np_type_to_mlir_type(rt_data.arr_type)
            for rt_data in self._rt_data
            if rt_data is not None
        ]
        active = ActiveSequence(self)

        seq_op = RuntimeSequenceOp(sym_name="sequence")
        entry_block = seq_op.body.blocks.append(*rt_dtypes)
        with ir.InsertionPoint(entry_block):
            block_args = iter(entry_block.arguments)
            for rt_data in self._rt_data:
                if rt_data is not None:
                    rt_data.op = next(block_args)

            if self._trace_size is not None and self._trace_size > 0:
                trace_utils.start_trace(
                    trace_size=self._trace_size,
                    reuse_output_buffer=self._reuse_output_buffer,
                    routing="single",
                    egress_shim_col=self._egress_shim_col,
                )

            # Build the body's positional args, one per declared input:
            #   * folded-constant input -> an arith.constant of that value;
            #   * scalar type input     -> its live SSA value (used in arithmetic
            #                              and range_/if_ bounds);
            #   * tensor type input     -> its RuntimeData handle (fill/drain).
            body_args = []
            for const_val, rt_data in zip(self._const_inputs, self._rt_data):
                if const_val is not None:
                    # i32 to mirror the dynamic np.int32 scalar path, so the same
                    # body's arithmetic (extsi to i64, etc.) lowers identically.
                    body_args.append(
                        constant(int(const_val), np_dtype_to_mlir_type(np.int32))
                    )
                elif rt_data.is_scalar:
                    body_args.append(rt_data.op)
                else:
                    body_args.append(rt_data)

            # fn_args (fifos, buffers, ...) follow the inputs, as declared.
            body_args.extend(self._fn_args)

            with active_sequence_scope(active):
                self._seq_fn(*body_args)
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


def sync_parameters() -> None:
    """Emit ``aiex.sync_scratchpad_parameters_from_host`` in the sequence body.

    Call after all scratchpad parameters have been written on the host side and
    before starting workers that read them.
    """
    active_sequence()  # ensure we're inside a sequence body
    _SyncParametersTask().resolve()
