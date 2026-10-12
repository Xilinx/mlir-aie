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

import itertools
import logging
from typing import TYPE_CHECKING, Callable, Sequence, get_origin

import numpy as np

from ... import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ...dialects._aie_ops_gen import (  # pyright: ignore[reportMissingImports]
    RuntimeSequenceOp,
)
from ...dialects.aiex import (
    dma_await_task,
    dma_free_task,
    npu_load_pdi,  # pyright: ignore[reportAttributeAccessIssue]
    run,  # pyright: ignore[reportAttributeAccessIssue]
    sync_scratchpad_parameters_from_host,  # pyright: ignore[reportAttributeAccessIssue]
)
from ...extras.dialects.arith import constant  # pyright: ignore[reportMissingImports]
from ...helpers.sourceloc import SourceSite, traced_body
from ...helpers.util import (
    flatten_fn_args,
    np_dtype_to_mlir_type,
    try_convert_np_type_to_mlir_type,
)
from ...utils import trace as trace_utils
from ...utils.compile.jit.markers import _DispatchParameter
from ..dataflow.objectfifo import ObjectFifoHandle
from ..resolvable import DeviceResources, Resolvable
from ..scratchpad_parameter import ScratchpadParameter
from ._context import active_configuration, active_sequence, active_sequence_scope
from .data import RuntimeData
from .dmatask import DMATask
from .endpoint import RuntimeEndpoint
from .taskgroup import TaskGroup

if TYPE_CHECKING:
    from ..configuration import DeviceConfiguration
    from ..device import Device

logger = logging.getLogger(__name__)


class IronRuntimeError(Exception):
    """Raised by the IRON Runtime when resolution encounters an unrecoverable state."""


class ActiveSequence:
    """The state of a runtime sequence body while it is being emitted.

    The body's data-movement verbs (``fifo.fill``/``fifo.drain``) and
    ``TaskGroup`` reach this object through the active-sequence ContextVar
    (see ``_context``) rather than a threaded ``rt`` reference, so the body
    signature carries only the runtime buffers.

    The body runs exactly once, inside the ``runtime_sequence`` op: each verb
    both binds its ObjectFifo's shim endpoint and emits the shim DMA. The DMA
    references the fifo by symbol name (a legal MLIR forward reference), so it
    does not require the fifo to be resolved yet. DeviceConfiguration resolves
    fifos and cores afterward, with every runtime endpoint already bound.
    """

    def __init__(self, runtime: "Runtime", seq_op: RuntimeSequenceOp, device=None):
        self._runtime = runtime
        self._seq_op = seq_op
        self._device = device
        # The implicit group for fill/drain calls that pass no explicit group.
        self._default_task_group = TaskGroup(next(runtime._task_group_index))
        self._open_task_groups: list[TaskGroup] = []
        self._used_default = False
        self._used_explicit = False

    def note_fifo(self, handle: ObjectFifoHandle) -> None:
        """Record that ``handle`` is driven from the runtime (its shim endpoint)."""
        self._runtime._fifos[handle] = None

    def resolve_in_device(self, resolvable) -> None:
        """Resolve a Buffer or Lock the body reaches first, ahead of the sequence.

        DeviceConfiguration resolves discoverable objects before the body runs.
        This method places an object named only by a runtime task at device scope.
        """
        configuration = self._runtime.configuration
        if configuration is not None:
            configuration.claim_runtime_resource(resolvable)
        with ir.InsertionPoint(self._seq_op):
            if self._device is not None:
                self._device.resolve_tile(resolvable.tile)
            resolvable.resolve()

    def register_task_group(self, tg: TaskGroup) -> None:
        self._open_task_groups.append(tg)

    def finish_task_group(self, tg: TaskGroup) -> None:
        """Close a task group: await its waited tasks, then free the rest.

        Waits are ordered before frees within the group, matching the
        hardware-safe order the old flat-list runtime used.
        """
        if tg in self._open_task_groups:
            self._open_task_groups.remove(tg)
        actions = tg._actions
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
            with a[0].location:
                fn(*a)
        tg._actions = []

    def emit_transfer(self, task: DMATask, task_group: TaskGroup | None) -> None:
        """Emit a DMA transfer and record its await/free action(s) for group close.

        A waited transfer is both awaited and freed: the dynamic BD-pool pass
        only returns an id to the runtime free-list on an explicit free (an
        await is a pure TCT sync), so a waited-but-never-freed task would leak
        a pool slot on every rolled-loop iteration. Matches the static path's
        long-standing "await implies release" convention -- it just does so
        with an explicit free instead of folding it into the await.
        """
        task.resolve()
        if task_group is not None:
            self._used_explicit = True
            group = task_group
        else:
            self._used_default = True
            group = self._default_task_group
        if task.will_wait():
            group._actions.append((dma_await_task, [task.task]))
        group._actions.append((dma_free_task, [task.task]))

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
        if self._default_task_group._actions:
            self.finish_task_group(self._default_task_group)


class Runtime(DeviceResources, Resolvable):
    """The host-side sequence of data-movement operations that execute an IRON design.

    A Runtime describes what the host does at runtime: filling input
    [`ObjectFifo`][iron.ObjectFifo]s with data and draining results back to host
    buffers. The sequence is the ``seq_fn`` callback passed to the
    constructor; its body reads the runtime buffers as parameters and moves
    data with ``fifo.fill(...)`` / ``fifo.drain(...)``.
    """

    def __init__(
        self,
        seq_fn: Callable,
        fn_args: "Sequence | None" = None,
        *,
        name: str | None = None,
        strict_task_groups: bool = True,
        implicit_configure: bool = True,
    ) -> None:
        """Create a runtime from its sequence body and fn_args.

        Mirrors [`Worker`][iron.Worker]``(core_fn, fn_args)``: ``seq_fn`` runs
        inside ``@runtime_sequence`` at resolve time, called as
        ``seq_fn(*fn_args)``, and can use native ``range_``/``if_`` and
        ``fifo.fill``/``fifo.drain``. Objects in ``fn_args`` (ObjectFifoHandles,
        Buffers, ...) are registered eagerly at construction -- fifo shim
        endpoints bind now (from ``prod(tile=)``/``cons(tile=)``) so the Program
        resolves fifos and cores before the body emits, letting body verbs read
        resolved worker state (e.g. ``barrier.set``).

        Each ``fn_args`` entry is one of:

        * an unbound **DispatchTime parameter**: replaced with its live SSA
          scalar. Forward each parameter once, in any order; parameter identity
          determines the host binding, not its position in ``fn_args``.
        * a **type** (a tensor type or a scalar type like ``np.int32``): declares
          a runtime input and is replaced with a live SSA value bound to a new
          ``runtime_sequence`` block arg -- a tensor type becomes a
          ``RuntimeData`` (``fill``/``drain`` target), a scalar type becomes the
          bare SSA value (``scf`` survives to the dynamic EmitC path).
        * a concrete **int or NumPy integer value**: also declares a runtime input,
          but is folded into a constant instead of a block arg (constant-bound
          ``range_``/``if_`` unrolls to the static binary path). One body thus
          serves both lowerings depending on whether the caller passes a type or
          an integer here. NumPy integers retain their scalar dtype; plain Python
          integers use i32 for compatibility with the common ``np.int32`` path.
        * any other object (ObjectFifoHandle, Buffer, Kernel, ScratchpadParameter,
          WorkerRuntimeBarrier, ...): passed through to the body unchanged, as
          with ``Worker.fn_args``.

        Args:
            seq_fn (Callable): The sequence body; params bound to ``fn_args`` in order.
            fn_args (Sequence | None): Types/ints (runtime inputs) and shared objects,
                in the order ``seq_fn`` expects them. Defaults to None (empty list).
            name (str | None): The runtime-sequence symbol name. Program assigns
                a unique name when this value is None.
            strict_task_groups (bool): Disallow mixing the default and explicit task groups. Defaults to True.
            implicit_configure (bool): Configure the full-ELF entry sequence's
                device when the body contains no ``aiex.configure`` or
                ``aiex.npu.load_pdi`` operation. Defaults to True.
        """
        super().__init__()
        if name == "":
            raise ValueError("Runtime name must not be empty.")
        self._site = SourceSite.capture()
        self._name = name
        self._configuration: DeviceConfiguration | None = None
        self._seq_fn: Callable = seq_fn
        self._fn_args = list(fn_args) if fn_args is not None else []
        self._dispatch_binding = object()
        dispatch_args = [
            arg for arg in self._fn_args if isinstance(arg, _DispatchParameter)
        ]
        if len({id(arg) for arg in dispatch_args}) != len(dispatch_args):
            raise TypeError(
                "Forward each DispatchTime parameter exactly once to Runtime."
            )
        if len({id(arg.owner) for arg in dispatch_args}) > 1:
            raise TypeError(
                "Runtime cannot mix DispatchTime parameters from different designs."
            )
        for container in self._fn_args:
            if isinstance(container, (list, tuple)):
                for arg in flatten_fn_args(container):
                    if isinstance(arg, _DispatchParameter):
                        raise TypeError(
                            f"DispatchTime parameter {arg.name!r} must be a direct Runtime fn_args entry."
                        )
        # A concrete int entry is a folded constant; a type/generic-alias entry
        # is a runtime input; anything else passes through as an object fn_arg.
        self._const_inputs: list[int | np.integer | None] = [
            v if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else None
            for v in self._fn_args
        ]
        self._rt_data: list["RuntimeData | None"] = [
            (
                RuntimeData(
                    arg.scalar_type if isinstance(arg, _DispatchParameter) else arg
                )
                if c is None
                and (
                    isinstance(arg, (type, _DispatchParameter))
                    or get_origin(arg) is np.ndarray
                )
                else None
            )
            for c, arg in zip(self._const_inputs, self._fn_args)
        ]
        # Keep the host scalar ABI in signature order, while the callback and
        # tensor arguments retain fn_args order. Only dispatch slots move.
        dispatch_data = iter(
            data
            for _, data in sorted(
                (
                    (arg.position, data)
                    for arg, data in zip(self._fn_args, self._rt_data)
                    if isinstance(arg, _DispatchParameter)
                ),
                key=lambda item: item[0],
            )
        )
        self._block_data = []
        for arg, data in zip(self._fn_args, self._rt_data):
            if isinstance(arg, _DispatchParameter):
                self._block_data.append(next(dispatch_data))
            elif data is not None:
                if dispatch_args and data.is_scalar:
                    raise TypeError(
                        "Runtime cannot mix bare scalar types with DispatchTime parameters."
                    )
                self._block_data.append(data)
        self._fifos: dict[ObjectFifoHandle, None] = {}
        self._register_fn_args()
        # Lower-level explicit-routing primitives (peers of ObjectFifo for
        # designs that hand-wire flows + DMA programs instead of letting
        # ObjectFifo manage them).
        self._scratchpad_parameters: list[ScratchpadParameter] = []
        self._strict_task_groups = strict_task_groups
        self._implicit_configure = implicit_configure
        self._task_group_index = itertools.count()

    @property
    def name(self) -> str | None:
        return self._name

    def _assign_name(self, name: str) -> None:
        if self._name is None:
            self._name = name

    @property
    def configuration(self) -> DeviceConfiguration | None:
        return self._configuration

    def _bind_configuration(self, configuration: DeviceConfiguration) -> None:
        if self._configuration is not None:
            raise ValueError(
                f"Runtime {self._name!r} already belongs to configuration "
                f"{self._configuration.name!r}."
            )
        self._configuration = configuration
        for flow in self._flows:
            configuration.add_flow(flow)
        for lock in self._locks:
            configuration.add_lock(lock)
        for tile_dma in self._tile_dmas:
            configuration.add_tile_dma(tile_dma)

    def call(self, *args) -> None:
        configuration = active_configuration()
        if configuration is None:
            raise RuntimeError(
                f"Runtime {self._name!r} must be called inside a configuration scope."
            )
        if configuration is not self._configuration:
            owner = (
                self._configuration.name if self._configuration is not None else None
            )
            raise ValueError(
                f"Runtime {self._name!r} belongs to configuration {owner!r}, "
                f"not {configuration.name!r}."
            )
        values = [arg.op if isinstance(arg, RuntimeData) else arg for arg in args]
        run(self._name, values)

    def _register_fn_args(self) -> None:
        """Bind shared objects before DeviceConfiguration resolves them.

        Mirrors Worker.__init__: an ObjectFifoHandle gets its shim endpoint bound
        (from the handle's prod()/cons() tile) and is recorded, so the fifo has
        both ends known when DeviceConfiguration resolves it, letting the sequence body
        emit last (after workers), which the body's worker-reading verbs need.

        A fn_args entry may be a nested list/tuple of handles (e.g. one per
        column); the flattened leaves are registered while the body still
        receives the structured argument.
        """
        for arg in flatten_fn_args(self._fn_args):
            if isinstance(arg, ObjectFifoHandle):
                if arg.endpoint is None:
                    arg.endpoint = RuntimeEndpoint(arg._shim_tile)
                self._fifos[arg] = None

    def add_flow(self, flow) -> None:
        """Register a flow for resolution alongside the ObjectFifos.

        Accepts a [`Flow`][iron.Flow] or [`PacketFlow`][iron.PacketFlow].
        """
        super().add_flow(flow)
        if self._configuration is not None:
            self._configuration.add_flow(flow)

    def add_lock(self, lock) -> None:
        """Register an explicit [`Lock`][iron.Lock] no TileDma, task or Worker reaches.

        Locks a [`TileDma`][iron.TileDma]'s or a task's Bds use are found
        from them, and a Worker's from its ``fn_args``.
        """
        super().add_lock(lock)
        if self._configuration is not None:
            self._configuration.add_lock(lock)

    def add_tile_dma(self, tile_dma) -> None:
        """Register a TileDma; channels sharing a Tile are combined at resolution."""
        if self._configuration is not None:
            self._configuration.add_tile_dma(tile_dma)
        super().add_tile_dma(tile_dma)

    def resolve_tile_dmas(self) -> None:
        """Validate and emit one DMA region per Tile without changing registrations."""
        if self._configuration is not None:
            self._configuration.resolve_tile_dmas()
            return
        super().resolve_tile_dmas()

    @property
    def fifos(self) -> list[ObjectFifoHandle]:
        """The ObjectFifoHandles driven from the runtime by fill()/drain()."""
        return list(self._fifos)

    def register_parameter(self, param: ScratchpadParameter | str | None) -> str | None:
        """Record a ScratchpadParameter a transfer uses, for the Program to declare.

        Args:
            param: The parameter, the name of one the Program declares anyway
                (e.g. from a Worker's `fn_args`), or None.

        Returns:
            The parameter's name, or None for None.
        """
        if isinstance(param, ScratchpadParameter):
            if param not in self._scratchpad_parameters:
                self._scratchpad_parameters.append(param)
            return param.name
        return param

    @property
    def scratchpad_parameters(self) -> list[ScratchpadParameter]:
        """The ScratchpadParameters the sequence's transfers registered."""
        return list(self._scratchpad_parameters)

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
        *,
        trace_size: int | None = None,
        reuse_output_buffer: bool = False,
        egress_shim_col: int = 0,
        implicit_configure_device_ref: str | None = None,
        device: Device | None = None,
    ) -> None:
        """Build the ``runtime_sequence`` op and run the sequence body inside it.

        The body runs exactly once. Each ``fill``/``drain`` verb binds its
        ObjectFifo's shim endpoint and emits the shim DMA (referencing the fifo
        by symbol name, a forward reference). The Program calls this before it
        resolves ObjectFifos and cores, so by the time a fifo is resolved every
        runtime endpoint -- including those on link siblings -- is already bound.

        Args:
            loc: Optional MLIR source location for the generated ops.
            ip: Optional MLIR insertion point for the generated ops.
            trace_size: Forwarded from
                [`Program.enable_trace`][iron.program.Program.enable_trace]; see
                there. ``None``/``0`` disables tracing.
            reuse_output_buffer: Forwarded from
                [`Program.enable_trace`][iron.program.Program.enable_trace]; see
                there.
            egress_shim_col: Forwarded from
                [`Program.enable_trace`][iron.program.Program.enable_trace]; see
                there.
            implicit_configure_device_ref: The enclosing device symbol for an
                implicit full-ELF ``aiex.npu.load_pdi``. ``None`` on the xclbin path.
            device: The [`Device`][iron.Device] that places tiles the body
                reaches first, such as a buffer only a runtime task names.
        """
        # A runtime_sequence block arg per runtime (type) input; folded-constant
        # inputs contribute no block arg.
        rt_dtypes = [
            try_convert_np_type_to_mlir_type(rt_data.arr_type)
            for rt_data in self._block_data
            if rt_data is not None
        ]
        loc = loc or self._site.location()
        seq_op = RuntimeSequenceOp(sym_name=self._name, loc=loc, ip=ip)
        active = ActiveSequence(self, seq_op, device)
        entry_block = seq_op.body.blocks.append(
            *rt_dtypes, arg_locs=[loc] * len(rt_dtypes)
        )
        with ir.InsertionPoint(entry_block), seq_op.location:
            block_args = iter(entry_block.arguments)
            for rt_data in self._block_data:
                if rt_data is not None:
                    rt_data.op = next(block_args)
            for arg in self._fn_args:
                if isinstance(arg, _DispatchParameter):
                    arg._bind(self._dispatch_binding)

            if trace_size is not None and trace_size > 0:
                trace_utils.start_trace(
                    trace_size=trace_size,
                    reuse_output_buffer=reuse_output_buffer,
                    routing="single",
                    egress_shim_col=egress_shim_col,
                )

            # Build the body's positional args, one per fn_args entry, in order:
            #   * folded-constant entry -> an arith.constant of that value;
            #   * scalar type entry     -> its live SSA value (used in arithmetic
            #                              and range_/if_ bounds);
            #   * tensor type entry     -> its RuntimeData handle (fill/drain);
            #   * anything else         -> passed through unchanged (fifos, etc).
            body_args = []
            for arg, const_val, rt_data in zip(
                self._fn_args, self._const_inputs, self._rt_data
            ):
                if const_val is not None:
                    dtype = (
                        type(const_val)
                        if isinstance(const_val, np.integer)
                        else np.int32
                    )
                    scalar_type = np_dtype_to_mlir_type(dtype)
                    value = int(const_val)
                    # IntegerAttr's C API accepts int64_t, including the bit
                    # pattern of a uint64/index value above INT64_MAX.
                    if value > np.iinfo(np.int64).max:
                        value -= 1 << 64
                    if (
                        isinstance(scalar_type, ir.IntegerType)
                        and ir.IntegerType(scalar_type).is_unsigned
                    ):
                        # arith.constant requires signless integers. EmitC's
                        # ConstantLike op preserves unsigned types and folds.
                        from ...dialects.emitc import (  # pyright: ignore[reportMissingImports]
                            ConstantOp,
                        )

                        body_args.append(
                            ConstantOp(
                                scalar_type, ir.IntegerAttr.get(scalar_type, value)
                            ).result
                        )
                    else:
                        body_args.append(constant(value, scalar_type))
                elif rt_data is not None:
                    body_args.append(rt_data.op if rt_data.is_scalar else rt_data)
                else:
                    body_args.append(arg)

            with active_sequence_scope(active):
                with traced_body(self._seq_fn):
                    self._seq_fn(*body_args)
                active.finalize()

        if implicit_configure_device_ref is not None and self._implicit_configure:

            def contains_configuration(block: ir.Block) -> bool:
                for operation in block.operations:
                    if operation.name in ("aiex.configure", "aiex.npu.load_pdi"):
                        return True
                    for region in operation.regions:
                        if any(
                            contains_configuration(nested_block)
                            for nested_block in region.blocks
                        ):
                            return True
                return False

            if not contains_configuration(entry_block):
                operations = list(entry_block.operations)
                with ir.InsertionPoint(entry_block):
                    load_pdi = npu_load_pdi(device_ref=implicit_configure_device_ref)
                if operations:
                    load_pdi.operation.move_before(operations[0])

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
                        and c.from_stream == runtime_cons.from_stream
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
