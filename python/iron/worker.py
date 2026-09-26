# worker.py -*- Python -*-
#
# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Worker and WorkerRuntimeBarrier: compute-core tasks and runtime synchronization primitives."""

import sys
from typing import Callable

from .. import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ..dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIETileType,
)
from ..dialects.aie import (
    core,
    lock,
    use_lock,  # pyright: ignore[reportAttributeAccessIssue]
)
from ..dialects.aiex import (
    LockAction,  # pyright: ignore[reportAttributeAccessIssue]
    set_lock_value,
)
from ..helpers.dialects.scf import _for as range_
from ..helpers.util import flatten_fn_args
from ..utils.compile.jit.markers import _DispatchParameter
from .buffer import Buffer
from .dataflow.endpoint import ObjectFifoEndpoint
from .dataflow.objectfifo import ObjectFifo, ObjectFifoHandle
from .device import AnyComputeTile, Tile
from .resolvable import Resolvable
from .scratchpad_parameter import ScratchpadParameter


class Worker(ObjectFifoEndpoint):
    """A task to be run on an AIE compute core.

    A Worker takes a ``core_fn`` callable and the arguments it needs (ObjectFIFO handles,
    Buffers, Kernels, etc.). Each Worker is placed on a single compute tile, either
    explicitly via ``tile`` or automatically by the ``--aie-place-tiles`` compiler pass.
    """

    def __init__(
        self,
        core_fn: Callable | None,
        fn_args: list | None = None,
        tile: Tile | None = AnyComputeTile,
        while_true: bool = True,
        stack_size: int | None = None,
        data_size: int | None = None,
        trace: int | None = None,
        trace_events: list | None = None,
        dynamic_objfifo_lowering: bool | None = None,
    ):
        """Construct a Worker.

        Args:
            core_fn (Callable | None): The task to run on a core. If None, a busy-loop (`while(true): pass`) core will be generated.
            fn_args (list | None, optional): Pointers to arguments, which should include all context the core_fn needs to run. Defaults to None (empty list).
            tile (Tile, optional): The compute tile for the Worker. Also accepts None (treated as AnyComputeTile). Defaults to AnyComputeTile.
            while_true (bool, optional): If true, will wrap the core_fn in a while(true) loop to ensure it runs until reconfiguration. Defaults to True.
            stack_size (int, optional): The stack_size in bytes for the worker. Defaults to AIETargetModel::getDefaultCoreStackSize() (currently 1024 bytes).
            data_size (int, optional): Bytes of data memory to reserve for this
                core's compiled sections (.data/.rodata/.bss), beyond the stack. The
                buffer allocator packs the tile's buffers around the reservation. None
                leaves the core whatever contiguous run the buffers leave.
                Defaults to None.
            trace (int, optional): If >0, enable tracing for this worker.
            trace_events (list | None, optional): Custom list of trace events for this worker. Defaults to None.
            dynamic_objfifo_lowering (bool | None, optional): Per-core override for the
                ``aie-objectFifo-stateful-transform`` pass's lowering choice. ``True`` forces
                dynamic (loop-preserving) lowering for this core; ``False`` forces static
                LCM-based unrolling. ``None`` (default) leaves the choice to the compiler's
                global ``--dynamic-objFifos`` flag. Note: the per-core attribute is only
                honored when the global flag is ``false``; when global is ``true`` the
                attribute is ignored. Defaults to None.

        Raises:
            ValueError: Parameters are validated.
        """
        for arg in flatten_fn_args(
            [
                fn_args or [],
                tile,
                while_true,
                stack_size,
                data_size,
                trace,
                trace_events,
                dynamic_objfifo_lowering,
            ]
        ):
            if isinstance(arg, _DispatchParameter):
                arg._misuse()
        if tile is None:
            tile = AnyComputeTile
        if tile.tile_type is not None and tile.tile_type != AIETileType.CoreTile:
            raise ValueError(
                f"Worker requires a compute tile, but got tile_type={tile.tile_type}"
            )
        if stack_size is not None:
            if not isinstance(stack_size, int) or isinstance(stack_size, bool):
                raise ValueError(
                    f"Worker stack_size must be an int, but got "
                    f"{type(stack_size).__name__}"
                )
            if stack_size < 1:
                raise ValueError(
                    f"Worker stack_size must be >= 1, but got {stack_size}"
                )
        if data_size is not None:
            if not isinstance(data_size, int) or isinstance(data_size, bool):
                raise ValueError(
                    f"Worker data_size must be an int, but got "
                    f"{type(data_size).__name__}"
                )
            if data_size < 0:
                raise ValueError(
                    f"Worker data_size must be >= 0, but got " f"{data_size}"
                )
        # Store the user's Tile directly when it is already typed as CoreTile.
        # This preserves Python object identity so a Buffer and a Worker that
        # share the same Tile object resolve to a single LogicalTileOp. When a
        # fresh copy is needed (untyped tile or the singleton default) use
        # with_type() — it always returns a new object.
        if tile.tile_type == AIETileType.CoreTile and tile is not AnyComputeTile:
            self._tile = tile
        else:
            self._tile = tile.with_type(AIETileType.CoreTile)
        self._while_true = while_true
        self.stack_size = stack_size
        self.data_size = data_size
        self._dynamic_objfifo_lowering = dynamic_objfifo_lowering
        self.trace = trace
        self.trace_events = trace_events

        # If no core_fn is given, make a simple while(true) loop.
        if core_fn is None:

            def do_nothing_core_fun(*args) -> None:
                for _ in range_(sys.maxsize):
                    pass

            self.core_fn = do_nothing_core_fun
        else:
            self.core_fn = core_fn
        self.fn_args = fn_args if fn_args is not None else []
        self._fifos = []
        self._buffers = []
        self._barriers = []
        # CascadeFlow objects whose source is this Worker. Populated by
        # CascadeFlow(src, dst).__init__ and consumed by Program.resolve()
        # to emit aie.cascade_flow ops after worker placement.
        self._outgoing_cascades: list = []

        # Check arguments to the core. Some information is saved for resolution.
        # fn_args may nest lists (e.g. one fifo per column); iterate the flattened
        # leaves for registration while the core_fn still receives the structure.
        for arg in flatten_fn_args(self.fn_args):
            if isinstance(arg, ObjectFifoHandle):
                arg.endpoint = self
                self._fifos.append(arg)
            elif isinstance(arg, Buffer):
                # A Buffer pinned to an EXPLICIT tile may legitimately be shared
                # across Workers: AIE compute tiles can read a neighbor tile's L1
                # directly, so a producer core's output buffer can be an input to a
                # consumer core on an adjacent tile. In that case the FIRST worker that
                # references it "owns"/places it and later workers are non-owning
                # readers. We only forbid sharing for AUTO-PLACED buffers (no explicit
                # tile), where two owners would race to pin it to different tiles.
                # Note: ``_tile`` alone is not a reliable signal — the owning Worker
                # auto-pins ``_tile`` to its own tile below — so we key off
                # ``_explicit_tile``, which records the user's construction-time intent.
                if arg._owner_worker is not None and arg._owner_worker is not self:
                    if not arg._explicit_tile:
                        raise ValueError(
                            f"Buffer '{arg._name}' has no explicit tile and is shared "
                            f"across Workers; pin it to a tile (Buffer(tile=...)) so "
                            f"placement is unambiguous."
                        )
                    # shared reader: keep original owner, just record the reference.
                    self._buffers.append(arg)
                else:
                    arg._owner_worker = self
                    self._buffers.append(arg)
                    # If the Buffer has no tile, pin it to the Worker's tile as a
                    # convenience.  If the user pinned it explicitly to a neighbor
                    # tile (AIE compute tiles can read N/S/E/W neighbors' L1
                    # directly), honor that placement — Program.resolve discovers
                    # the neighbor tile via Buffer.tiles().
                    if arg._tile is None:
                        arg._tile = self._tile
            elif isinstance(arg, ScratchpadParameter):
                pass  # ScratchpadParameters are device-level symbols; no tile placement needed
            elif isinstance(arg, ObjectFifo):
                # This is an easy error to make, so we catch it early
                raise ValueError(
                    "Cannot give an ObjectFifo directly to a worker; "
                    "must give an ObjectFifoHandle obtained through "
                    "ObjectFifo.prod() or ObjectFifo.cons()"
                )
            elif isinstance(arg, WorkerRuntimeBarrier):
                self._barriers.append(arg)
            # Kernel/ExternalFunction instances are valid fn_args — they resolve to
            # func.call ops when invoked inside core_fn and carry link_with on their
            # func.func declaration. Other unrecognized args are assumed to be
            # metaprogramming values (Python scalars, etc.).

    @staticmethod
    def grid(
        rows: int,
        cols: int,
        factory: Callable[[int, int], "Worker"],
    ) -> list[list["Worker"]]:
        """Build a 2D grid of Workers; ``factory(r, c)`` returns one Worker.

        Replaces the common pattern::

            ws = [Worker(...) for i in range(R) for j in range(C)]
            ws[i * C + j]  # 1-D index arithmetic

        with::

            ws = Worker.grid(R, C, lambda r, c: Worker(...))
            ws[i][j]       # natural 2-D access

        Args:
            rows: Outer-dimension count (e.g. column index).
            cols: Inner-dimension count (e.g. channel index).
            factory: Called once per cell with ``(r, c)``; must return a Worker.

        Returns:
            ``rows``-by-``cols`` nested list of Worker instances.
        """
        return [[factory(r, c) for c in range(cols)] for r in range(rows)]

    @property
    def tile(self) -> Tile:
        """The compute tile this Worker is placed on."""
        assert self._tile is not None
        return self._tile

    @property
    def flat_fn_args(self) -> list:
        """fn_args with any nested lists/tuples flattened to their leaves.

        Use this (not ``fn_args``) when iterating to register/resolve individual
        arguments; ``fn_args`` keeps its structure for the core_fn call.
        """
        return list(flatten_fn_args(self.fn_args))

    @property
    def fifos(self) -> list[ObjectFifoHandle]:
        """Returns a list of ObjectFifoHandles given to the Worker via fn_args.

        Returns:
            list[ObjectFifoHandle]: ObjectFifoHandles used by the Worker.
        """
        return self._fifos.copy()

    @property
    def buffers(self) -> list[Buffer]:
        """Returns a list of Buffers given to the Worker via fn_args.

        Returns:
            list[Buffer]: Buffer used by the Worker.
        """
        return self._buffers.copy()

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if not self._tile:
            raise ValueError("Must place Worker before it can be resolved.")
        my_tile = self._tile.op

        # Create the necessary locks for the core operation to synchronize with the runtime sequence
        # and register them in the corresponding barriers.
        for barrier in self._barriers:
            barrier_lock = lock(my_tile)
            barrier._add_worker_lock(barrier_lock)

        try:

            @core(
                my_tile,
                stack_size=self.stack_size,
                data_size=self.data_size,
                dynamic_objfifo_lowering=self._dynamic_objfifo_lowering,
            )
            def core_body():
                # Always wrap in an scf.for so the lowered MLIR matches expectations
                # downstream (the lower-level aie dialect uses the same pattern with
                # bound=1 for single-shot workers). Using Python range(1) here would
                # emit the body inline with no scf.for wrapper, which the dataflow
                # lowerer treats differently and can cause runtime hangs.
                for _ in range_(sys.maxsize if self._while_true else 1):
                    self.core_fn(*self.fn_args)

        finally:
            # Only this Worker's lock may be live in each barrier while its core
            # body is being built; see _add_worker_lock below.
            for barrier in self._barriers:
                barrier._clear_pending_lock()


class WorkerRuntimeBarrier:
    """A barrier allowing individual workers to synchronize with the runtime sequence."""

    def __init__(self, initial_value: int = 0):
        """Initialize a WorkerRuntimeBarrier.

        Args:
            initial_value (int, optional): The initial lock value. Defaults to 0.
        """
        self.initial_value = initial_value
        self.worker_locks = []
        # Lock most recently appended by _add_worker_lock(), cleared by
        # Worker.resolve() once its core_fn has run. wait_for_value()/
        # release_with_value() check worker_locks[-1] against this instead of
        # trusting position alone -- Worker identity isn't available at their
        # call site, so this turns a silent resolve-order mis-bind into a loud one.
        self._pending_lock = None

    def wait_for_value(self, value: int):
        """Wait for the barrier to be set to `value`.

        Should be called from inside a core function.

        Args:
            value (int): The value to wait for.
        """
        self._check_pending_lock()
        use_lock(self.worker_locks[-1], LockAction.Acquire, value=value)

    def set(self, value: int):
        """Set the barrier to ``value`` from within a runtime sequence body.

        Args:
            value (int): The value to set the barrier to.
        """
        _BarrierSetOp(self, value).resolve()

    def _add_worker_lock(self, lock):
        """Register an additional lock in the barrier."""
        if self._pending_lock is not None:
            raise RuntimeError(
                "WorkerRuntimeBarrier registered a second Worker's lock before "
                "the first Worker's core body finished resolving. "
                "wait_for_value()/release_with_value() bind to worker_locks[-1] "
                "positionally, so resolve() must fully finish one Worker before "
                "starting the next."
            )
        self.worker_locks.append(lock)
        self._pending_lock = lock

    def _clear_pending_lock(self):
        """Mark the current pending lock as consumed; called by Worker.resolve()."""
        self._pending_lock = None

    def _check_pending_lock(self):
        if len(self.worker_locks) == 0:
            raise ValueError(
                "No workers have been registered for this barrier. Need to pass the barrier as an argument to the worker."
            )
        if self.worker_locks[-1] is not self._pending_lock:
            raise RuntimeError(
                "WorkerRuntimeBarrier.wait_for_value()/release_with_value() called "
                "outside their registering Worker's resolve() -- the positional "
                "worker_locks[-1] binding is stale."
            )

    def _set_barrier_value(self, value: int):
        """Set the value of the barrier."""
        for worker_lock in self.worker_locks:
            set_lock_value(worker_lock, value)

    def release_with_value(self, value: int):
        """Release and decrement the barrier by `value` inside the core.

        Args:
            value (int): The value to decrement by in Release.
        """
        self._check_pending_lock()
        use_lock(self.worker_locks[-1], LockAction.Release, value=value)


class _BarrierSetOp(Resolvable):
    """A resolvable instance of a WorkerRuntimeBarrier. This should not be used directly."""

    def __init__(self, barrier: WorkerRuntimeBarrier, value: int):
        """Construct a _BarrierSetOp.

        Args:
            barrier (WorkerRuntimeBarrier): The barrier whose value will be set.
            value (int): The value to set.
        """
        self.barrier: WorkerRuntimeBarrier = barrier
        self.value: int = value

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        self.barrier._set_barrier_value(self.value)
