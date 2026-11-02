# worker.py -*- Python -*-
#
# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Worker and WorkerRuntimeBarrier: compute-core tasks and runtime synchronization primitives."""

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
    wait_scratchpad_parameters,  # pyright: ignore[reportAttributeAccessIssue]
)
from ..helpers.dialects.scf import _for as range_
from ..helpers.dialects.scf import forever_
from ..helpers.sourceloc import Declared, SourceSite, iron_name, traced_body
from ..helpers.taplib import Affine
from ..helpers.util import flatten_fn_args
from ..utils.compile.jit.markers import _DispatchParameter
from .buffer import Buffer
from .dataflow.endpoint import ObjectFifoEndpoint
from .dataflow.objectfifo import ObjectFifo, ObjectFifoHandle
from .dataflow.objectfifopool import ObjectFifoPoolHandle
from .device import AnyComputeTile, Tile
from .kernel import Kernel
from .resolvable import PerDeviceConfiguration, Resolvable
from .scratchpad_parameter import ScratchpadParameter


class Worker(Declared, ObjectFifoEndpoint, PerDeviceConfiguration):
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
        wait_for_parameters: bool = False,
        name: str | None = None,
    ):
        """Construct a Worker.

        Args:
            core_fn (Callable | None): The task to run on a core. If None, a busy-loop (`while(true): pass`) core will be generated.
            fn_args (list | None, optional): Pointers to arguments, which should include all context the core_fn needs to run. Defaults to None (empty list).
            tile (Tile, optional): The compute tile for the Worker. Also accepts None (treated as AnyComputeTile). Defaults to AnyComputeTile.
            while_true (bool, optional): If true, will wrap the core_fn in a while(true) loop to ensure it runs until reconfiguration. Defaults to True.
            stack_size (int, optional): The stack_size in bytes for the worker.
                None lets aiecc reserve the core's measured stack requirement, never
                less than AIETargetModel::getDefaultCoreStackSize() (currently 1024
                bytes). Set it where aiecc cannot measure the core, as in a Chess
                build. Defaults to None.
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
            wait_for_parameters (bool, optional): If true, each iteration of the
                Worker's loop first waits for the current dispatch's
                ScratchpadParameter values (``aiex.wait_scratchpad_parameters``),
                so the reads in ``core_fn`` see this dispatch's values. Set it when
                one iteration of a ``while_true`` Worker is one dispatch; without
                it the Worker waits only once, before its first iteration.
                Requires a ScratchpadParameter in ``fn_args``. Defaults to False.
            name (str | None, optional): Names the Worker's core in its source
                location (``iron:<name>``). None lets its configuration name it
                ``worker<N>``. Defaults to None.

        Raises:
            ValueError: Parameters are validated.
            TypeError: An argument is computed per call.
        """
        self._site = SourceSite.capture()
        self.name = name
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
                arg.refuse()
            if isinstance(arg, Affine):
                raise TypeError(
                    f"A Worker cannot take {arg}, which is computed per call; use "
                    "CompileTime or specialize"
                )
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
        if wait_for_parameters and not any(
            isinstance(arg, ScratchpadParameter)
            for arg in flatten_fn_args(fn_args or [])
        ):
            raise ValueError(
                "Worker(wait_for_parameters=True) needs a ScratchpadParameter "
                "in fn_args: there are no parameters to wait for."
            )
        self._wait_for_parameters = wait_for_parameters
        self.trace = trace
        self.trace_events = trace_events

        # If no core_fn is given, make a simple while(true) loop.
        if core_fn is None:

            def do_nothing_core_fun(*args) -> None:
                for _ in forever_():
                    pass

            self.core_fn = do_nothing_core_fun
        else:
            self.core_fn = core_fn
        self.fn_args = fn_args if fn_args is not None else []
        self._fifos = []
        self._buffers = []
        self._barriers = []
        # CascadeFlow registers each outgoing edge here for configuration discovery.
        self._outgoing_cascades: list = []

        # Check arguments to the core. Some information is saved for resolution.
        # fn_args may nest lists (e.g. one fifo per column); iterate the flattened
        # leaves for registration while the core_fn still receives the structure.
        for arg in flatten_fn_args(self.fn_args):
            if isinstance(arg, (ObjectFifoHandle, ObjectFifoPoolHandle)):
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
                            f"Buffer {arg._name or arg._arr_type} has no explicit tile and is shared "
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
                    # directly), honor that placement. DeviceConfiguration discovers
                    # the neighbor tile via Buffer.tiles().
                    arg.place(self._tile)
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

        # A library kernel's contract may name a setup kernel, such as the
        # rounding mode its bf16 stores assume; a fresh core boots in floor.
        # Run each once before the loop, unless fn_args already hands it over
        # for core_fn to call.
        kernels = [a for a in flatten_fn_args(self.fn_args) if isinstance(a, Kernel)]
        handed = {k.name for k in kernels}
        self._setup_kernels = []
        for k in kernels:
            if k.contract is None or k.contract.setup is None:
                continue
            setup = k.contract.setup()
            if setup.name not in handed:
                handed.add(setup.name)
                self._setup_kernels.append(setup)

    @staticmethod
    def grid(
        rows: int,
        cols: int,
        factory: Callable[[int, int], "Worker"],
    ) -> list[list["Worker"]]:
        """Build a 2D grid of Workers; ``factory(r, c)`` returns one Worker.

        Replaces the common pattern:

        ```python
        ws = [Worker(...) for i in range(R) for j in range(C)]
        ws[i * C + j]  # 1-D index arithmetic
        ```

        with:

        ```python
        ws = Worker.grid(R, C, lambda r, c: Worker(...))
        ws[i][j]       # natural 2-D access
        ```

        Args:
            rows: Outer-dimension count (e.g. column index).
            cols: Inner-dimension count (e.g. channel index).
            factory: Called once per cell with ``(r, c)``; must return a Worker.

        Returns:
            ``rows``-by-``cols`` nested list of Worker instances.
        """
        return [[factory(r, c) for c in range(cols)] for r in range(rows)]

    def __str__(self) -> str:
        return f"Worker({self.name or self._site.describe(type(self).__name__)!r})"

    @property
    def tile(self) -> Tile:
        """The compute tile this Worker is placed on."""
        assert self._tile is not None
        return self._tile

    @property
    def while_true(self) -> bool:
        """Whether the core loops until reconfiguration rather than running once."""
        return self._while_true

    @property
    def wait_for_parameters(self) -> bool:
        """Whether each pass of the core's loop waits for a dispatch's parameters."""
        return self._wait_for_parameters

    @property
    def flat_fn_args(self) -> list:
        """fn_args with any nested lists/tuples flattened to their leaves.

        Use this (not ``fn_args``) when iterating to register/resolve individual
        arguments; ``fn_args`` keeps its structure for the core_fn call. The
        setup kernels the Worker runs for its kernels' contracts come last.
        """
        return list(flatten_fn_args(self.fn_args)) + self._setup_kernels

    @property
    def fifos(self) -> list[ObjectFifoHandle | ObjectFifoPoolHandle]:
        """Returns the fifo and pool handles given to the Worker via fn_args.

        Returns:
            list[ObjectFifoHandle | ObjectFifoPoolHandle]: handles the Worker
                acquires and releases objects through.
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
        with loc or self._site.location():
            # Create the necessary locks for the core operation to synchronize with the runtime sequence
            # and register them in the corresponding barriers.
            for barrier in self._barriers:
                barrier_lock = lock(my_tile)
                barrier.worker_locks.append(barrier_lock)

            # The core is named after the Worker; its body's statements keep
            # their own locations.
            core_loc = (
                ir.Location.name(iron_name(self.name), childLoc=ir.Location.current)
                if self.name
                else ir.Location.current
            )
            with core_loc:

                @core(
                    my_tile,
                    stack_size=self.stack_size,
                    data_size=self.data_size,
                    dynamic_objfifo_lowering=self._dynamic_objfifo_lowering,
                )
                def core_body():
                    # A Worker that runs until reconfiguration loops with no counter.
                    # A single-shot Worker still wraps its body in an scf.for with
                    # bound 1 so the lowered MLIR matches expectations downstream (the
                    # lower-level aie dialect uses the same pattern). Using Python
                    # range(1) here would emit the body inline with no scf.for wrapper,
                    # which the dataflow lowerer treats differently and can cause
                    # runtime hangs.
                    for setup in self._setup_kernels:
                        setup()
                    for _ in forever_() if self._while_true else range_(1):
                        if self._wait_for_parameters:
                            wait_scratchpad_parameters()
                        with traced_body(self.core_fn):
                            self.core_fn(*self.fn_args)


class WorkerRuntimeBarrier(PerDeviceConfiguration):
    """A barrier allowing individual workers to synchronize with the runtime sequence."""

    def __init__(self, initial_value: int = 0):
        """Initialize a WorkerRuntimeBarrier.

        Args:
            initial_value (int, optional): The initial lock value. Defaults to 0.
        """
        self.initial_value = initial_value
        self.worker_locks = []

    def _get_worker_lock(self) -> ir.Value:
        op = ir.InsertionPoint.current.block.owner.operation
        while op is not None:
            if op.name == "aie.core":
                core_tile = op.operands[0].owner.operation
                for worker_lock in self.worker_locks:
                    lock_tile = worker_lock.owner.operation.operands[0].owner.operation
                    # Value equality emits arith.cmpi; compare defining operations.
                    if lock_tile == core_tile:
                        return worker_lock
                break
            if op.name in ("aie.device", "builtin.module"):
                break
            op = op.parent
        raise ValueError(
            "No lock is registered for this barrier on the current core's tile; "
            "pass the barrier in this Worker's fn_args."
        )

    def wait_for_value(self, value: int, *, greater_equal: bool = False):
        """Wait for the barrier to be set to `value`.

        Should be called from inside a core function. The wait leaves the
        barrier at ``value``; a worker that loops over dispatches calls
        ``release_with_value`` after reading its runtime parameters so the
        next iteration waits for the next ``set``.

        Args:
            value (int): The value to wait for.
            greater_equal (bool): Use a consuming greater-than-or-equal lock wait.
        """
        action = LockAction.AcquireGreaterEqual if greater_equal else LockAction.Acquire
        use_lock(self._get_worker_lock(), action, value=value)

    def set(self, value: int):
        """Set the barrier to ``value`` from within a runtime sequence body.

        Args:
            value (int): The value to set the barrier to.
        """
        _BarrierSetOp(self, value).resolve()

    def _set_barrier_value(self, value: int):
        """Set the value of the barrier."""
        for worker_lock in self.worker_locks:
            set_lock_value(worker_lock, value)

    def release_with_value(self, value: int):
        """Release the barrier, adding ``value`` to it, inside the core.

        Args:
            value (int): The value to add.
        """
        use_lock(self._get_worker_lock(), LockAction.Release, value=value)


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
