# worker.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
"""Worker and WorkerRuntimeBarrier: compute-core tasks and runtime synchronization primitives."""

import sys
from typing import Callable

from .. import ir  # type: ignore
from ..dialects.aie import core, lock, use_lock
from ..dialects.aiex import set_lock_value, LockAction
from ..helpers.dialects.scf import _for as range_
from .device import PlacementTile, AnyComputeTile, Tile
from .dataflow.objectfifo import ObjectFifoHandle, ObjectFifo
from .dataflow.endpoint import ObjectFifoEndpoint
from .buffer import Buffer
from .resolvable import Resolvable


class Worker(ObjectFifoEndpoint):
    """A task to be run on an AIE compute core.

    A Worker takes a ``core_fn`` callable and the arguments it needs (ObjectFIFO handles,
    Buffers, Kernels, etc.). Each Worker must be placed on a single compute tile, either
    explicitly via ``placement`` or automatically by a :class:`~aie.iron.placers.Placer`.
    """

    def __init__(
        self,
        core_fn: Callable | None,
        fn_args: list = [],
        placement: PlacementTile | None = AnyComputeTile,
        while_true: bool = True,
        stack_size: int = None,
        allocation_scheme: str = None,
        trace: int = None,
        trace_events: list = None,
        dynamic_objfifo_lowering: bool = None,
    ):
        """Construct a Worker

        Args:
            core_fn (Callable | None): The task to run on a core. If None, a busy-loop (`while(true): pass`) core will be generated.
            fn_args (list, optional): Pointers to arguments, which should include all context the core_fn needs to run. Defaults to [].
            placement (PlacementTile | None, optional): The placement for the Worker. Defaults to AnyComputeTile.
            while_true (bool, optional): If true, will wrap the core_fn in a while(true) loop to ensure it runs until reconfiguration. Defaults to True.
            stack_size (int, optional): The stack_size in bytes to be allocated for the worker. Defaults to 1024 bytes.
            allocation_scheme (str, optional): The memory allocation scheme to use for the Worker, either 'basic-sequential' or 'bank-aware'. If None, defaults to bank-aware.
                Will override any allocation scheme set on the tile given as placement.
            trace (int, optional): If >0, enable tracing for this worker.
            trace_events (list | None, optional): Custom list of trace events for this worker. Defaults to None.
            dynamic_objfifo_lowering (bool, optional): If True, enables dynamic ObjectFifo lowering for runtime-parameterized loop bounds. Defaults to None.

        Raises:
            ValueError: Parameters are validated.
        """
        self._tile = placement
        self._while_true = while_true
        self.stack_size = stack_size
        self.dynamic_objfifo_lowering = dynamic_objfifo_lowering
        self.allocation_scheme = allocation_scheme
        if allocation_scheme:
            self._tile.allocation_scheme = allocation_scheme
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
        self.fn_args = fn_args
        self._fifos = []
        self._buffers = []
        self._barriers = []

        # Check arguments to the core. Some information is saved for resolution.
        for arg in self.fn_args:
            if isinstance(arg, ObjectFifoHandle):
                arg.endpoint = self
                self._fifos.append(arg)
            elif isinstance(arg, Buffer):
                self._buffers.append(arg)
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

    def place(self, tile: Tile) -> None:
        """Set the placement of the Worker.

        Args:
            tile (Tile): The placement location.
        """
        tile.allocation_scheme = self.allocation_scheme
        ObjectFifoEndpoint.place(self, tile)

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
            l = lock(my_tile)
            barrier._add_worker_lock(l)

        @core(my_tile, stack_size=self.stack_size,
              dynamic_objfifo_lowering=self.dynamic_objfifo_lowering)
        def core_body():
            for _ in range_(sys.maxsize) if self._while_true else range(1):
                self.core_fn(*self.fn_args)


class WorkerRuntimeBarrier:
    """A barrier allowing individual workers to synchronize with the runtime sequence."""

    def __init__(self, initial_value: int = 0):
        """Initialize a WorkerRuntimeBarrier.

        Args:
            initial_value (int, optional): The initial lock value. Defaults to 0.
        """
        self.initial_value = initial_value
        self.worker_locks = []

    def wait_for_value(self, value: int):
        """
        Should be called from inside a core function.
        Wait for the barrier to be set to `value`.

        Args:
            value (int): The value to wait for.
        """
        # Here this is assuming that the we are currently placing the last added lock
        # And therefore that wait_for_value operations are placed just after their corresponding Worker...
        # This is a pretty bad assumption, think about an alternative way to solve this
        if len(self.worker_locks) == 0:
            raise ValueError(
                "No workers have been registered for this barrier. Need to pass the barrier as an argument to the worker."
            )
        use_lock(self.worker_locks[-1], LockAction.Acquire, value=value)

    def _add_worker_lock(self, lock):
        """Register an additional lock in the barrier."""
        self.worker_locks.append(lock)

    def _set_barrier_value(self, value: int):
        """Set the value of the barrier."""
        for lock in self.worker_locks:
            set_lock_value(lock, value)

    def release_with_value(self, value: int):
        """
        Release and decrement the barrier by `value` inside the core.

        Args:
            value (int): The value to decrement by in Release.
        """
        if len(self.worker_locks) == 0:
            raise ValueError(
                "No workers have been registered for this barrier. Need to pass the barrier as an argument to the worker."
            )
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
