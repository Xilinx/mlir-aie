# worker.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
import contextvars
import sys
from typing import Callable

from .. import ir  # type: ignore
from ..dialects.aie import core, lock, use_lock
from ..dialects.aiex import set_lock_value, LockAction, DMAChannelDir, reconfigure_dma
from ..helpers.dialects.ext.scf import _for as range_
from .device import PlacementTile, AnyComputeTile, Tile
from .dataflow.objectfifo import ObjectFifoHandle, ObjectFifo
from .dataflow.endpoint import ObjectFifoEndpoint
from .kernel import Kernel, ExternalFunction
from .globalbuffer import GlobalBuffer
from .resolvable import Resolvable


class Worker(ObjectFifoEndpoint):
    """_summary_
    Worker is an object that takes a `core_fn` and a set of arguments.
    A Worker must be placed on a Compute Core.
    """

    """This variable is the current core if resolving() within the Worker, or None otherwise."""
    current_core_placement = contextvars.ContextVar(
        "current_core_placement", default=None
    )

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

        Raises:
            ValueError: Parameters are validated.
        """
        self._tile = placement
        self._while_true = while_true
        self.stack_size = stack_size
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
        self.link_with: str | None = None
        self.fn_args = fn_args
        bin_names = set()
        self._fifos = []
        self._buffers = []
        self._barriers = []

        # Check arguments to the core. Some information is saved for resolution.
        for arg in self.fn_args:
            if isinstance(arg, (Kernel, ExternalFunction)):
                bin_names.add(arg.bin_name)
            elif isinstance(arg, ObjectFifoHandle):
                arg.endpoint = self
                self._fifos.append(arg)
            elif isinstance(arg, GlobalBuffer):
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
            # We assume other arguments are metaprogramming (e.g, Python args)
            # This could allow some errors to sink through, but we allow it for now.
            # TODO: this could be cleaned up through creation of a MetaArgs struct, so you
            # could access values through meta.my_var within the function.

        if len(bin_names) > 1:
            raise ValueError(
                f"Currently, only one binary per works is supported. Found: {bin_names}"
            )
        if len(bin_names) == 1:
            self.link_with = list(bin_names)[0]

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
    def buffers(self) -> list[GlobalBuffer]:
        """Returns a list of GlobalBuffers given to the Worker via fn_args.

        Returns:
            list[GlobalBuffer]: GlobalBuffers used by the Worker.
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
        my_link = self.link_with

        # Create the necessary locks for the core operation to synchronize with the runtime sequence
        # and register them in the corresponding barriers.
        for barrier in self._barriers:
            l = lock(my_tile)
            barrier._add_worker_lock(l)

        # Set the current_core_placement context variable to the current placement.
        # If there are objects within a core_fn that that need a placement, they can
        # query this value, e.g., Worker.current_core_placement.get()
        self.current_core_placement.set(my_tile)

        @core(my_tile, link_with=my_link, stack_size=self.stack_size)
        def core_body():
            for _ in range_(sys.maxsize) if self._while_true else range(1):
                self.core_fn(*self.fn_args)

        # Once we are done resolving the core, remove the placement context information
        self.current_core_placement.set(None)


class WorkerRuntimeBarrier:
    """A barrier allowing individual workers to synchronize with the runtime sequence."""

    def __init__(self, initial_value: int = 0):
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
    def __init__(self, obj: ObjectFifoHandle, address: int, value: int):
        self.obj: ObjectFifoHandle = obj
        self.address: int = address
        self.value: int = value

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        # Determine the tile from the ObjectFifoHandle
        # If it's a producer, use the producer tile; if consumer, use the consumer tile
        if self.obj._is_prod:
            tile = self.obj._object_fifo._prod._endpoint._tile
        else:
            tile = self.obj._object_fifo._cons[0]._endpoint._tile
        set_write32(self.obj, tile, self.address, self.value)

class _BarrierSetOp(Resolvable):
    """A resolvable instance of a WorkerRuntimeBarrier. This should not be used directly."""

    def __init__(self, barrier: WorkerRuntimeBarrier, value: int):
        self.barrier: WorkerRuntimeBarrier = barrier
        self.value: int = value

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        self.barrier._set_barrier_value(self.value)

class ConfigureDMATask(Resolvable):
    """A resolvable instance of a ConfigureDMATask. This should not be used directly."""

    def __init__(
        self,
        obj: ObjectFifoHandle,
        length: int = None,
        offset: int = 0,
        repeat: int = 0,
        sizes: list[int] = None,
        strides: list[int] = None,
        pad_before: list[int] = None,
        pad_after: list[int] = None,
    ):
        self.obj: ObjectFifoHandle = obj
        self.length = length
        self.offset = offset
        self.repeat = repeat
        self.sizes = sizes
        self.strides = strides
        self.pad_before = pad_before
        self.pad_after = pad_after

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        # Determine the tile from the ObjectFifoHandle
        # If it's a producer, use the producer tile; if consumer, use the consumer tile
        if self.obj._is_prod:
            tile = self.obj._object_fifo._prod._endpoint._tile
        else:
            try:
                tile = self.obj._object_fifo._cons[0]._endpoint._tile
            except Exception as e:
                raise RuntimeError("Consumer tile is not available; placer may not have run yet.") from e
        # Set the direction based on producer or consumer side
        if self.obj._is_prod:
            direction = DMAChannelDir.MM2S
        else:
            direction = DMAChannelDir.S2MM

        # If any of the optional attributes are not given, pull from the obj handle
        if self.length is not None:
            length = self.length
        else:
            # Try to get length from obj, else from obj_type shape
            length = getattr(self.obj, "length", None)
            if length is None:
                # Try to get from obj_type shape (assume last dimension is length)
                obj_type = getattr(self.obj, "obj_type", None)
                # In IRON, obj_type is typically a typing-annotated np.ndarray, e.g. np.ndarray[(line_size,), np.dtype[np.int32]]
                # The shape is in obj_type.__args__[0] if using typing-extensions or similar
                shape = None
                if obj_type is not None:
                    args = getattr(obj_type, "__args__", None)
                    if args and len(args) > 0:
                        shape = args[0]
                if shape is not None and hasattr(shape, "__len__") and len(shape) > 0:
                    length = shape[-1]
                else:
                    raise RuntimeError("Failed to get length from objectFifo: obj_type.shape is empty or not a sequence.")
        offset = self.offset if self.offset is not None else getattr(self.obj, "offset", 0)
        repeat = self.repeat if self.repeat is not None else getattr(self.obj, "repeat", 0)
        if self.sizes is not None and self.strides is not None:
            sizes = self.sizes
            strides = self.strides
        else:
            # Determine which dims to use based on producer/consumer (channel direction)
            if self.obj._is_prod:
                dims = getattr(self.obj._object_fifo, "dims_to_stream", None)
            else:
                dims = getattr(self.obj._object_fifo, "default_dims_from_stream_per_cons", None)
            # If dims is a list of tuples like [(size0, stride0), (size1, stride1), ...]
            if dims is not None and len(dims) > 0:
                # Unzip the list of tuples into sizes and strides
                sizes, strides = zip(*dims)
                sizes = list(sizes)
                strides = list(strides)
            else:
                # If dims is None or empty, set sizes and strides to zeros
                sizes = [0, 0, 0]
                strides = [0, 0, 0]
        pad_before = self.pad_before if self.pad_before is not None else [0, 0, 0]
        pad_after = self.pad_after if self.pad_after is not None else [0, 0, 0]

        reconfigure_dma(self.obj, tile, direction, length, offset, repeat, sizes, strides, pad_before, pad_after)