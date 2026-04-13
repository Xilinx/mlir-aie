# persistentbuffer.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""PersistentBuffer: static weight store in MemTile, streamed to compute tile via DMA."""

from ..resolvable import Resolvable


class PersistentBufferHandle:
    """Consumer handle passed as Worker fn_arg.

    Provides acquire/release API similar to ObjectFifoHandle.
    When used inside a Worker function body, acquire() returns
    the weight buffer and release() signals the DMA to re-send.
    """

    def __init__(self, pb: "PersistentBuffer"):
        self._pb = pb

    def acquire(self, n: int = 1):
        """Acquire the weight buffer (signals DMA transfer complete)."""
        raise NotImplementedError(
            "PersistentBuffer.acquire() can only be called inside a Worker function body "
            "after the design is resolved."
        )

    def release(self, n: int = 1):
        """Release the weight buffer (signals DMA to re-send for next inference)."""
        raise NotImplementedError(
            "PersistentBuffer.release() can only be called inside a Worker function body "
            "after the design is resolved."
        )


class PersistentBuffer(Resolvable):
    """Static weight store: data lives in MemTile, streamed to compute tile via DMA.

    Replaces the placed-API pattern of:
        buffer(memtile, ..., initial_value=arr)
        + 4x lock()
        + flow()
        + @memtile_dma
        + @mem

    Usage::

        wts = PersistentBuffer(
            np.ndarray[(n,), np.dtype[np.int8]],
            initial_value=weight_array,
            name="my_wts",
        )
        worker = Worker(my_fn, [act_in.cons(), wts.cons(), act_out.prod()])

    Inside the worker function::

        def my_fn(act_in, wts, act_out):
            buf = wts.acquire(1)   # acquire weight buffer
            # ... use buf ...
            wts.release(1)         # release for next inference
    """

    def __init__(self, obj_type, initial_value, name: str):
        """Construct a PersistentBuffer.

        Args:
            obj_type: The numpy ndarray type descriptor for the weight buffer.
            initial_value: Initial weight data (numpy array).
            name (str): Unique name for this buffer.
        """
        self._obj_type = obj_type
        self._initial_value = initial_value
        self._name = name

    def cons(self) -> PersistentBufferHandle:
        """Returns a consumer handle to pass as a Worker fn_arg."""
        return PersistentBufferHandle(self)

    def resolve(self, loc=None, ip=None) -> None:
        """Resolve the PersistentBuffer into MLIR ops.

        Full implementation requires:
        1. Locks on MemTile (prod_lock init=0, cons_lock init=1)
        2. Static buffer on MemTile with initial_value
        3. Locks on compute tile (prod_lock init=1, cons_lock init=0)
        4. Receive buffer on compute tile
        5. DMA flow: MemTile MM2S -> compute S2MM
        6. MemTile DMA region (MM2S, loops forever)
        7. Compute tile DMA region (S2MM, loops forever)
        """
        raise NotImplementedError(
            "PersistentBuffer.resolve() is not yet implemented. "
            "Use the placed AIE dialect API directly for static weight buffers."
        )
