# tensor.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""HSA-backed Tensor: a single vmem allocation mapped coherently for CPU + AIE.

The vmem mapping is accessible to both the CPU and AIE agents, so the numpy
view over the mapped virtual address is directly readable/writable by the host
and by device dispatches. Coherence needs no host staging copy, so the sync
hooks are no-ops.
"""

import ctypes

import numpy as np
from aie.helpers.util import np_ndarray_type_get_shape

from ..tensor_class import Tensor
from .context import HSAContext


class _VmemMapping:
    """Sole owner of one vmem allocation, released when the last reference drops.

    Kept separate from :class:`HSATensor` so the numpy views handed out by
    ``numpy()`` / ``data`` / ``to_torch()`` can keep the mapping alive. Those
    views are ``np.frombuffer`` over a ctypes array built with ``from_address``,
    which owns nothing -- so with the free tied to the tensor instead, dropping
    the tensor while an array still referenced it unmapped the range under that
    array, leaving the caller reading freed virtual addresses.
    """

    def __init__(self, ctx, size):
        self._ctx = ctx
        self.handle, self.va, self.size = ctx.vmem_alloc(size)

    def __del__(self):
        va = getattr(self, "va", None)
        if va:
            try:
                self._ctx.vmem_free(self.handle, va, self.size)
            except Exception:
                pass
            self.va = None


class HSATensor(Tensor):
    """Tensor backed by an HSA vmem allocation (CPU+AIE coherent)."""

    def __init__(self, shape_or_data, dtype=np.uint32, device="npu", **kwargs):
        """Allocate a coherent vmem buffer and wrap it as a tensor.

        Args:
            shape_or_data: Either a shape ``tuple`` to allocate a zero-filled
                buffer, or an array-like (anything with a ``shape``, or something
                ``numpy.asarray`` accepts) whose contents are copied in.
            dtype (numpy.dtype, optional): Element type used when
                ``shape_or_data`` is a shape or a plain sequence. Defaults to
                ``numpy.uint32``.
            device (str, optional): Initial residency. Defaults to ``"npu"``.
            **kwargs: Accepted and ignored, for API compatibility with the other
                tensor backends -- the tensor factories forward backend-specific
                keywords (XRT's ``flags``/``group_id``) to whichever class is
                selected, and HSA has no analogue for them.
        """
        super().__init__(shape_or_data, dtype=dtype, device=device)
        self._ctx = HSAContext.get()

        np_data = None
        if isinstance(shape_or_data, tuple):
            # Subscripting ndarray here is a runtime trick to validate
            # "ShapeLike"-ness; only the shape is read back out. Same line as the
            # XRT and HRX tensors, which escape the check only because their
            # unresolvable backend imports leave pyright with Unknown types.
            np_type = np.ndarray[
                shape_or_data,
                np.dtype[dtype],  # pyright: ignore[reportInvalidTypeArguments]
            ]
            self._shape = np_ndarray_type_get_shape(np_type)
        elif hasattr(shape_or_data, "shape"):
            self._shape = shape_or_data.shape
            np_data = shape_or_data
        else:
            np_data = np.asarray(shape_or_data, dtype=dtype)
            self._shape = np_data.shape

        # vmem rejects zero-size; keep a 1-byte floor (designs never use 0-size IO).
        request = max(self.nbytes, 1)
        self._mapping = _VmemMapping(self._ctx, request)
        self._va = self._mapping.va
        self._alloc_size = self._mapping.size

        count = self.numel()
        # `from_address` yields a view that owns nothing, so the mapping has to
        # stay reachable by another route: numpy records this buffer instance as
        # the array's `base`, so hanging the mapping off it gives the chain
        # array -> buffer -> mapping. That is what keeps the range mapped for as
        # long as any array over it is alive, even once this tensor is gone.
        buf_type = ctypes.c_char * self._alloc_size
        self._cbuf = buf_type.from_address(self._va)
        self._cbuf._mapping = self._mapping
        self._data = np.frombuffer(self._cbuf, dtype=self.dtype, count=count).reshape(
            self._shape
        )

        if np_data is not None:
            np.copyto(self._data, np_data)
        else:
            self._data.fill(0)

    @property
    def data(self):
        assert self._data is not None
        return self._data

    @property
    def shape(self):
        return self._shape

    def _sync_to_device(self):
        # Coherent vmem mapping: nothing to do.
        pass

    def _sync_from_device(self):
        # Coherent vmem mapping: nothing to do.
        pass

    def buffer_object(self):
        """Return the device-visible virtual address (int)."""
        return self._va

    def nbytes_alloc(self) -> int:
        return self._alloc_size
