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

from ..tensor_class import Tensor
from aie.helpers.util import np_ndarray_type_get_shape
from .context import HSAContext


class HSATensor(Tensor):
    """Tensor backed by an HSA vmem allocation (CPU+AIE coherent)."""

    def __init__(self, shape_or_data, dtype=np.uint32, device="npu"):
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
        self._handle, self._va, self._alloc_size = self._ctx.vmem_alloc(request)

        count = self.numel()
        buf_type = ctypes.c_char * self._alloc_size
        self._cbuf = buf_type.from_address(self._va)
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

    def __del__(self):
        self._data = None
        self._cbuf = None
        va = getattr(self, "_va", None)
        if va:
            try:
                self._ctx.vmem_free(self._handle, self._va, self._alloc_size)
            except Exception:
                pass
            self._va = None

    def buffer_object(self):
        """Return the device-visible virtual address (int)."""
        return self._va

    def nbytes_alloc(self) -> int:
        return self._alloc_size
